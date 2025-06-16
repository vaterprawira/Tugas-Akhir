import matplotlib
matplotlib.use('Agg')
from flask import Flask, request, jsonify
import mysql.connector
from datetime import datetime, timedelta
from flask_cors import CORS
import numpy as np
import random
import copy
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import io
import base64
import logging
from enum import Enum

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class TariffType(Enum):
    WITH_DEMAND = "With Demand Charge"
    NO_DEMAND = "No Demand Charge"

def get_db_connection():
    try:
        return mysql.connector.connect(
            host="127.0.0.1",
            port=3306,
            user="dsm",
            password="DSM#Pr0j3ct#2025#db!",
            database="dsmdb"
        )
    except mysql.connector.Error as err:
        logger.error(f"Database connection failed: {err}")
        raise

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor(buffered=True)

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS products (
            product_id INT PRIMARY KEY AUTO_INCREMENT,
            product_name VARCHAR(50) NOT NULL
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS simulations (
            simulation_id INT PRIMARY KEY AUTO_INCREMENT,
            tariff_type VARCHAR(50) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS processes (
            process_id INT PRIMARY KEY AUTO_INCREMENT,
            simulation_id INT NOT NULL,
            process_name VARCHAR(100) NOT NULL,
            process_order INT NOT NULL,
            FOREIGN KEY (simulation_id) REFERENCES simulations(simulation_id) ON DELETE CASCADE
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS machines (
            machine_id INT PRIMARY KEY AUTO_INCREMENT,
            process_id INT NOT NULL,
            machine_name VARCHAR(100) NOT NULL,
            quantity INT NOT NULL,
            power DECIMAL(10, 2) NOT NULL,
            start_time TIME NOT NULL,
            stop_time TIME NOT NULL,
            FOREIGN KEY (process_id) REFERENCES processes(process_id) ON DELETE CASCADE
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS optimization_results (
            result_id INT PRIMARY KEY AUTO_INCREMENT,
            simulation_id INT NOT NULL,
            total_cost DECIMAL(15, 2) NOT NULL,
            peak_load DECIMAL(10, 2) NOT NULL,
            total_energy DECIMAL(10, 2) NOT NULL,
            demand_cost DECIMAL(15, 2) NOT NULL,
            energy_cost DECIMAL(15, 2) NOT NULL,
            load_profile TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (simulation_id) REFERENCES simulations(simulation_id) ON DELETE CASCADE
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS optimized_schedules (
            schedule_id INT PRIMARY KEY AUTO_INCREMENT,
            result_id INT NOT NULL,
            process_id INT NOT NULL,
            machine_id INT NOT NULL,
            start_time TIME NOT NULL,
            duration INT NOT NULL,
            FOREIGN KEY (result_id) REFERENCES optimization_results(result_id) ON DELETE CASCADE,
            FOREIGN KEY (process_id) REFERENCES processes(process_id) ON DELETE CASCADE,
            FOREIGN KEY (machine_id) REFERENCES machines(machine_id) ON DELETE CASCADE
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ga_parameters (
            ga_param_id INT PRIMARY KEY AUTO_INCREMENT,
            simulation_id INT NOT NULL,
            population_size INT NOT NULL,
            crossover_rate DECIMAL(5, 4) NOT NULL,
            mutation_rate DECIMAL(5, 4) NOT NULL,
            tournament_size INT NOT NULL,
            elitism_rate DECIMAL(5, 4) NOT NULL,
            FOREIGN KEY (simulation_id) REFERENCES simulations(simulation_id) ON DELETE CASCADE
        )
    ''')

    cursor.execute("INSERT IGNORE INTO products (product_id, product_name) VALUES (1, 'Wafer Cream')")
    cursor.execute("INSERT IGNORE INTO products (product_id, product_name) VALUES (2, 'Wafer Stick')")

    conn.commit()
    cursor.close()
    conn.close()

HOURS_IN_DAY = 24
POPULATION_SIZE = 300
MAX_GENERATIONS = 1000
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.2
TOURNAMENT_SIZE = 3
ELITISM_RATE = 0.05
STAGNATION_LIMIT = 200
FITNESS_THRESHOLD = 0.000001
MAX_DELAY = 3
ENERGY_TOLERANCE = 0.05

def parse_start_hour(start_time):
    """Parse start_time from various formats to hour integer."""
    if isinstance(start_time, str):
        try:
            return int(start_time.split(':')[0])
        except (ValueError, IndexError):
            logger.warning(f"Invalid time string format: {start_time}")
            return 0
    elif hasattr(start_time, 'hour'):
        return start_time.hour
    elif hasattr(start_time, 'seconds'):
        return int(start_time.seconds // 3600)
    logger.warning(f"Unknown start_time format: {start_time}")
    return 0

class Machine:
    def __init__(self, name, power_kw, count, operation_hours, machine_id=None):
        self.name = name
        self.power_kw = power_kw
        self.count = count
        self.connected_load = power_kw * count
        self.operation_hours = operation_hours
        self.is_operating = True
        self.machine_id = machine_id
    
    def get_load_profile(self, start_hour):
        profile = np.zeros(HOURS_IN_DAY)
        if self.is_operating:
            adjusted_load = self.connected_load
            for h in range(self.operation_hours):
                hour_index = (start_hour + h) % HOURS_IN_DAY
                profile[hour_index] = adjusted_load
        return profile

class ProductionProcess:
    def __init__(self, name, machines, process_id=None):
        self.name = name
        self.machines = machines if isinstance(machines, list) else [machines]
        self.process_id = process_id

class Tariff:
    def __init__(self):
        self.peak_hours = list(range(17, 22))
        self.off_peak_hours = [h for h in range(24) if h not in self.peak_hours]
        self.lwbp_rate = 0.064
        self.peak_factor = 1.4
        self.demand_charge = 8.69
    
    def get_rate(self):
        return self.lwbp_rate
    
    def calculate_peak_rate(self, hour):
        return self.lwbp_rate * self.peak_factor if hour in self.peak_hours else self.lwbp_rate
    
    def calculate_demand_cost(self, peak_load, tariff_type: TariffType):
        return peak_load * self.demand_charge if tariff_type == TariffType.WITH_DEMAND else 0.0
    
    def calculate_energy_cost(self, load_profile):
        wbp_energy = sum(load_profile[h] * self.calculate_peak_rate(h) for h in self.peak_hours)
        lwbp_energy = sum(load_profile[h] * self.get_rate() for h in self.off_peak_hours)
        return wbp_energy, lwbp_energy
    
    def calculate_total_cost(self, load_profile, tariff_type: TariffType):
        peak_load = max(load_profile)
        demand_cost = self.calculate_demand_cost(peak_load, tariff_type)
        wbp_cost, lwbp_cost = self.calculate_energy_cost(load_profile)
        total_energy_cost = (wbp_cost + lwbp_cost) * 30
        return {
            'demand_cost': demand_cost,
            'wbp_energy': wbp_cost,
            'lwbp_energy': lwbp_cost,
            'total_energy_cost': total_energy_cost,
            'peak_load': peak_load,
            'total_cost': demand_cost + total_energy_cost,
            'total_energy': np.sum(load_profile)
        }

def initialize_system_from_db(simulation_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        cursor.execute('''
            SELECT p.process_id, p.process_name, p.process_order, 
                   m.machine_id, m.machine_name, m.quantity, m.power, 
                   TIME_TO_SEC(TIMEDIFF(m.stop_time, m.start_time)) / 3600 AS operation_hours
            FROM processes p
            LEFT JOIN machines m ON p.process_id = m.process_id
            WHERE p.simulation_id = %s
            ORDER BY p.process_order
        ''', (simulation_id,))
        processes_data = cursor.fetchall()
        
        processes = {}
        for row in processes_data:
            process_name = row['process_name']
            if process_name not in processes:
                processes[process_name] = []
            if row['machine_id']:
                operation_hours = int(row['operation_hours']) if row['operation_hours'] else 1
                machine = Machine(
                    row['machine_name'],
                    float(row['power']),
                    row['quantity'],
                    operation_hours,
                    machine_id=row['machine_id']
                )
                processes[process_name].append(machine)
        
        production_line = [ProductionProcess(name, machines, process_id=row['process_id']) 
                         for name, machines in processes.items()]
        
        return {"selected_product": production_line}
    
    finally:
        cursor.close()
        conn.close()

class Chromosome:
    def __init__(self, production_lines, original_peak_load, original_load):
        self.production_lines = production_lines
        self.original_peak_load = original_peak_load
        self.original_load = original_load
        self.fitness = 0
        self.load_profile = np.zeros(HOURS_IN_DAY)
        self.genes = None
    
    def get_process_start_times(self):
        start_times = {"selected_product": {}}
        delays = self.genes["delays"]
        
        synced_processes = [
            "Wafer Cream - Baking",
            "Wafer Cream - Cooling Sheet and Conditioning", 
            "Wafer Cream - Spreading and Stacking",
            "Wafer Cream - Sandwich Cooling",
            "Wafer Cream - Cutting"
        ]
        
        batter_cream_idx = None
        batter_stick_idx = None
        
        for idx, process in enumerate(self.production_lines["selected_product"]):
            if process.name == "Wafer Cream - Batter Preparation":
                batter_cream_idx = idx
            elif process.name == "Wafer Stick - Batter Preparation":
                batter_stick_idx = idx
        
        if batter_cream_idx is not None:
            start_times["selected_product"]["Wafer Cream - Batter Preparation"] = self.genes["start_times"][batter_cream_idx]
            if batter_stick_idx is not None:
                start_times["selected_product"]["Wafer Stick - Batter Preparation"] = self.genes["start_times"][batter_cream_idx]
        elif batter_stick_idx is not None:
            start_times["selected_product"]["Wafer Stick - Batter Preparation"] = self.genes["start_times"][batter_stick_idx]
        
        sync_start_time = None
        
        for idx, curr_process in enumerate(self.production_lines["selected_product"]):
            if curr_process.name in ["Wafer Cream - Batter Preparation", "Wafer Stick - Batter Preparation"]:
                continue
            
            if idx == 0:
                start_times["selected_product"][curr_process.name] = self.genes["start_times"][idx]
                if curr_process.name in synced_processes:
                    sync_start_time = self.genes["start_times"][idx]
            else:
                prev_process = None
                prev_start = None
                
                for prev_idx in range(idx - 1, -1, -1):
                    candidate_process = self.production_lines["selected_product"][prev_idx]
                    if candidate_process.name in start_times["selected_product"]:
                        prev_process = candidate_process
                        prev_start = start_times["selected_product"][candidate_process.name]
                        break
                
                if prev_process is None or prev_start is None:
                    start_times["selected_product"][curr_process.name] = self.genes["start_times"][idx]
                    if curr_process.name in synced_processes and sync_start_time is None:
                        sync_start_time = self.genes["start_times"][idx]
                    continue
                
                prev_duration = max(m.operation_hours for m in prev_process.machines)
                prev_end = (prev_start + prev_duration) % HOURS_IN_DAY
                delay = delays[max(0, idx - 1)]
                
                if curr_process.name in synced_processes:
                    if sync_start_time is None:
                        actual_delay = min(max(1, delay), MAX_DELAY)
                        start_hour = (prev_end + actual_delay) % HOURS_IN_DAY
                        sync_start_time = start_hour
                    else:
                        start_hour = sync_start_time
                else:
                    actual_delay = min(max(1, delay), MAX_DELAY)
                    start_hour = (prev_end + actual_delay) % HOURS_IN_DAY
                
                start_times["selected_product"][curr_process.name] = start_hour
        
        return start_times
    
    def calculate_load_profile(self):
        self.load_profile = np.zeros(HOURS_IN_DAY)
        start_times = self.get_process_start_times()
        for process in self.production_lines["selected_product"]:
            start_hour = start_times["selected_product"][process.name]
            for machine in process.machines:
                self.load_profile += machine.get_load_profile(start_hour)
        return self.load_profile
    
    def calculate_fitness(self, tariff, tariff_type: TariffType):
        """Calculate fitness based on cost, peak load, and scheduling constraints."""
        load_profile = self.calculate_load_profile()
        cost_details = tariff.calculate_total_cost(load_profile, tariff_type)
        
        start_times = self.get_process_start_times()
        sequence_violations = 0
        delay_violations = 0
        
        synced_processes = [
            "Wafer Cream - Baking",
            "Wafer Cream - Cooling Sheet and Conditioning",
            "Wafer Cream - Spreading and Stacking", 
            "Wafer Cream - Sandwich Cooling",
            "Wafer Cream - Cutting"
        ]
        
        processes = self.production_lines["selected_product"]
        
        for i in range(len(processes) - 1):
            current_process = processes[i]
            next_process = processes[i + 1]
            
            current_start = start_times["selected_product"].get(current_process.name)
            next_start = start_times["selected_product"].get(next_process.name)
            
            if current_start is None or next_start is None:
                continue
            
            if (current_process.name in synced_processes and next_process.name in synced_processes):
                if current_start != next_start:
                    sequence_violations += 1  
                continue
            
            current_duration = max(m.operation_hours for m in current_process.machines)
            current_end = (current_start + current_duration) % HOURS_IN_DAY
            
            if current_end <= next_start:
                time_gap = next_start - current_end
            else:
                time_gap = (HOURS_IN_DAY - current_end) + next_start
            
            if time_gap < 1:
                sequence_violations += 1
            if time_gap > MAX_DELAY:
                delay_violations += 1
        
        for i in range(1, len(processes)):
            current_process = processes[i]
            
            predecessor = None
            predecessor_end = None
            
            if current_process.name in synced_processes:
                for j in range(i - 1, -1, -1):
                    candidate = processes[j]
                    if candidate.name not in synced_processes:
                        predecessor = candidate
                        pred_start = start_times["selected_product"].get(candidate.name)
                        if pred_start is not None:
                            pred_duration = max(m.operation_hours for m in candidate.machines)
                            predecessor_end = (pred_start + pred_duration) % HOURS_IN_DAY
                        break
            else:
                for j in range(i - 1, -1, -1):
                    candidate = processes[j]
                    candidate_start = start_times["selected_product"].get(candidate.name)
                    if candidate_start is not None:
                        predecessor = candidate
                        pred_duration = max(m.operation_hours for m in candidate.machines)
                        predecessor_end = (candidate_start + pred_duration) % HOURS_IN_DAY
                        break
            
            if predecessor is not None and predecessor_end is not None:
                current_start = start_times["selected_product"].get(current_process.name)
                if current_start is not None:
                    if predecessor_end <= current_start:
                        time_gap = current_start - predecessor_end
                    else:
                        time_gap = (HOURS_IN_DAY - predecessor_end) + current_start
                    
                    if time_gap > MAX_DELAY:
                        delay_violations += 2 
        
        sync_violations = 0
        sync_start_time = None
        for process in processes:
            if process.name in synced_processes:
                process_start = start_times["selected_product"].get(process.name)
                if process_start is not None:
                    if sync_start_time is None:
                        sync_start_time = process_start
                    elif process_start != sync_start_time:
                        sync_violations += 1
        
        optimized_peak = cost_details['peak_load']
        peak_excess = max(0, optimized_peak - (self.original_peak_load or 1))
        wbp_load = sum(load_profile[h] for h in tariff.peak_hours)
        original_cost = tariff.calculate_total_cost(self.original_load, tariff_type)['total_cost']
        MAX_WBP_LOAD = self.original_peak_load * len(tariff.peak_hours)
        expected_total_energy = sum(sum(m.connected_load * m.operation_hours for m in p.machines) 
                                   for p in self.production_lines["selected_product"])
        actual_total_energy = np.sum(load_profile)
        energy_deviation = abs(actual_total_energy - expected_total_energy) / (expected_total_energy or 1)
        
        norm_violations = min(sequence_violations, 15) / 15 
        norm_peak_excess = min(peak_excess / (self.original_peak_load or 1), 1)
        norm_wbp = min(wbp_load / (MAX_WBP_LOAD or 1), 1)
        norm_cost = min(cost_details['total_cost'] / (original_cost or 1), 1)
        norm_energy_deviation = min(energy_deviation, 1)
        norm_delay_violations = min(delay_violations, 10) / 10 
        norm_sync_violations = min(sync_violations, 5) / 5
        
        objective = (
            0.3 * norm_violations +
            0.3 * norm_peak_excess +
            0.3 * norm_wbp +
            0.3 * norm_cost +
            0.3 * norm_energy_deviation +
            0.8 * norm_delay_violations + 
            0.6 * norm_sync_violations
        )
        
        self.fitness = 1 / (objective + 1e-6)
        return self.fitness
    
    def debug_violations(self):
        """Debug scheduling violations and return details."""
        debug_info = []
        start_times = self.get_process_start_times()
        processes = self.production_lines["selected_product"]
        
        debug_info.append("=== VIOLATION DEBUG ===")
        debug_info.append("Process Start Times:")
        for process in processes:
            start_time = start_times["selected_product"].get(process.name, "N/A")
            debug_info.append(f"  {process.name}: {start_time}")
        
        debug_info.append("\nSequential Process Analysis:")
        for i in range(len(processes) - 1):
            current = processes[i]
            next_proc = processes[i + 1]
            
            current_start = start_times["selected_product"].get(current.name)
            next_start = start_times["selected_product"].get(next_proc.name)
            
            if current_start is not None and next_start is not None:
                current_duration = max(m.operation_hours for m in current.machines)
                current_end = (current_start + current_duration) % HOURS_IN_DAY
                
                if current_end <= next_start:
                    time_gap = next_start - current_end
                else:
                    time_gap = (HOURS_IN_DAY - current_end) + next_start
                
                violation = ""
                if time_gap < 1:
                    violation += "TOO_SHORT "
                if time_gap > MAX_DELAY:
                    violation += "TOO_LONG "
                
                debug_info.append(f"  {current.name} -> {next_proc.name}")
                debug_info.append(f"    End: {current_end}, Start: {next_start}, Gap: {time_gap}h {violation}")
        
        debug_info.append("=== END DEBUG ===")
        logger.debug("\n".join(debug_info))
        return debug_info

def setup_deap(production_lines, original_peak_load, original_load, tariff, tariff_type: TariffType):
    """Set up DEAP toolbox for genetic algorithm."""
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", dict, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    def init_individual():
        num_processes = len(production_lines["selected_product"])
        return creator.Individual({
            "start_times": [random.randint(0, 23) for _ in range(num_processes)],
            "delays": [random.randint(1, MAX_DELAY) for _ in range(num_processes - 1)]  
        })
    
    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    def evaluate(individual):
        chromosome = Chromosome(production_lines, original_peak_load, original_load)
        chromosome.genes = individual
        return (chromosome.calculate_fitness(tariff, tariff_type),)
    
    def cx_custom(ind1, ind2):
        if random.random() > CROSSOVER_RATE:
            return ind1, ind2
        child1, child2 = copy.deepcopy(ind1), copy.deepcopy(ind2)
        if random.random() < 0.5:
            cx_point = random.randint(1, len(ind1["start_times"]) - 1)
            child1["start_times"][cx_point:], child2["start_times"][cx_point:] = (
                child2["start_times"][cx_point:], child1["start_times"][cx_point:]
            )
        else:
            cx_point = random.randint(1, len(ind1["delays"]) - 1)
            child1["delays"][cx_point:], child2["delays"][cx_point:] = (
                child2["delays"][cx_point:], child1["delays"][cx_point:]
            )
        return child1, child2
    
    def mut_custom(individual):
        if random.random() > MUTATION_RATE:
            return (individual,)
        mutated = copy.deepcopy(individual)
        
        mutation_type = random.random()
        
        if mutation_type < 0.4:  
            gene_index = random.randint(0, len(mutated["start_times"]) - 1)
            if random.random() < 0.5:
                mutated["start_times"][gene_index] = random.randint(0, 23)
            else:
                shift = random.randint(-3, 3)
                mutated["start_times"][gene_index] = (mutated["start_times"][gene_index] + shift) % 24
        
        elif mutation_type < 0.8:   
            gene_index = random.randint(0, len(mutated["delays"]) - 1)
            mutated["delays"][gene_index] = random.randint(1, MAX_DELAY)
        
        else:  
            num_mutations = random.randint(1, min(3, len(mutated["start_times"])))
            for _ in range(num_mutations):
                if random.random() < 0.5:
                    gene_index = random.randint(0, len(mutated["start_times"]) - 1)
                    mutated["start_times"][gene_index] = random.randint(0, 23)
                else:
                    gene_index = random.randint(0, len(mutated["delays"]) - 1)
                    mutated["delays"][gene_index] = random.randint(1, MAX_DELAY)
        
        return (mutated,)
    
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", cx_custom)
    toolbox.register("mutate", mut_custom)
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)
    
    return toolbox

def genetic_algorithm(production_lines, tariff, original_load, tariff_type: TariffType, 
                     population_size=POPULATION_SIZE, crossover_rate=CROSSOVER_RATE, 
                     mutation_rate=MUTATION_RATE, tournament_size=TOURNAMENT_SIZE, 
                     elitism_rate=ELITISM_RATE):
    """Run genetic algorithm with configurable parameters."""
    import time
    random.seed(int(time.time() * 1000) % 2**32)
    np.random.seed(int(time.time() * 1000) % 2**32)
    
    original_peak_load = max(original_load)
    toolbox = setup_deap(production_lines, original_peak_load, original_load, tariff, tariff_type)
    
    population = toolbox.population(n=population_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    
    hof = tools.HallOfFame(1)
    best_fitness_history = []
    stagnation_counter = 0
    previous_fitness = None
    
    try:
        population, logbook = algorithms.eaMuPlusLambda(
            population,
            toolbox,
            mu=int(population_size * (1 - elitism_rate)),
            lambda_=int(population_size * elitism_rate),
            cxpb=crossover_rate,
            mutpb=mutation_rate,
            ngen=MAX_GENERATIONS,
            stats=stats,
            halloffame=hof,
            verbose=False
        )
        
        for gen, stat in enumerate(logbook):
            current_fitness = stat["max"]
            best_fitness_history.append(current_fitness)
            if previous_fitness is not None:
                fitness_change = abs(current_fitness - previous_fitness)
                if fitness_change < FITNESS_THRESHOLD:
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0
                if stagnation_counter >= STAGNATION_LIMIT:
                    break
            previous_fitness = current_fitness
        
        best_individual = hof.items[0]
        best_chromosome = Chromosome(production_lines, original_peak_load, original_load)
        best_chromosome.genes = best_individual
        best_chromosome.fitness = best_individual.fitness.values[0]
        
        return best_chromosome, best_fitness_history
    
    except Exception as e:
        logger.error(f"Genetic algorithm failed: {e}")
        raise

def get_original_schedule(simulation_id):
    """Fetch original schedule from database."""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        cursor.execute('''
            SELECT p.process_name, m.machine_id, m.machine_name, m.quantity, m.power, m.start_time, m.stop_time,
                   TIME_TO_SEC(TIMEDIFF(m.stop_time, m.start_time)) / 3600 AS operation_hours
            FROM processes p
            JOIN machines m ON p.process_id = m.process_id
            WHERE p.simulation_id = %s
            ORDER BY p.process_order, m.machine_name
        ''', (simulation_id,))
        machines_data = cursor.fetchall()
        
        original_schedule = {}
        for data in machines_data:
            process_name = data['process_name']
            start_time = data['start_time']
            stop_time = data['stop_time']
            
            start_hour = parse_start_hour(start_time)
            duration = int(data['operation_hours']) if data['operation_hours'] else 1
            
            if process_name not in original_schedule:
                original_schedule[process_name] = {'machines': []}
            
            original_schedule[process_name]['machines'].append({
                'machine_id': data['machine_id'],
                'machine_name': data['machine_name'],
                'quantity': data['quantity'],
                'power': float(data['power']),
                'start_hour': start_hour,
                'duration': duration
            })
        
        return original_schedule
    
    finally:
        cursor.close()
        conn.close()

process_order_map = {
    "Wafer Cream": [
        "Batter Preparation",
        "Baking",
        "Cooling Sheet and Conditioning",
        "Spreading and Stacking",
        "Sandwich Cooling",
        "Cutting",
        "Enrobing",
        "Packaging"
    ],
    "Wafer Stick": [
        "Batter Preparation",
        "Wafer Stick",
        "Packaging"
    ]
}

def get_process_sort_key(process_name):
    product, step = process_name.split(' - ', 1)
    order_list = process_order_map.get(product, [])
    return (product, order_list.index(step) if step in order_list else 99)

def plot_results(original_load, best_solution, tariff, scenario_name, simulation_id, original_schedule, 
                 tariff_type: TariffType, optimized_schedule):
    """Generate visualization of optimization results."""
    optimized_load = best_solution.calculate_load_profile()
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    for h in tariff.peak_hours:
        ax1.axvspan(h, h+1, color='red', alpha=0.1, label='On-Peak Hours' if h == tariff.peak_hours[0] else "")

    hours = list(range(24))
    ax1.plot(hours, original_load, 'r-', label='Original Load', linewidth=2)
    ax1.plot(hours, optimized_load, 'b--', label=f'Optimized Load ({scenario_name})', linewidth=2)
    
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Load (kW)')
    ax1.set_title(f'Load Profile Comparison - {scenario_name}')
    ax1.legend()
    ax1.grid(True)
    ax1.set_xticks(hours)
    
    y_pos = 0
    y_labels = []
    sorted_processes = sorted(
        best_solution.production_lines["selected_product"],
        key=lambda p: get_process_sort_key(p.name)
    )

    for process in sorted_processes:
        process_name = process.name
        machines = original_schedule.get(process_name, {'machines': []})['machines']
        if not machines:
            machines = [{'machine_id': None, 'machine_name': f'{process_name} (No Machine)', 
                        'quantity': 1, 'power': 0, 'start_hour': 0, 'duration': 1}]
        for machine in machines:
            start_hour = machine['start_hour']
            duration = machine['duration']
            end_hour = (start_hour + duration) % HOURS_IN_DAY
            label = machine['machine_name']

            product_name, process_step = process_name.split(' - ', 1)
            ytick_label = f"{product_name} - {process_step}"

            if start_hour > end_hour:
                ax2.barh(y_pos, HOURS_IN_DAY - start_hour, left=start_hour, height=0.75, 
                         align='center', alpha=0.8, color='gray')
                ax2.barh(y_pos, end_hour, left=0, height=0.75, align='center', alpha=0.8, color='gray')
            else:
                ax2.barh(y_pos, duration, left=start_hour, height=0.75, align='center', 
                         alpha=0.8, color='gray')
            ax2.text((start_hour + end_hour) / 2 % HOURS_IN_DAY, y_pos, label,
                     ha='center', va='center', color='black', fontsize=11)
            y_labels.append(ytick_label)
            y_pos += 1

    ax2.set_yticks(range(len(y_labels)))
    ax2.set_yticklabels(y_labels, fontsize=11)
    ax2.set_xlabel('Hour of Day')
    ax2.set_xlim(0, 24)
    ax2.set_title('Original Machine Production Schedule')
    ax2.grid(True, axis='x')
    ax2.set_xticks(hours)
    
    y_pos = 0
    y_labels = []
    for process in sorted_processes:
        process_name = process.name
        if process_name not in optimized_schedule:
            continue
        process_data = optimized_schedule[process_name]
        start_hour = process_data['start_hour']  
        machines = process_data['machines']

        product_name, process_step = process_name.split(' - ', 1)
        
        for machine in machines:
            duration = machine['duration']
            end_hour = (start_hour + duration) % HOURS_IN_DAY
            label = machine['machine_name']
            
            ytick_label = f"{product_name} - {process_step}"

            if start_hour > end_hour:
                ax3.barh(y_pos, HOURS_IN_DAY - start_hour, left=start_hour, height=0.75, 
                         align='center', alpha=0.8, color='lightblue')
                ax3.barh(y_pos, end_hour, left=0, height=0.75, align='center', alpha=0.8, 
                         color='lightblue')
            else:
                ax3.barh(y_pos, duration, left=start_hour, height=0.75, align='center', 
                         alpha=0.8, color='lightblue')
            ax3.text((start_hour + end_hour) / 2 % HOURS_IN_DAY, y_pos, label,
                     ha='center', va='center', color='black', fontsize=11)
            y_labels.append(ytick_label)
            y_pos += 1

    ax3.set_yticks(range(len(y_labels)))
    ax3.set_yticklabels(y_labels, fontsize=11)
    ax3.set_xlabel('Hour of Day')
    ax3.set_xlim(0, 24)
    ax3.set_title(f'Optimized Machine Production Schedule - {scenario_name}')
    ax3.grid(True, axis='x')
    ax3.set_xticks(hours)
    
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    return plot_data

@app.route('/api/insert_data', methods=['POST'])
def insert_data():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    products_data = data.get('products')
    tariff_type_str = data.get('tariff')

    try:
        tariff_type = TariffType(tariff_type_str)
    except ValueError:
        logger.error(f"Invalid tariff_type received: {tariff_type_str}")
        return jsonify({'error': f'Invalid tariff_type: {tariff_type_str}'}), 400

    if not products_data or not isinstance(products_data, list):
        return jsonify({'error': 'Missing products data'}), 400

    conn = get_db_connection()
    cursor = conn.cursor(buffered=True)

    try:
        cursor.execute('''
            INSERT INTO simulations (tariff_type)
            VALUES (%s)
        ''', (tariff_type.value,))
        conn.commit()
        simulation_id = cursor.lastrowid
        process_mapping = {}

        for product_entry in products_data:
            product_name = product_entry.get('product')
            processes_data = product_entry.get('processes')

            if not product_name or not processes_data:
                return jsonify({'error': 'Missing product or processes data'}), 400

            cursor.execute('SELECT product_id FROM products WHERE product_name = %s', (product_name,))
            product = cursor.fetchone()
            if not product:
                return jsonify({'error': f'Product {product_name} not found'}), 404
            product_id = product[0]

            for process_data in processes_data:
                process_name = process_data.get('process_name')
                machines = process_data.get('machines', [])

                process_order = {
                    'Wafer Cream': {
                        'Batter Preparation': 1,
                        'Baking': 2,
                        'Cooling Sheet and Conditioning': 3,
                        'Spreading and Stacking': 4,
                        'Sandwich Cooling': 5,
                        'Cutting': 6,
                        'Enrobing': 7,
                        'Packaging': 8
                    },
                    'Wafer Stick': {
                        'Batter Preparation': 1,
                        'Wafer Stick': 2,
                        'Packaging': 3
                    }
                }
                order = process_order[product_name].get(process_name)
                if not order:
                    return jsonify({'error': f'Invalid process name {process_name} for product {product_name}'}), 400

                cursor.execute('''
                    INSERT INTO processes (simulation_id, process_name, process_order)
                    VALUES (%s, %s, %s)
                ''', (simulation_id, f"{product_name} - {process_name}", order))
                conn.commit()
                process_id = cursor.lastrowid
                process_mapping[f"{product_name} - {process_name}"] = process_id

                for machine in machines:
                    cursor.execute('''
                        INSERT INTO machines (process_id, machine_name, quantity, power, start_time, stop_time)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    ''', (
                        process_id,
                        machine['machine_name'],
                        machine['quantity'],
                        machine['power'],
                        machine['start_time'],
                        machine['stop_time']
                    ))
                    conn.commit()

        conn.commit()
        return jsonify({'message': 'Data inserted successfully', 'simulation_id': simulation_id}), 200

    except Exception as e:
        conn.rollback()
        logger.error(f"Insert data failed: {e}")
        return jsonify({'error': str(e)}), 500

    finally:
        cursor.close()
        conn.close()

@app.route('/api/simulations', methods=['GET'])
def get_simulations():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        cursor.execute('''
            SELECT s.simulation_id, s.tariff_type, GROUP_CONCAT(p.process_name) as products
            FROM simulations s
            JOIN processes p ON s.simulation_id = p.simulation_id
            GROUP BY s.simulation_id, s.tariff_type
        ''')
        simulations = cursor.fetchall()
        return jsonify(simulations), 200
    
    except Exception as e:
        logger.error(f"Get simulations failed: {e}")
        return jsonify({'error': str(e)}), 500
    
    finally:
        cursor.close()
        conn.close()

def fetch_simulation_data(cursor, simulation_id):
    """Fetch simulation details from database."""
    cursor.execute('SELECT tariff_type FROM simulations WHERE simulation_id = %s', (simulation_id,))
    return cursor.fetchone()

def fetch_ga_parameters(cursor, simulation_id):
    """Fetch GA parameters or return defaults."""
    cursor.execute('''
        SELECT population_size, crossover_rate, mutation_rate, tournament_size, elitism_rate
        FROM ga_parameters WHERE simulation_id = %s
    ''', (simulation_id,))
    params = cursor.fetchone()
    return params or {
        'population_size': POPULATION_SIZE,
        'crossover_rate': CROSSOVER_RATE,
        'mutation_rate': MUTATION_RATE,
        'tournament_size': TOURNAMENT_SIZE,
        'elitism_rate': ELITISM_RATE
    }

def calculate_original_load(cursor, simulation_id):
    """Calculate original load profile from machine data."""
    cursor.execute('''
        SELECT p.process_id, p.process_name, p.process_order, 
               m.machine_id, m.machine_name, m.quantity, m.power, 
               m.start_time, m.stop_time,
               TIME_TO_SEC(TIMEDIFF(m.stop_time, m.start_time)) / 3600 AS operation_hours
        FROM processes p
        JOIN machines m ON p.process_id = m.process_id
        WHERE p.simulation_id = %s
        ORDER BY p.process_order
    ''', (simulation_id,))
    machines_data = cursor.fetchall()
    
    original_load = np.zeros(HOURS_IN_DAY)
    for machine_data in machines_data:
        machine = Machine(
            machine_data['machine_name'],
            float(machine_data['power']),
            machine_data['quantity'],
            int(machine_data['operation_hours']) or 1,
            machine_data['machine_id']
        )
        start_hour = parse_start_hour(machine_data['start_time'])
        original_load += machine.get_load_profile(start_hour)
    
    return original_load, machines_data

def save_optimization_results(cursor, conn, simulation_id, optimized_load, optimized_cost_details, 
                             production_lines, start_times):
    """Save optimization results to database."""
    cursor.execute('''
        INSERT INTO optimization_results (
            simulation_id, total_cost, peak_load, total_energy, 
            demand_cost, energy_cost, load_profile
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
    ''', (
        simulation_id,
        float(optimized_cost_details['total_cost']),
        float(optimized_cost_details['peak_load']),
        float(optimized_cost_details['total_energy']),
        float(optimized_cost_details['demand_cost']),
        float(optimized_cost_details['total_energy_cost']),
        str([float(x) for x in optimized_load.tolist()])
    ))
    result_id = cursor.lastrowid
    
    for process in production_lines['selected_product']:
        cursor.execute('SELECT process_id FROM processes WHERE simulation_id = %s AND process_name = %s',
                      (simulation_id, process.name))
        result = cursor.fetchone()
        if not result:
            continue
        process_id = result['process_id']
        start_hour = start_times['selected_product'][process.name]
        
        cursor.execute('''
            SELECT m.machine_id, TIME_TO_SEC(TIMEDIFF(m.stop_time, m.start_time)) / 3600 AS operation_hours
            FROM machines m
            JOIN processes p ON m.process_id = p.process_id
            WHERE p.simulation_id = %s AND p.process_name = %s
        ''', (simulation_id, process.name))
        machines = cursor.fetchall()
        
        for machine in machines:
            duration = int(machine['operation_hours']) or 1
            cursor.execute('''
                INSERT INTO optimized_schedules (result_id, process_id, machine_id, start_time, duration)
                VALUES (%s, %s, %s, %s, %s)
            ''', (result_id, process_id, machine['machine_id'], f"{int(start_hour):02d}:00:00", duration))
    
    conn.commit()
    return result_id

def build_optimized_schedule(cursor, simulation_id, production_lines, start_times):
    """Build optimized schedule dictionary."""
    optimized_schedule = {}
    for process in production_lines['selected_product']:
        start_hour = start_times['selected_product'][process.name]
        cursor.execute('''
            SELECT m.machine_id, m.machine_name, m.quantity, m.power, 
                   TIME_TO_SEC(TIMEDIFF(m.stop_time, m.start_time)) / 3600 AS operation_hours
            FROM machines m
            JOIN processes p ON m.process_id = p.process_id
            WHERE p.simulation_id = %s AND p.process_name = %s
        ''', (simulation_id, process.name))
        machines = cursor.fetchall()
        
        optimized_schedule[process.name] = {
            'start_hour': start_hour,
            'machines': [
                {
                    'machine_id': machine['machine_id'],
                    'machine_name': machine['machine_name'],
                    'quantity': machine['quantity'],
                    'power': float(machine['power']),
                    'duration': int(machine['operation_hours']) or 1,
                    'start_time': f"{int(start_hour):02d}:00",
                    'start_hour': start_hour,
                    'end_time': f"{int((start_hour + (int(machine['operation_hours']) or 1)) % 24):02d}:00"
                } for machine in machines
            ]
        }
    return optimized_schedule

def build_response(original_load, optimized_load, original_cost_details, optimized_cost_details, 
                  optimized_schedule, plot_data):
    """Build API response with optimization results."""
    original_peak_load = original_cost_details['peak_load']
    optimized_peak_load = optimized_cost_details['peak_load']
    original_total_energy = original_cost_details['total_energy']
    optimized_total_energy = optimized_cost_details['total_energy']
    original_demand_cost = original_cost_details['demand_cost']
    optimized_demand_cost = optimized_cost_details['demand_cost']
    original_energy_cost = original_cost_details['total_energy_cost']
    optimized_energy_cost = optimized_cost_details['total_energy_cost']
    original_total_cost = original_cost_details['total_cost']
    optimized_total_cost = optimized_cost_details['total_cost']
    
    energy_diff = abs(optimized_total_energy - original_total_energy) / (original_total_energy or 1)
    peak_reduction = original_peak_load - optimized_peak_load
    peak_reduction_percent = (peak_reduction / original_peak_load * 100) if original_peak_load > 0 else 0
    cost_savings = original_total_cost - optimized_total_cost
    demand_cost_savings = original_demand_cost - optimized_demand_cost
    demand_cost_savings_percent = (demand_cost_savings / original_demand_cost * 100) if original_demand_cost > 0 else 0
    energy_cost_savings = original_energy_cost - optimized_energy_cost
    energy_cost_savings_percent = (energy_cost_savings / original_energy_cost * 100) if original_energy_cost > 0 else 0
    
    return {
        'summary': {
            'original_peak_load': round(original_peak_load, 2),
            'optimized_peak_load': round(optimized_peak_load, 2),
            'peak_reduction': round(peak_reduction, 2),
            'peak_reduction_percent': round(peak_reduction_percent, 2),
            'original_total_energy': round(original_total_energy, 2),
            'optimized_total_energy': round(optimized_total_energy, 2),
            'energy_deviation_percent': round(energy_diff * 100, 2),
            'original_demand_cost': round(original_demand_cost, 2),
            'optimized_demand_cost': round(optimized_demand_cost, 2),
            'original_energy_cost': round(original_energy_cost, 2),
            'optimized_energy_cost': round(optimized_energy_cost, 2),
            'original_total_cost': round(original_total_cost, 2),
            'optimized_total_cost': round(optimized_total_cost, 2),
            'cost_savings': round(cost_savings, 2),
            'demand_cost_savings': round(demand_cost_savings, 2),
            'demand_cost_savings_percent': round(demand_cost_savings_percent, 2),
            'energy_cost_savings': round(energy_cost_savings, 2),
            'energy_cost_savings_percent': round(energy_cost_savings_percent, 2)
        },
        'visualization': {
            'results_plot': f'data:image/png;base64,{plot_data}'
        },
        'schedule': {
            'optimized_schedule': [
                {
                    'process_name': process_name,
                    'machines': details['machines'],
                    'start_time': details['machines'][0]['start_time'] if details['machines'] else f"{int(details['start_hour']):02d}:00",
                    'end_time': details['machines'][0]['end_time'] if details['machines'] else f"{int((details['start_hour'] + 1) % 24):02d}:00"
                }
                for process_name, details in sorted(
                    optimized_schedule.items(),
                    key=lambda x: (
                        x[0].split(' - ')[0],
                        process_order_map[x[0].split(' - ')[0]].index(x[0].split(' - ')[1])
                    )
                )
            ]
        }
    }

@app.route('/api/run_optimization', methods=['POST'])
def run_optimization():
    """Run optimization for a given simulation."""
    data = request.get_json() or {}
    simulation_id = data.get('simulation_id')
    try:
        tariff_type = TariffType(data.get('tariff_type'))
    except ValueError:
        logger.error(f"Invalid tariff_type: {data.get('tariff_type')}")
        return jsonify({'error': 'Invalid tariff_type'}), 400

    if not simulation_id:
        return jsonify({'error': 'Missing simulation_id'}), 400

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        simulation = fetch_simulation_data(cursor, simulation_id)
        if not simulation:
            return jsonify({'error': f'Simulation {simulation_id} not found'}), 404
        
        db_tariff_type = TariffType(simulation['tariff_type'])
        if db_tariff_type != tariff_type:
            logger.warning(f"Tariff mismatch: using {db_tariff_type.value}")
            tariff_type = db_tariff_type

        ga_params = fetch_ga_parameters(cursor, simulation_id)
        production_lines = initialize_system_from_db(simulation_id)
        original_load, machines_data = calculate_original_load(cursor, simulation_id)
        original_schedule = get_original_schedule(simulation_id)
        
        tariff = Tariff()
        original_cost_details = tariff.calculate_total_cost(original_load, tariff_type)
        
        best_solution, fitness_history = genetic_algorithm(
            production_lines, tariff, original_load, tariff_type,
            ga_params['population_size'], ga_params['crossover_rate'],
            ga_params['mutation_rate'], ga_params['tournament_size'],
            ga_params['elitism_rate']
        )
        
        optimized_load = best_solution.calculate_load_profile()
        optimized_cost_details = tariff.calculate_total_cost(optimized_load, tariff_type)
        
        start_times = best_solution.get_process_start_times()
        optimized_schedule = build_optimized_schedule(cursor, simulation_id, production_lines, start_times)
        plot_data = plot_results(original_load, best_solution, tariff, tariff_type.value, 
                               simulation_id, original_schedule, tariff_type, optimized_schedule)
        
        save_optimization_results(cursor, conn, simulation_id, optimized_load, 
                                optimized_cost_details, production_lines, start_times)
        
        response = build_response(original_load, optimized_load, original_cost_details, 
                               optimized_cost_details, optimized_schedule, plot_data)
        
        return jsonify(response), 200
    
    except Exception as e:
        conn.rollback()
        logger.error(f"Optimization failed: {e}")
        return jsonify({'error': f'Optimization error: {str(e)}'}), 500
    
    finally:
        cursor.close()
        conn.close()

@app.route('/api/delete_data', methods=['DELETE'])
def delete_data():
    data = request.get_json()
    if not data or 'simulation_id' not in data:
        return jsonify({'error': 'simulation_id is required'}), 400

    simulation_id = data['simulation_id']
    if not isinstance(simulation_id, int):
        return jsonify({'error': 'simulation_id must be a number'}), 400

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute('SELECT simulation_id FROM simulations WHERE simulation_id = %s', (simulation_id,))
        if not cursor.fetchone():
            return jsonify({'error': f'Simulation with ID {simulation_id} not found'}), 404
        
        cursor.execute('DELETE FROM simulations WHERE simulation_id = %s', (simulation_id,))
        conn.commit()
        return jsonify({'message': f'Simulation {simulation_id} and all related data successfully deleted'}), 200

    except mysql.connector.Error as err:
        conn.rollback()
        logger.error(f"Failed to delete simulation: {err}")
        return jsonify({'error': f'Failed to delete simulation: {str(err)}'}), 500
    finally:
        cursor.close()
        conn.close()

@app.route('/api/save_ga_parameters', methods=['POST'])
def save_ga_parameters():
    data = request.get_json()
    if not data or 'simulation_id' not in data:
        return jsonify({'error': 'simulation_id is required'}), 400

    simulation_id = data['simulation_id']
    population_size = data.get('population_size', POPULATION_SIZE)
    crossover_rate = data.get('crossover_rate', CROSSOVER_RATE)
    mutation_rate = data.get('mutation_rate', MUTATION_RATE)
    tournament_size = data.get('tournament_size', TOURNAMENT_SIZE)
    elitism_rate = data.get('elitism_rate', ELITISM_RATE)

    try:
        population_size = int(population_size)
        crossover_rate = float(crossover_rate)
        mutation_rate = float(mutation_rate)
        tournament_size = int(tournament_size)
        elitism_rate = float(elitism_rate)

        if population_size < 10:
            return jsonify({'error': 'population_size must be at least 10'}), 400
        if not (0 <= crossover_rate <= 1):
            return jsonify({'error': 'crossover_rate must be between 0 and 1'}), 400
        if not (0 <= mutation_rate <= 1):
            return jsonify({'error': 'mutation_rate must be between 0 and 1'}), 400
        if tournament_size < 2:
            return jsonify({'error': 'tournament_size must be at least 2'}), 400
        if not (0 <= elitism_rate <= 0.5):
            return jsonify({'error': 'elitism_rate must be between 0 and 0.5'}), 400

    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid parameter types'}), 400

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute('SELECT simulation_id FROM simulations WHERE simulation_id = %s', (simulation_id,))
        if not cursor.fetchone():
            return jsonify({'error': f'Simulation with ID {simulation_id} not found'}), 404

        cursor.execute('DELETE FROM ga_parameters WHERE simulation_id = %s', (simulation_id,))

        cursor.execute('''
            INSERT INTO ga_parameters (
                simulation_id, population_size, crossover_rate, mutation_rate, 
                tournament_size, elitism_rate
            ) VALUES (%s, %s, %s, %s, %s, %s)
        ''', (simulation_id, population_size, crossover_rate, mutation_rate, tournament_size, elitism_rate))
        conn.commit()

        return jsonify({'message': 'GA parameters saved successfully'}), 200

    except mysql.connector.Error as err:
        conn.rollback()
        logger.error(f"Failed to save GA parameters: {err}")
        return jsonify({'error': f'Failed to save GA parameters: {str(err)}'}), 500
    finally:
        cursor.close()
        conn.close()

@app.route('/api/ga_parameters/<int:simulation_id>', methods=['GET'])
def get_ga_parameters(simulation_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        cursor.execute('''
            SELECT population_size, crossover_rate, mutation_rate, tournament_size, elitism_rate
            FROM ga_parameters 
            WHERE simulation_id = %s
        ''', (simulation_id,))
        
        params = cursor.fetchone()
        if params:
            return jsonify(params), 200
        else:
            return jsonify({
                'population_size': POPULATION_SIZE,
                'crossover_rate': CROSSOVER_RATE,
                'mutation_rate': MUTATION_RATE,
                'tournament_size': TOURNAMENT_SIZE,
                'elitism_rate': ELITISM_RATE
            }), 200

    except mysql.connector.Error as err:
        logger.error(f"Failed to get GA parameters: {err}")
        return jsonify({'error': f'Failed to get GA parameters: {str(err)}'}), 500
    finally:
        cursor.close()
        conn.close()

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)
