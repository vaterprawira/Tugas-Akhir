import matplotlib
matplotlib.use('Agg')
from flask import Flask, request, jsonify
import mysql.connector
from datetime import datetime, timedelta, time
import numpy as np
import random
import copy
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import io
import base64
import logging
from enum import Enum
from flask_cors import CORS
import re

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

def get_tariff_details_from_db(cursor, simulation_id):
    cursor.execute('''
        SELECT s.tariff_type, ct.lwbp_rate, ct.wbp_rate, 
               ct.peak_start_hour, ct.peak_end_hour, ct.demand_charge
        FROM simulations s
        JOIN custom_tariffs ct ON s.simulation_id = ct.simulation_id
        WHERE s.simulation_id = %s
    ''', (simulation_id,))
    tariff_details = cursor.fetchone()
    return tariff_details

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor(buffered=True)

    try:
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
            CREATE TABLE IF NOT EXISTS custom_tariffs (
                tariff_id INT PRIMARY KEY AUTO_INCREMENT,
                simulation_id INT NOT NULL,
                lwbp_rate DECIMAL(10, 4) NOT NULL,
                wbp_rate DECIMAL(10, 4) NOT NULL,
                peak_start_hour TIME NOT NULL,
                peak_end_hour TIME NOT NULL,
                demand_charge DECIMAL(10, 2) NOT NULL,
                FOREIGN KEY (simulation_id) REFERENCES simulations(simulation_id) ON DELETE CASCADE
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
    except mysql.connector.Error as err:
        logger.error(f"Database connection failed: {err}")
        conn.rollback()
    finally:
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
ENERGY_TOLERANCE = 0.05

def _parse_time_to_hour(time_input):
    if time_input is None:
        logger.warning("Received None for time input. Defaulting to 0.")
        return 0
    
    if isinstance(time_input, time):
        return time_input.hour
    elif isinstance(time_input, timedelta):
        return int(time_input.total_seconds() // 3600)
    elif isinstance(time_input, str):
        try:
            return int(time_input.split(':')[0])
        except (ValueError, IndexError):
            logger.warning(f"Invalid time string format for hour parsing: {time_input}. Defaulting to 0.")
            return 0
    else:
        logger.warning(f"Unknown time input format: {type(time_input)}. Defaulting to 0.")
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
                profile[hour_index] += adjusted_load
        return profile

class ProductionProcess:
    def __init__(self, name, machines, process_id=None):
        self.name = name
        self.machines = machines if isinstance(machines, list) else [machines]
        self.process_id = process_id

class Tariff:
    def __init__(self, lwbp_rate, wbp_rate, peak_start_hour, peak_end_hour, demand_charge):
        self.lwbp_rate = lwbp_rate
        self.wbp_rate = wbp_rate
        self.demand_charge = demand_charge
        
        self.peak_hours = []
        if peak_start_hour <= peak_end_hour:
            self.peak_hours = list(range(peak_start_hour, peak_end_hour))
        else:
            self.peak_hours = list(range(peak_start_hour, 24)) + list(range(0, peak_end_hour))
        
        self.off_peak_hours = [h for h in range(24) if h not in self.peak_hours]
    
    def get_rate(self):
        return self.lwbp_rate
    
    def calculate_peak_rate(self, hour):
        return self.wbp_rate if hour in self.peak_hours else self.lwbp_rate
    
    def calculate_demand_cost(self, peak_load, tariff_type: TariffType):
        return peak_load * self.demand_charge if tariff_type == TariffType.WITH_DEMAND else 0.0
    
    def calculate_energy_cost(self, load_profile):
        wbp_energy_cost_daily = sum(load_profile[h] * self.calculate_peak_rate(h) for h in self.peak_hours)
        lwbp_energy_cost_daily = sum(load_profile[h] * self.get_rate() for h in self.off_peak_hours)
        return wbp_energy_cost_daily, lwbp_energy_cost_daily
    
    def calculate_total_cost(self, load_profile, tariff_type: TariffType):
        peak_load = max(load_profile)
        
        demand_cost_monthly = self.calculate_demand_cost(peak_load, tariff_type)
        demand_cost_daily = demand_cost_monthly / 30 if demand_cost_monthly > 0 else 0.0

        wbp_energy_cost_daily, lwbp_energy_cost_daily = self.calculate_energy_cost(load_profile)
        total_energy_cost_daily = wbp_energy_cost_daily + lwbp_energy_cost_daily
        total_energy_cost_monthly = total_energy_cost_daily * 30

        total_daily_energy_kwh = np.sum(load_profile)
        total_monthly_energy_kwh = total_daily_energy_kwh * 30
        
        total_cost_daily = demand_cost_daily + total_energy_cost_daily
        total_cost_monthly = demand_cost_monthly + total_energy_cost_monthly
        
        return {
            'peak_load': peak_load,
            
            'total_daily_energy': total_daily_energy_kwh,
            'total_monthly_energy': total_monthly_energy_kwh,

            'demand_cost_daily': demand_cost_daily,
            'demand_cost_monthly': demand_cost_monthly,

            'energy_cost_daily': total_energy_cost_daily,
            'energy_cost_monthly': total_energy_cost_monthly,

            'total_cost_daily': total_cost_daily,
            'total_cost_monthly': total_cost_monthly,
        }

def initialize_system_from_db(simulation_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        cursor.execute('''
            SELECT p.process_id, p.process_name, p.process_order, 
                   m.machine_id, m.machine_name, m.quantity, m.power, 
                   (TIME_TO_SEC(TIMEDIFF(
                       CASE WHEN m.stop_time < m.start_time THEN ADDTIME(m.stop_time, '24:00:00') ELSE m.stop_time END,
                       m.start_time
                   )) / 3600) AS operation_hours
            FROM processes p
            LEFT JOIN machines m ON p.process_id = m.process_id
            WHERE p.simulation_id = %s
            ORDER BY p.process_order
        ''', (simulation_id,))
        processes_data = cursor.fetchall()
        
        processes_dict = {}
        for row in processes_data:
            full_process_name = row['process_name']
            if full_process_name not in processes_dict:
                processes_dict[full_process_name] = {
                    'process_id': row['process_id'],
                    'process_name': full_process_name,
                    'process_order': row['process_order'],
                    'machines': []
                }
            if row['machine_id']:
                operation_hours = int(row['operation_hours']) if row['operation_hours'] is not None else 1
                machine = Machine(
                    row['machine_name'],
                    float(row['power']),
                    row['quantity'],
                    operation_hours,
                    machine_id=row['machine_id']
                )
                processes_dict[full_process_name]['machines'].append(machine)
        
        production_line = []
        sorted_process_names_all = []
        for product_key, process_steps in process_order_map.items():
            for step in process_steps:
                sorted_process_names_all.append(f"{product_key} - {step}")

        for p_name in sorted_process_names_all:
            if p_name in processes_dict:
                process_info = processes_dict[p_name]
                production_line.append(ProductionProcess(
                    process_info['process_name'], 
                    process_info['machines'], 
                    process_id=process_info['process_id']
                ))
        
        return {"selected_product": production_line}
    
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
    parts = process_name.split(' - ', 1)
    if len(parts) == 2:
        product, step = parts
        order_list = process_order_map.get(product, [])
        return (product, order_list.index(step) if step in order_list else 999)
    return (process_name, 999)

class Chromosome:
    def __init__(self, production_lines, original_peak_load, original_load, shift_hour=0):
        self.production_lines = production_lines
        self.original_peak_load = original_peak_load
        self.original_load = original_load
        self.fitness = 0
        self.load_profile = np.zeros(HOURS_IN_DAY)
        self.genes = None
        self.shift_hour = shift_hour
    
    def get_process_start_times(self):
        calculated_start_times = {}
        
        processes_for_sorting = []
        for p_obj in self.production_lines["selected_product"]:
            processes_for_sorting.append((get_process_sort_key(p_obj.name), p_obj))
        
        sorted_process_objects = [p_obj for _, p_obj in sorted(processes_for_sorting)]

        process_name_to_gene_idx = {p.name: i for i, p in enumerate(self.production_lines["selected_product"])}

        current_product_line_end_time = {product: 0 for product in process_order_map.keys()}
        
        synced_processes_names = [
            "Wafer Cream - Baking",
            "Wafer Cream - Cooling Sheet and Conditioning",
            "Wafer Cream - Spreading and Stacking",
            "Wafer Cream - Sandwich Cooling",
            "Wafer Cream - Cutting"
        ]
        
        sync_group_actual_start_time = None
        sync_group_latest_end_time = None
        
        for i, curr_process in enumerate(sorted_process_objects):
            current_process_max_duration = 0
            if curr_process.machines:
                current_process_max_duration = max(m.operation_hours for m in curr_process.machines)

            current_process_start_hour = 0
            
            product_name = curr_process.name.split(' - ')[0]

            if curr_process.name == "Wafer Cream - Batter Preparation" or \
               curr_process.name == "Wafer Stick - Batter Preparation":
                
                gene_idx = process_name_to_gene_idx.get(curr_process.name, 0)
                current_process_start_hour = (self.genes["start_times"][gene_idx] + self.shift_hour)
                
                current_product_line_end_time[product_name] = (current_process_start_hour + current_process_max_duration)
                
                sync_group_actual_start_time = None
                sync_group_latest_end_time = None 
                
            elif curr_process.name in synced_processes_names:
                if sync_group_actual_start_time is None:
                    predecessor_product_name = "Wafer Cream"
                    
                    if predecessor_product_name in current_product_line_end_time:
                        sync_group_actual_start_time = current_product_line_end_time[predecessor_product_name]
                    else:
                        gene_idx = process_name_to_gene_idx.get(curr_process.name, 0)
                        sync_group_actual_start_time = (self.genes["start_times"][gene_idx] + self.shift_hour)
                    
                    current_process_start_hour = sync_group_actual_start_time
                else:
                    current_process_start_hour = sync_group_actual_start_time
                
                current_process_end_hour_linear = current_process_start_hour + current_process_max_duration
                
                if sync_group_latest_end_time is None:
                    sync_group_latest_end_time = current_process_end_hour_linear
                else:
                    sync_group_latest_end_time = max(sync_group_latest_end_time, current_process_end_hour_linear)
                
            else:
                if sync_group_actual_start_time is not None and sync_group_latest_end_time is not None:
                    current_process_start_hour = sync_group_latest_end_time
                    
                    sync_group_actual_start_time = None 
                    sync_group_latest_end_time = None
                else:
                    current_process_start_hour = current_product_line_end_time.get(product_name, 0)

                current_product_line_end_time[product_name] = (current_process_start_hour + current_process_max_duration)

            calculated_start_times[curr_process.name] = current_process_start_hour % HOURS_IN_DAY
            
        for p_name in calculated_start_times:
            if p_name not in [p.name for p in sorted_process_objects if p.machines]:
                pass


        result_start_times_formatted = {"selected_product": {}}
        for p in self.production_lines["selected_product"]:
            result_start_times_formatted["selected_product"][p.name] = calculated_start_times.get(p.name, 0)

        return result_start_times_formatted
    
    def calculate_load_profile(self):
        self.load_profile = np.zeros(HOURS_IN_DAY)
        start_times = self.get_process_start_times()
        for process in self.production_lines["selected_product"]:
            start_hour = start_times["selected_product"][process.name]
            for machine in process.machines:
                machine_load = machine.get_load_profile(start_hour)
                self.load_profile += machine_load
        return self.load_profile
    
    def calculate_fitness(self, tariff, tariff_type: TariffType):
        load_profile = self.calculate_load_profile()
        cost_details_electricity_only = tariff.calculate_total_cost(load_profile, tariff_type)
        
        optimized_peak = cost_details_electricity_only['peak_load']
        total_optimized_monthly_electricity_cost = cost_details_electricity_only['total_cost_monthly']
        total_optimized_monthly_energy = cost_details_electricity_only['total_monthly_energy']

        energy_deviation_penalty = 0.0
        if self.original_load.sum() > 0: 
            original_total_daily_energy_kwh = np.sum(self.original_load) 
            original_total_monthly_energy_kwh = original_total_daily_energy_kwh * 30 
            
            if original_total_monthly_energy_kwh > 0:
                energy_diff_ratio = abs(total_optimized_monthly_energy - original_total_monthly_energy_kwh) / original_total_monthly_energy_kwh
                if energy_diff_ratio > ENERGY_TOLERANCE:
                    energy_deviation_penalty = energy_diff_ratio * 100 

        NIGHT_START_HOURS = set(range(22, 24)) | set(range(0, 6))
        
        night_start_violations = 0
        start_times_dict = self.get_process_start_times()

        for process in self.production_lines["selected_product"]:
            start_hour = start_times_dict["selected_product"].get(process.name)
            if start_hour is not None and start_hour in NIGHT_START_HOURS:
                night_start_violations += 1

        num_total_processes = len(self.production_lines["selected_product"])
        normalized_night_start_penalty = night_start_violations / (num_total_processes if num_total_processes > 0 else 1)

        total_optimized_on_peak_energy = sum(load_profile[h] for h in tariff.peak_hours)
        original_total_on_peak_energy = sum(self.original_load[h] for h in tariff.peak_hours)

        normalized_on_peak_energy = 0.0
        if original_total_on_peak_energy < 1e-6:
            if total_optimized_on_peak_energy > 0:
                normalized_on_peak_energy = total_optimized_on_peak_energy * 1000
            else:
                normalized_on_peak_energy = 0.0
        elif original_total_on_peak_energy > 1e-6:
            normalized_on_peak_energy = total_optimized_on_peak_energy / original_total_on_peak_energy

        if energy_deviation_penalty > 0:
             self.fitness = 1e-9
        else:
            original_cost_details_electricity_only = tariff.calculate_total_cost(self.original_load, tariff_type)
            original_total_monthly_electricity_cost = original_cost_details_electricity_only['total_cost_monthly']
            original_peak_load_for_norm = original_cost_details_electricity_only['peak_load']
            
            norm_factor_electricity_cost = original_total_monthly_electricity_cost if original_total_monthly_electricity_cost > 1e-6 else 1.0
            
            norm_factor_peak = original_peak_load_for_norm if original_peak_load_for_norm > 1e-6 else 1.0
            
            normalized_electricity_cost = total_optimized_monthly_electricity_cost / norm_factor_electricity_cost
            normalized_peak = optimized_peak / norm_factor_peak
            
            FACTOR_PEAK_LOAD = 5.0
            FACTOR_ON_PEAK_ENERGY = 4.0
            FACTOR_NIGHT_START = 2.0

            objective_value = normalized_electricity_cost + \
                              (normalized_peak * FACTOR_PEAK_LOAD) + \
                              (normalized_night_start_penalty * FACTOR_NIGHT_START) + \
                              (normalized_on_peak_energy * FACTOR_ON_PEAK_ENERGY)
            
            self.fitness = 1 / (objective_value + 1e-6)
            
        return self.fitness
    
    def debug_violations(self):
        debug_info = []
        start_times = self.get_process_start_times()
        processes = self.production_lines["selected_product"]
        
        debug_info.append("=== VIOLATION DEBUG (Post-Calculation) ===")
        debug_info.append("Process Start Times (Calculated):")
        
        processes_for_sorting = []
        for p_obj in processes:
            processes_for_sorting.append((get_process_sort_key(p_obj.name), p_obj))
        sorted_processes_debug = [p_obj for _, p_obj in sorted(processes_for_sorting)]

        for process in sorted_processes_debug:
            start_time = start_times["selected_product"].get(process.name, "N/A")
            process_duration = max(m.operation_hours for m in process.machines) if process.machines else 0
            end_time = (start_time + process_duration) % HOURS_IN_DAY if isinstance(start_time, int) else "N/A"
            debug_info.append(f"  {process.name}: Start={start_time:02d}, Duration={process_duration}, End={end_time:02d}")
        
        debug_info.append("\nSequential Process Analysis: (Constraints are enforced upstream)")
        
        debug_info.append("=== END DEBUG ===")
        logger.debug("\n".join(debug_info))
        return debug_info


def setup_deap(production_lines, original_peak_load, original_load, tariff, tariff_type: TariffType, shift_hour=0):
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", dict, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    def init_individual():
        num_processes = len(production_lines["selected_product"])
        return creator.Individual({
            "start_times": [random.randint(0, HOURS_IN_DAY - 1) for _ in range(num_processes)],
        })
    
    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    def evaluate(individual):
        chromosome = Chromosome(production_lines, original_peak_load, original_load, shift_hour)
        chromosome.genes = individual
        return (chromosome.calculate_fitness(tariff, tariff_type),)
    
    def cx_custom(ind1, ind2):
        child1, child2 = copy.deepcopy(ind1), copy.deepcopy(ind2)

        for i in range(len(child1["start_times"])):
            if random.random() < CROSSOVER_RATE: 
                child1["start_times"][i], child2["start_times"][i] = \
                    child2["start_times"][i], child1["start_times"][i]

        return child1, child2
    
    def mut_custom(individual):
        mutated = copy.deepcopy(individual)
        
        if random.random() < MUTATION_RATE:
            gene_index = random.randint(0, len(mutated["start_times"]) - 1)
            mutated["start_times"][gene_index] = random.randint(0, HOURS_IN_DAY - 1)
            
        return (mutated,)
    
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", cx_custom)
    toolbox.register("mutate", mut_custom)
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)
    
    return toolbox

def genetic_algorithm(production_lines, tariff, original_load, tariff_type: TariffType, 
                      population_size=POPULATION_SIZE, crossover_rate=CROSSOVER_RATE, 
                      mutation_rate=MUTATION_RATE, tournament_size=TOURNAMENT_SIZE, 
                      elitism_rate=ELITISM_RATE, shift_hour=0):
    import time
    random.seed(int(time.time() * 1000) % 2**32)
    np.random.seed(int(time.time() * 1000) % 2**32)
    
    original_peak_load = max(original_load)
    toolbox = setup_deap(production_lines, original_peak_load, original_load, tariff, tariff_type, shift_hour)
    
    population = toolbox.population(n=population_size)
    
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

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
                    logger.info(f"GA terminated due to stagnation at generation {gen}.")
                    break
            previous_fitness = current_fitness
            logger.info(f"Generation {gen}: Best Fitness = {current_fitness:.6f}")
        
        best_individual = hof.items[0]
        best_chromosome = Chromosome(production_lines, original_peak_load, original_load, shift_hour)
        best_chromosome.genes = best_individual
        best_chromosome.fitness = best_individual.fitness.values[0]
        
        return best_chromosome, best_fitness_history
    
    except Exception as e:
        logger.error(f"Genetic algorithm failed: {e}")
        raise

def get_original_schedule(simulation_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        cursor.execute('''
            SELECT p.process_name, m.machine_id, m.machine_name, m.quantity, m.power, m.start_time, m.stop_time,
                   (TIME_TO_SEC(TIMEDIFF(
                       CASE WHEN m.stop_time < m.start_time THEN ADDTIME(m.stop_time, '24:00:00') ELSE m.stop_time END,
                       m.start_time
                   )) / 3600) AS operation_hours
            FROM processes p
            JOIN machines m ON p.process_id = m.process_id
            WHERE p.simulation_id = %s
            ORDER BY p.process_order, m.machine_name
        ''', (simulation_id,))
        machines_data = cursor.fetchall()
        
        original_schedule = {}
        sorted_process_names_all = []
        for product_key, process_steps in process_order_map.items():
            for step in process_steps:
                sorted_process_names_all.append(f"{product_key} - {step}")

        for p_name in sorted_process_names_all:
            original_schedule[p_name] = {'machines': []}

        for data in machines_data:
            process_name = data['process_name']
            
            start_hour_parsed = _parse_time_to_hour(data['start_time'])
            duration = int(data['operation_hours']) if data['operation_hours'] is not None else 1
            
            if process_name in original_schedule:
                original_schedule[process_name]['machines'].append({
                    'machine_id': data['machine_id'],
                    'machine_name': data['machine_name'],
                    'quantity': data['quantity'],
                    'power': float(data['power']),
                    'start_hour': start_hour_parsed,
                    'duration': duration,
                    'start_time': str(data['start_time'])[:5],
                    'end_time': f"{(start_hour_parsed + duration) % HOURS_IN_DAY:02d}:00"
                })
        
        original_schedule = {k: v for k, v in original_schedule.items() if v['machines']}
        
        return original_schedule
    
    finally:
        cursor.close()
        conn.close()

def plot_results(original_load, best_solution, tariff, scenario_name, simulation_id, original_schedule, tariff_type, optimized_schedule):
    HOURS_IN_DAY = 24

    fig, ax1 = plt.subplots(figsize=(14, 8))

    for h in tariff.peak_hours:
        ax1.axvspan(h, h + 1, color='red', alpha=0.1, label='Peak Hours' if h == tariff.peak_hours[0] else "")

    hours = list(range(HOURS_IN_DAY))
    ax1.plot(hours, original_load, 'r-', label='Original Load', linewidth=2)
    ax1.plot(hours, best_solution.calculate_load_profile(), 'b--', label=f'Optimised Load\n({scenario_name})', linewidth=2)

    ax1.set_xlabel('Hour of Day', fontsize=16)
    ax1.set_ylabel('Load (kW)', fontsize=16)
    ax1.set_title(f'Load Profile Comparison - {scenario_name}', fontsize=18)

    ax1.legend(loc='upper left', fontsize=14, frameon=True, borderpad=1)

    ax1.grid(True)
    ax1.set_xticks(hours)
    ax1.tick_params(axis='both', which='major', labelsize=14)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)

    return plot_data


@app.route('/api/insert_data', methods=['POST'])
def insert_data():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    products_data = data.get('products')
    tariff_type_str = data.get('tariff')
    tariff_params = data.get('tariff_params')

    try:
        tariff_type = TariffType(tariff_type_str)
    except ValueError:
        logger.error(f"Invalid tariff_type received: {tariff_type_str}")
        return jsonify({'error': f'Invalid tariff_type: {tariff_type_str}'}), 400

    if not products_data or not isinstance(products_data, list):
        return jsonify({'error': 'Missing products data'}), 400
    
    if not tariff_params:
        return jsonify({'error': 'Missing custom tariff parameters.'}), 400
    
    try:
        lwbp_rate = float(tariff_params.get('offPeakRate'))
        wbp_rate = float(tariff_params.get('onPeakRate'))
        peak_start_time_str = tariff_params.get('peakPeriodStartHour')
        peak_end_time_str = tariff_params.get('peakPeriodEndHour')
        demand_charge = float(tariff_params.get('demandChargeInput'))

        time_pattern = r"^(?:2[0-3]|[01]?[0-9]):[0-5][0-9]$"
        if not (isinstance(peak_start_time_str, str) and re.match(time_pattern, peak_start_time_str)):
            raise ValueError("peakPeriodStartHour must be in HH:MM format.")
        if not (isinstance(peak_end_time_str, str) and re.match(time_pattern, peak_end_time_str)):
            raise ValueError("peakPeriodEndHour must be in HH:MM format.")
        
        peak_start_time_str_db = peak_start_time_str + ":00"
        peak_end_time_str_db = peak_end_time_str + ":00"

    except (ValueError, TypeError) as e:
        return jsonify({'error': f'Invalid format for custom tariff parameters: {e}'}), 400


    conn = get_db_connection()
    cursor = conn.cursor(buffered=True)

    try:
        cursor.execute('''
            INSERT INTO simulations (tariff_type)
            VALUES (%s)
        ''', (tariff_type.value,))
        conn.commit()
        simulation_id = cursor.lastrowid
        
        cursor.execute('''
            INSERT INTO custom_tariffs (simulation_id, lwbp_rate, wbp_rate, peak_start_hour, peak_end_hour, demand_charge)
            VALUES (%s, %s, %s, %s, %s, %s)
        ''', (simulation_id, lwbp_rate, wbp_rate, peak_start_time_str_db, peak_end_time_str_db, demand_charge))
        conn.commit()

        for product_entry in products_data:
            product_name = product_entry.get('product')
            processes_data = product_entry.get('processes')

            if not product_name or not processes_data:
                return jsonify({'error': 'Missing product or processes data'}), 400

            cursor.execute('SELECT product_id FROM products WHERE product_name = %s', (product_name,))
            product = cursor.fetchone()
            if not product:
                return jsonify({'error': f'Product {product_name} not found'}), 404

            for process_data in processes_data:
                process_step_name = process_data.get('process_name')
                machines = process_data.get('machines', [])

                full_process_name = f"{product_name} - {process_step_name}"
                
                order_list = process_order_map.get(product_name, [])
                process_order_value = order_list.index(process_step_name) if process_step_name in order_list else 999
                
                if process_order_value == 999:
                    return jsonify({'error': f'Invalid process name {full_process_name} for product {product_name}'}), 400

                cursor.execute('''
                    INSERT INTO processes (simulation_id, process_name, process_order)
                    VALUES (%s, %s, %s)
                ''', (simulation_id, full_process_name, process_order_value))
                conn.commit()
                process_id = cursor.lastrowid

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
            SELECT s.simulation_id, s.tariff_type, 
                   GROUP_CONCAT(DISTINCT SUBSTRING_INDEX(p.process_name, ' - ', 1)) as products,
                   s.created_at
            FROM simulations s
            JOIN processes p ON s.simulation_id = p.simulation_id
            GROUP BY s.simulation_id, s.tariff_type, s.created_at
            ORDER BY s.created_at DESC
        ''')
        simulations = cursor.fetchall()
        return jsonify(simulations), 200
    
    except Exception as e:
        logger.error(f"Get simulations failed: {e}")
        return jsonify({'error': str(e)}), 500
    
    finally:
        cursor.close()
        conn.close()

@app.route('/api/simulations/<int:simulation_id>/tariff_details', methods=['GET'])
def get_tariff_details(simulation_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        tariff_details = get_tariff_details_from_db(cursor, simulation_id)
        if tariff_details:
            tariff_details['peak_start_hour'] = str(tariff_details['peak_start_hour'])[:5]
            tariff_details['peak_end_hour'] = str(tariff_details['peak_end_hour'])[:5]
            return jsonify(tariff_details), 200
        else:
            return jsonify({'error': 'Tariff details not found for this simulation.'}), 404
    except Exception as e:
        logger.error(f"Error fetching tariff details: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        conn.close()


def fetch_simulation_data(cursor, simulation_id):
    cursor.execute('SELECT tariff_type FROM simulations WHERE simulation_id = %s', (simulation_id,))
    return cursor.fetchone()

def fetch_ga_parameters(cursor, simulation_id):
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
    cursor.execute('''
        SELECT m.machine_name, m.quantity, m.power, m.start_time, m.stop_time,
               (TIME_TO_SEC(TIMEDIFF(
                   CASE WHEN m.stop_time < m.start_time THEN ADDTIME(m.stop_time, '24:00:00') ELSE m.stop_time END,
                   m.start_time
               )) / 3600) AS operation_hours
            FROM processes p
            JOIN machines m ON p.process_id = m.process_id
            WHERE p.simulation_id = %s
            ORDER BY p.process_order
        ''', (simulation_id,))
    machines_data = cursor.fetchall()
    
    original_load_profile = np.zeros(HOURS_IN_DAY)
    for machine_data in machines_data:
        machine = Machine(
            machine_data['machine_name'],
            float(machine_data['power']),
            machine_data['quantity'],
            int(machine_data['operation_hours']) if machine_data['operation_hours'] is not None else 1,
        )
        start_hour = _parse_time_to_hour(machine_data['start_time']) 
        original_load_profile += machine.get_load_profile(start_hour)
    
    return original_load_profile, machines_data

def save_optimization_results(cursor, conn, simulation_id, optimized_load, optimized_electricity_cost_details, 
                              production_lines, start_times):
    try:
        cursor.execute('''
            INSERT INTO optimization_results (
                simulation_id, total_cost, peak_load, total_energy, 
                demand_cost, energy_cost, load_profile
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        ''', (
            simulation_id,
            float(optimized_electricity_cost_details['total_cost_monthly']),
            float(optimized_electricity_cost_details['peak_load']),
            float(optimized_electricity_cost_details['total_monthly_energy']),
            float(optimized_electricity_cost_details['demand_cost_monthly']),
            float(optimized_electricity_cost_details['energy_cost_monthly']),
            str([float(x) for x in optimized_load.tolist()])
        ))
        result_id = cursor.lastrowid
        
        for process in production_lines['selected_product']:
            cursor.execute('SELECT process_id FROM processes WHERE simulation_id = %s AND process_name = %s',
                            (simulation_id, process.name))
            result = cursor.fetchone()
            if not result:
                logger.warning(f"Process {process.name} not found for saving optimized schedule.")
                continue
            process_id = result['process_id']
            
            start_hour_optimized = start_times['selected_product'][process.name]
            
            cursor.execute('''
                SELECT m.machine_id, TIME_TO_SEC(TIMEDIFF(m.stop_time, m.start_time)) / 3600 AS operation_hours
                FROM machines m
                JOIN processes p ON m.process_id = p.process_id
                WHERE p.simulation_id = %s AND p.process_name = %s
            ''', (simulation_id, process.name))
            machines = cursor.fetchall()
            
            for machine_db_data in machines:
                machine_id = machine_db_data['machine_id']
                duration = int(machine_db_data['operation_hours']) if machine_db_data['operation_hours'] is not None else 1
                
                cursor.execute('''
                    INSERT INTO optimized_schedules (result_id, process_id, machine_id, start_time, duration)
                    VALUES (%s, %s, %s, %s, %s)
                ''', (result_id, process_id, machine_id, f"{int(start_hour_optimized):02d}:00:00", duration))
        
        conn.commit()
        return result_id
    except mysql.connector.Error as err:
        conn.rollback()
        logger.error(f"Failed to save optimization results: {err}")
        raise

def build_optimized_schedule(cursor, simulation_id, production_lines, start_times):
    optimized_schedule_dict = {}
    
    processes_for_sorting = []
    for p_obj in production_lines["selected_product"]:
        processes_for_sorting.append((get_process_sort_key(p_obj.name), p_obj))
    sorted_process_objects = [p_obj for _, p_obj in sorted(processes_for_sorting)]

    for process in sorted_process_objects:
        start_hour = start_times['selected_product'][process.name]
        
        cursor.execute('''
            SELECT m.machine_id, m.machine_name, m.quantity, m.power, 
                   (TIME_TO_SEC(TIMEDIFF(
                       CASE WHEN m.stop_time < m.start_time THEN ADDTIME(m.stop_time, '24:00:00') ELSE m.stop_time END,
                       m.start_time
                   )) / 3600) AS operation_hours
            FROM machines m
            JOIN processes p ON m.process_id = p.process_id
            WHERE p.simulation_id = %s AND p.process_name = %s
        ''', (simulation_id, process.name))
        machines_data = cursor.fetchall()
        
        formatted_machines = []
        for machine_db_data in machines_data:
            duration = int(machine_db_data['operation_hours']) if machine_db_data['operation_hours'] is not None else 1
            formatted_machines.append({
                'machine_id': machine_db_data['machine_id'],
                'machine_name': machine_db_data['machine_name'],
                'quantity': machine_db_data['quantity'],
                'power': float(machine_db_data['power']),
                'duration': duration,
                'start_time': f"{int(start_hour):02d}:00",
                'start_hour': start_hour,
                'end_time': f"{(start_hour + duration) % HOURS_IN_DAY:02d}:00"
            })
            
        optimized_schedule_dict[process.name] = {
            'start_hour': start_hour,
            'machines': formatted_machines,
            'process_id': process.process_id
        }
    return optimized_schedule_dict

def build_response(original_load, optimized_load, original_electricity_cost_details, optimized_electricity_cost_details, 
                   original_schedule, optimized_schedule, plot_data): 
    original_peak_load = original_electricity_cost_details['peak_load']
    original_total_daily_energy = original_electricity_cost_details['total_daily_energy']
    original_total_monthly_energy = original_electricity_cost_details['total_monthly_energy']
    original_demand_cost_daily = original_electricity_cost_details['demand_cost_daily']
    original_demand_cost_monthly = original_electricity_cost_details['demand_cost_monthly']
    original_energy_cost_daily = original_electricity_cost_details['energy_cost_daily']
    original_energy_cost_monthly = original_electricity_cost_details['energy_cost_monthly'] 
    
    original_total_cost_daily = original_electricity_cost_details['total_cost_daily']
    original_total_cost_monthly = original_electricity_cost_details['total_cost_monthly']
    
    optimized_peak_load = optimized_electricity_cost_details['peak_load']
    optimized_total_daily_energy = optimized_electricity_cost_details['total_daily_energy']
    optimized_total_monthly_energy = optimized_electricity_cost_details['total_monthly_energy']
    optimized_demand_cost_daily = optimized_electricity_cost_details['demand_cost_daily']
    optimized_demand_cost_monthly = optimized_electricity_cost_details['demand_cost_monthly']
    optimized_energy_cost_daily = optimized_electricity_cost_details['energy_cost_daily']
    optimized_energy_cost_monthly = optimized_electricity_cost_details['energy_cost_monthly']

    optimized_total_cost_daily = optimized_electricity_cost_details['total_cost_daily']
    optimized_total_cost_monthly = optimized_electricity_cost_details['total_cost_monthly']
    
    peak_reduction = original_peak_load - optimized_peak_load
    peak_reduction_percent = (peak_reduction / original_peak_load * 100) if original_peak_load > 0 else 0

    demand_cost_daily_savings = original_demand_cost_daily - optimized_demand_cost_daily
    demand_cost_daily_savings_percent = (demand_cost_daily_savings / original_demand_cost_daily * 100) if original_demand_cost_daily > 0 else 0
    demand_cost_monthly_savings = original_demand_cost_monthly - optimized_demand_cost_monthly
    demand_cost_monthly_savings_percent = (demand_cost_monthly_savings / original_demand_cost_monthly * 100) if original_demand_cost_monthly > 0 else 0

    energy_cost_daily_savings = original_energy_cost_daily - optimized_energy_cost_daily
    energy_cost_daily_savings_percent = (energy_cost_daily_savings / original_energy_cost_daily * 100) if original_energy_cost_daily > 0 else 0
    energy_cost_monthly_savings = original_energy_cost_monthly - optimized_energy_cost_monthly
    energy_cost_monthly_savings_percent = (energy_cost_monthly_savings / original_energy_cost_monthly * 100) if original_energy_cost_monthly > 0 else 0

    total_cost_daily_savings = original_total_cost_daily - optimized_total_cost_daily
    total_cost_daily_savings_percent = (total_cost_daily_savings / original_total_cost_daily * 100) if original_total_cost_daily > 0 else 0
    total_cost_monthly_savings = original_total_cost_monthly - optimized_total_cost_monthly
    total_cost_monthly_savings_percent = (total_cost_monthly_savings / original_total_cost_monthly * 100) if original_total_cost_monthly > 0 else 0

    energy_diff_percent = (abs(optimized_total_monthly_energy - original_total_monthly_energy) / (original_total_monthly_energy or 1)) * 100

    sorted_optimized_schedule_output = []
    
    for p_name, p_details in optimized_schedule.items():
        parts = p_name.split(' - ', 1)
        if len(parts) == 2:
            product, step = parts
            order_list = process_order_map.get(product, [])
            sort_key = (product, order_list.index(step) if step in order_list else 999)
            sorted_optimized_schedule_output.append((sort_key, p_name, p_details))
        else:
            sorted_optimized_schedule_output.append(((p_name, 999), p_name, p_details))
    
    sorted_optimized_schedule_output.sort()

    final_optimized_schedule_list = []
    for _, process_name, details in sorted_optimized_schedule_output:
        final_optimized_schedule_list.append({
            'process_name': process_name,
            'machines': details['machines'],
            'start_time': details['machines'][0]['start_time'] if details['machines'] else "00:00",
            'end_time': details['machines'][0]['end_time'] if details['machines'] else "01:00",
            'process_id': details['process_id']
        })
    
    sorted_original_schedule_output = []
    for p_name, p_details in original_schedule.items():
        parts = p_name.split(' - ', 1)
        if len(parts) == 2:
            product, step = parts
            order_list = process_order_map.get(product, [])
            sort_key = (product, order_list.index(step) if step in order_list else 999)
            sorted_original_schedule_output.append((sort_key, p_name, p_details))
        else:
            sorted_original_schedule_output.append(((p_name, 999), p_name, p_details))
    
    sorted_original_schedule_output.sort()

    final_original_schedule_list = []
    for _, process_name, details in sorted_original_schedule_output:
        final_original_schedule_list.append({
            'process_name': process_name,
            'machines': details['machines'],
            'start_time': details['machines'][0]['start_time'] if details['machines'] else "00:00",
            'end_time': details['machines'][0]['end_time'] if details['machines'] else "01:00",
        })


    return {
        'summary': {
            'original_peak_load': round(original_peak_load, 2),
            'optimized_peak_load': round(optimized_peak_load, 2),
            'peak_reduction': round(peak_reduction, 2),
            'peak_reduction_percent': round(peak_reduction_percent, 2),

            'original_total_daily_energy': round(original_total_daily_energy, 2),
            'optimized_total_daily_energy': round(optimized_total_daily_energy, 2),
            'original_total_monthly_energy': round(original_total_monthly_energy, 2),
            'optimized_total_monthly_energy': round(optimized_total_monthly_energy, 2),
            'energy_deviation_percent': round(energy_diff_percent, 2),

            'original_demand_cost_daily': round(demand_cost_daily_savings, 2), 
            'optimized_demand_cost_daily': round(optimized_demand_cost_daily, 2),
            'original_demand_cost_monthly': round(original_demand_cost_monthly, 2),
            'optimized_demand_cost_monthly': round(optimized_demand_cost_monthly, 2),
            'demand_cost_daily_savings': round(demand_cost_daily_savings, 2),
            'demand_cost_daily_savings_percent': round(demand_cost_daily_savings_percent, 2),
            'demand_cost_monthly_savings': round(demand_cost_monthly_savings, 2),
            'demand_cost_monthly_savings_percent': round(demand_cost_monthly_savings_percent, 2),

            'original_energy_cost_daily': round(original_energy_cost_daily, 2),
            'optimized_energy_cost_daily': round(optimized_energy_cost_daily, 2),
            'original_energy_cost_monthly': round(original_energy_cost_monthly, 2),
            'optimized_energy_cost_monthly': round(optimized_energy_cost_monthly, 2),
            'energy_cost_daily_savings': round(energy_cost_daily_savings, 2),
            'energy_cost_daily_savings_percent': round(energy_cost_daily_savings_percent, 2),
            'energy_cost_monthly_savings': round(energy_cost_monthly_savings, 2),
            'energy_cost_monthly_savings_percent': round(energy_cost_monthly_savings_percent, 2),

            'original_total_cost_daily': round(original_total_cost_daily, 2),
            'optimized_total_cost_daily': round(optimized_total_cost_daily, 2),
            'original_total_cost_monthly': round(original_total_cost_monthly, 2),
            'optimized_total_cost_monthly': round(optimized_total_cost_monthly, 2),
            'total_cost_daily_savings': round(total_cost_daily_savings, 2),
            'total_cost_daily_savings_percent': round(total_cost_daily_savings_percent, 2),
            'total_cost_monthly_savings': round(total_cost_monthly_savings, 2),
            'total_cost_monthly_savings_percent': round(total_cost_monthly_savings_percent, 2),
        },
        'visualization': {
            'results_plot': f'data:image/png;base64,{plot_data}'
        },
        'schedule': {
            'original_schedule': final_original_schedule_list, 
            'optimized_schedule': final_optimized_schedule_list
        }
    }

@app.route('/api/run_optimization', methods=['POST'])
def run_optimization():
    data = request.get_json() or {}
    simulation_id = data.get('simulation_id')
    shift_hour = data.get('shift_hour', 0)
    
    tariff_type_str = data.get('tariff_type')
    if not tariff_type_str:
        return jsonify({'error': 'tariff_type is required in the request body'}), 400
    
    try:
        tariff_type_from_request = TariffType(tariff_type_str)
    except ValueError:
        logger.error(f"Invalid tariff_type from request: {tariff_type_str}")
        return jsonify({'error': 'Invalid tariff_type provided in request'}), 400

    if not simulation_id:
        return jsonify({'error': 'Missing simulation_id'}), 400

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        simulation_data_from_db = get_tariff_details_from_db(cursor, simulation_id)
        if not simulation_data_from_db:
            return jsonify({'error': f'Simulation {simulation_id} or its tariff details not found'}), 404
        
        db_tariff_type = TariffType(simulation_data_from_db['tariff_type'])
        if db_tariff_type != tariff_type_from_request:
            logger.warning(f"Tariff mismatch: Request specified {tariff_type_from_request.value}, but DB has {db_tariff_type.value}. Using DB's tariff type.")
            tariff_type_used = db_tariff_type
        else:
            tariff_type_used = tariff_type_from_request

        lwbp_rate_db = float(simulation_data_from_db['lwbp_rate'])
        wbp_rate_db = float(simulation_data_from_db['wbp_rate'])
        peak_start_hour_db = _parse_time_to_hour(simulation_data_from_db['peak_start_hour'])
        peak_end_hour_db = _parse_time_to_hour(simulation_data_from_db['peak_end_hour'])
        demand_charge_db = float(simulation_data_from_db['demand_charge'])

        tariff = Tariff(lwbp_rate_db, wbp_rate_db, peak_start_hour_db, peak_end_hour_db, demand_charge_db)

        ga_params = fetch_ga_parameters(cursor, simulation_id)
        
        production_lines = initialize_system_from_db(simulation_id)
        
        original_load, machines_data = calculate_original_load(cursor, simulation_id)
        
        original_load = np.roll(original_load, -shift_hour)

        original_schedule = get_original_schedule(simulation_id)
        
        original_electricity_cost_details = tariff.calculate_total_cost(original_load, tariff_type_used)
        
        best_chromosome, fitness_history = genetic_algorithm(
            production_lines, tariff, original_load, tariff_type_used,
            ga_params['population_size'], ga_params['crossover_rate'],
            ga_params['mutation_rate'], ga_params['tournament_size'],
            ga_params['elitism_rate'], shift_hour
        )
        
        optimized_load = best_chromosome.calculate_load_profile()
        optimized_electricity_cost_details = tariff.calculate_total_cost(optimized_load, tariff_type_used)
        
        start_times = best_chromosome.get_process_start_times()
        
        optimized_schedule = build_optimized_schedule(cursor, simulation_id, production_lines, start_times)
        
        plot_data = plot_results(original_load, best_chromosome, tariff, tariff_type_used.value, 
                                 simulation_id, original_schedule, tariff_type_used, optimized_schedule)
        
        save_optimization_results(cursor, conn, simulation_id, optimized_load, 
                                  optimized_electricity_cost_details, production_lines, start_times)
        
        response = build_response(original_load, optimized_load, 
                                  original_electricity_cost_details, optimized_electricity_cost_details,
                                  original_schedule, optimized_schedule, plot_data)
        
        logger.info(f"[DEBUG] Total original energy (daily): {np.sum(original_load):.2f} kWh")
        logger.info(f"[DEBUG] Total optimized energy (daily): {np.sum(optimized_load):.2f} kWh")

        return jsonify(response), 200
    
    except Exception as e:
        conn.rollback()
        logger.error(f"Optimization failed: {e}", exc_info=True)
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
        
        cursor.execute('DELETE FROM optimized_schedules WHERE result_id IN (SELECT result_id FROM optimization_results WHERE simulation_id = %s)', (simulation_id,))
        cursor.execute('DELETE FROM optimization_results WHERE simulation_id = %s', (simulation_id,))
        cursor.execute('DELETE FROM machines WHERE process_id IN (SELECT process_id FROM processes WHERE simulation_id = %s)', (simulation_id,))
        cursor.execute('DELETE FROM processes WHERE simulation_id = %s', (simulation_id,))
        cursor.execute('DELETE FROM custom_tariffs WHERE simulation_id = %s', (simulation_id,))
        cursor.execute('DELETE FROM ga_parameters WHERE simulation_id = %s', (simulation_id,))
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
        return jsonify({'error': 'Invalid parameter types. Please ensure all GA parameters are numbers.'}), 400

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
