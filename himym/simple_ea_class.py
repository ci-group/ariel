import numpy as np
import mujoco
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.ec.a001 import Individual
from ea_config import config


class SimpleEA:    
    def __init__(self, population_size=None, num_generations=None, num_joints=None):
        self.population_size = population_size or config.population_size
        self.num_generations = num_generations or config.num_generations
        self.num_joints = num_joints or 8
        
        self.population = []
        self.generation = 0
        self.fitness_history = []
        
        self.world = None
        self.model = None
        self.current_individual = None
        
    def create_individual(self):
        individual = Individual()
        
        genotype = []
        for _ in range(self.num_joints):
            amplitude = np.random.uniform(config.amplitude_init_min, config.amplitude_init_max)
            frequency = np.random.uniform(config.frequency_init_min, config.frequency_init_max)
            phase = np.random.uniform(config.phase_min, config.phase_max)
            genotype.extend([amplitude, frequency, phase])
        
        individual.genotype = genotype
        individual.fitness = 0.0
        return individual
    
    def initialize_population(self):
        print(f"Initializing population of {self.population_size} individuals")
        self.population = [self.create_individual() for _ in range(self.population_size)]
    
    def sinusoidal_controller(self, model, data, to_track):
        if self.current_individual is None:
            return
        
        num_joints = model.nu
        genotype = self.current_individual.genotype
        
        # Apply sinusoidal control based on genotype
        for i in range(num_joints):
            if i * 3 + 2 < len(genotype): 
                amplitude = genotype[i * 3]
                frequency = genotype[i * 3 + 1]
                phase = genotype[i * 3 + 2]
                
                # Generate sinusoidal control signal
                control_value = amplitude * np.sin(frequency * data.time + phase)
                data.ctrl[i] = np.clip(control_value, config.control_clip_min, config.control_clip_max)
    
    def evaluate_fitness(self, individual, start_pos, end_pos):
        """Calculate fitness based on distance traveled."""
        distance = np.linalg.norm(np.array(end_pos) - np.array(start_pos))
        individual.fitness = distance
        return distance
    
    def evaluate_individual(self, individual):
        data = mujoco.MjData(self.model)
        geoms = self.world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
        to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]
        
        # Set current individual
        self.current_individual = individual
        
        # Setup simulation with controller
        mujoco.set_mjcb_control(lambda m, d: self.sinusoidal_controller(m, d, to_track))
        
        # Get starting position
        start_pos = to_track[0].xpos.copy()
        
        # Run simulation
        sim_steps = int(config.simulation_time / self.model.opt.timestep)
        for _ in range(sim_steps):
            mujoco.mj_step(self.model, data)
        
        # Get ending position
        end_pos = to_track[0].xpos.copy()
        
        # Calculate and return fitness
        return self.evaluate_fitness(individual, start_pos, end_pos)
    
    def tournament_selection(self, tournament_size=None):
        if tournament_size is None:
            tournament_size = config.tournament_size
        
        selected = []
        for _ in range(self.population_size):
            tournament = np.random.choice(self.population, tournament_size, replace=False)
            winner = max(tournament, key=lambda ind: ind.fitness)
            selected.append(winner)
        return selected
    
    def crossover(self, parent1, parent2):
        child1 = Individual()
        child2 = Individual()
        
        crossover_point = np.random.randint(1, len(parent1.genotype))
        
        child1.genotype = (parent1.genotype[:crossover_point] + 
                          parent2.genotype[crossover_point:])
        child2.genotype = (parent2.genotype[:crossover_point] + 
                          parent1.genotype[crossover_point:])
        
        return child1, child2
    
    def mutate(self, individual, mutation_rate=None, mutation_strength=None):
        if mutation_rate is None:
            mutation_rate = config.mutation_rate
        if mutation_strength is None:
            mutation_strength = config.mutation_strength
        
        mutated = Individual()
        mutated.genotype = individual.genotype.copy()
        
        for i in range(len(mutated.genotype)):
            if np.random.random() < mutation_rate:
                # Add Gaussian noise
                mutated.genotype[i] += np.random.normal(0, mutation_strength)
                
                # Clamp values to reasonable ranges
                param_type = i % 3
                if param_type == 0:  # amplitude
                    mutated.genotype[i] = np.clip(mutated.genotype[i], 
                                                  config.amplitude_min, config.amplitude_max)
                elif param_type == 1:  # frequency
                    mutated.genotype[i] = np.clip(mutated.genotype[i], 
                                                  config.frequency_min, config.frequency_max)
                else:  # phase
                    mutated.genotype[i] = mutated.genotype[i] % config.phase_max
        
        return mutated
    
    def run_evolution(self):
        print("=" * 60)
        print("SIMPLE EVOLUTIONARY ALGORITHM STARTING")
        print("=" * 60)
        print(f"Population size: {self.population_size}")
        print(f"Generations: {self.num_generations}")
        print(f"Robot joints: {self.num_joints}")
        print("=" * 60)
        
        # Initialize world and robot
        mujoco.set_mjcb_control(None)
        self.world = SimpleFlatWorld(config.world_size)
        gecko_robot = gecko()
        self.world.spawn(gecko_robot.spec, spawn_position=[0, 0, 0])
        self.model = self.world.spec.compile()
        
        print(f"Robot has {self.model.nu} joints")
        
        self.initialize_population()
        
        best_fitness_history = []
        
        # Evolution loop
        for gen in range(self.num_generations):
            self.generation = gen
            print(f"\n{'='*60}")
            print(f"Generation {gen + 1}/{self.num_generations}")
            print(f"{'='*60}")
            
            # Evaluate each individual
            generation_fitness = []
            for i, individual in enumerate(self.population):
                fitness = self.evaluate_individual(individual)
                generation_fitness.append(fitness)
                
                if config.print_individual_fitness:
                    print(f"  Individual {i+1}: fitness = {fitness:.4f}")
            
            # Track best fitness
            best_fitness = max(generation_fitness)
            best_fitness_history.append(best_fitness)
            best_individual = max(self.population, key=lambda ind: ind.fitness)
            
            # Track statistics
            mean_fitness = np.mean(generation_fitness)
            self.fitness_history.append({
                'generation': gen + 1,
                'best': best_fitness,
                'mean': mean_fitness,
                'min': min(generation_fitness)
            })
            
            print(f"  Best fitness this generation: {best_fitness:.4f}")
            print(f"  Average fitness: {mean_fitness:.4f}")
            
            # Create next generation (except for last generation)
            if gen < self.num_generations - 1:
                # Selection
                parents = self.tournament_selection()
                
                # Crossover and mutation
                new_population = []
                for i in range(0, len(parents), 2):
                    if i + 1 < len(parents):
                        child1, child2 = self.crossover(parents[i], parents[i + 1])
                        new_population.extend([self.mutate(child1), self.mutate(child2)])
                    else:
                        new_population.append(self.mutate(parents[i]))
                
                # Elitism: Keep best individual
                if config.elitism:
                    new_population[0] = best_individual
                
                self.population = new_population[:self.population_size]
        
        print(f"\n{'='*60}")
        print("EVOLUTION COMPLETE")
        print(f"{'='*60}")
        
        return self.get_best_individual()
    
    def get_best_individual(self):
        return max(self.population, key=lambda ind: ind.fitness)
