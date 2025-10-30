class EC:
    Decoder = None
    population = []
    population_size = 100
    population_fitness = []
    selected_parents = []
    fitness_selected_parents = []
    offsprings = []
    fitness_offsprings = []
    generation = 100
    has_crossover = True
    has_mutation = True

    def __init__(individuals=100,generations=100,has_crossover=True,has_mutation=True):
        self.generation=geneartions
        self.population_size = individuals
        self.initialize_population()
        self.has_crossover = has_crossover
        self.has_mutation = has_mutation

    def initialize_individual():
        return None

    def initialize_population():
        self.population=[]
        for i in range(0.self.population_size):
            self.population=np.append(self.population,initialize_individual())
    
    def evaluate_individual(indice)
        return 0

    def evalutate_population():
        self.population_fitness=[]
        for i in range(0,self.population_size):
            self.population_fitness=np.append(self.population_fitness,evalute_individual(i))

    def parent_selection():
        self.selected_parents = []
        self.fitness_selected_parents = []
        for i in range(0,self.population_size):
            parent_1_id = np.choice(range(0,self.population_size))
            parent_2_id = np.choice(range(0,self.population_size))
            
            parent_1 = population(parent_1_id)
            fitness_parent_1 = population_fitness(parent_1_id)
            parent_2 = population(parent_2_id)
            fitness_parent_2 = population_fitness(parent_2_id)

            if fitness_parent_1>=fitness_parent_2:
                self.selected_parents=np.append(self.selected_parents,parent_1)
                self.fitness_selected_parents=np.append(self.fitness_selected_parents,fitness_parent_1)
            else:
                self.selected_parents=np.append(self.selected_parents,parent_2)
                self.fitness_selected_parents=np.append(self.fitness_selected_parents,fitness_parent_2)

    def crossover_individual(parent_1,parent_2):
        return parent_1,parent_2
    
    def evaluate_offspring(offspring):
        return 0

    def crossover_population():
        self.offsprings=[]
        self.fitness_offstrings=[]
        for i in range(0,self.selected_parents,2):
            parent_1=self.selected_parents[i]
            parent_2=self.selected_parents[2]
            offspring1,offspring2=self.crossover_individual(parent1,parent2)
            self.offsprings=np.append(self.offsprings,offspring1)
            self.offsprings=np.append(self.offsprings,offspring2)
            self.fitness_offsprings=np.append(self.fitness_offsprings,offspring1)
            self.fitness_offsprings=np.append(self.fitness_offsprings,offspring2)
    
    def selection_survivors():
        for i in range(0,self.population_size):
            self.population=np.append(self.population,self.offsprings[i])
            self.population_fitness=np.append(self.population_fitness,self.fitness_offsprings[i])
        order = np.argsort(self.population_fitness)
        self.population=self.population[order]
        self.population_fitness=self.population_fitness[order]
        self.population=self.population[:100]
        self.population_fitness=self.population_fitness[0:100]

    def mutate_individual(i):
        return self.offsprings[i]

    def mutate_offspring():
        for i in range(0,self.population_size):
            self.offsprings[i]=self.mutate_offspring(i)

    def run_ec():
        self.initialize_population()
        for i in range(0,self.generation):
            self.parent_selection()
            self.crossover_population()
            self.mutate_offspring()
            self.selection_survivors()    
