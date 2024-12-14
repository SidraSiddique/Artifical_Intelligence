import random
POPULATION_SIZE = 50
MUTATION_RATE = 0.01
NUM_GENERATIONS = 500
WEIGHT_LIMIT = 50
ITEMS = [(random.randint(1, 10), random.randint(1, 30)) for _ in range(20)]  
def fitness(chromosome):
    weight = sum(ITEMS[i][0] * chromosome[i] for i in range(len(chromosome)))
    value = sum(ITEMS[i][1] * chromosome[i] for i in range(len(chromosome)))
    return value if weight <= WEIGHT_LIMIT else 0
def create_population():
    return [[random.randint(0, 1) for _ in range(len(ITEMS))] for _ in range(POPULATION_SIZE)]
def selection(population):
    fitnesses = [fitness(chromosome) for chromosome in population]
    total_fitness = sum(fitnesses)
    probabilities = [f / total_fitness for f in fitnesses]
    parents = random.choices(population, weights=probabilities, k=2)
    return parents
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    return parent1[:point] + parent2[point:]
def mutate(chromosome):
    for i in range(len(chromosome)):
        if random.random() < MUTATION_RATE:
            chromosome[i] = 1 - chromosome[i]
population = create_population()

for generation in range(NUM_GENERATIONS):
    new_population = []
    for _ in range(POPULATION_SIZE // 2):
        parent1, parent2 = selection(population)
        child1, child2 = crossover(parent1, parent2), crossover(parent2, parent1)
        mutate(child1)
        mutate(child2)
        new_population.extend([child1, child2])

    population = new_population
    if generation % 50 == 0:
        best_solution = max(population, key=fitness)
        print(f"Generation {generation}: Best fitness = {fitness(best_solution)}")

best_solution = max(population, key=fitness)
print("Best solution found:", best_solution)
print("Maximum value:", fitness(best_solution))
