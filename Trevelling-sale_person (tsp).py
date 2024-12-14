import random
import math
POPULATION_SIZE = 100
MUTATION_RATE = 0.02
NUM_GENERATIONS = 1000
NUM_CITIES = 10
cities = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(NUM_CITIES)]

def calculate_distance(route):
    distance = 0
    for i in range(len(route) - 1):
        distance += math.dist(cities[route[i]], cities[route[i + 1]])
    distance += math.dist(cities[route[-1]], cities[route[0]])  
    return distance


def fitness(route):
    return 1 / calculate_distance(route)


def create_population():
    return [random.sample(range(NUM_CITIES), NUM_CITIES) for _ in range(POPULATION_SIZE)]


def selection(population):
    fitnesses = [fitness(route) for route in population]
    total_fitness = sum(fitnesses)
    probabilities = [f / total_fitness for f in fitnesses]
    parents = random.choices(population, weights=probabilities, k=2)
    return parents


def crossover(parent1, parent2):
    start, end = sorted(random.sample(range(NUM_CITIES), 2))
    child = [-1] * NUM_CITIES
    child[start:end] = parent1[start:end]
    
    pointer = end
    for city in parent2:
        if city not in child:
            if pointer >= NUM_CITIES:
                pointer = 0
            child[pointer] = city
            pointer += 1
    return child


def mutate(route):
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(NUM_CITIES), 2)
        route[i], route[j] = route[j], route[i]
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
    if generation % 100 == 0:
        best_route = min(population, key=calculate_distance)
        print(f"Generation {generation}: Shortest distance = {calculate_distance(best_route)}")

best_route = min(population, key=calculate_distance)
print("Best route found:", best_route)
print("Shortest distance:", calculate_distance(best_route))
