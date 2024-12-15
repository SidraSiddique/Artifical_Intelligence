import random
def initialize_population(pop_size, string_length):
    return [''.join(random.choice('01') for _ in range(string_length)) for _ in range(pop_size)]
def calculate_fitness(individual):
    return individual.count('1')
def select_parents(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    probabilities = [fitness / total_fitness for fitness in fitness_scores]
    parent1 = random.choices(population, weights=probabilities, k=1)[0]
    parent2 = random.choices(population, weights=probabilities, k=1)[0]
    return parent1, parent2
def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    return parent1[:crossover_point] + parent2[crossover_point:]
def mutate(individual, mutation_rate):
    return ''.join(
        bit if random.random() > mutation_rate else str(1 - int(bit)) for bit in individual
    )
def genetic_algorithm(string_length, pop_size, num_generations, mutation_rate):
    population = initialize_population(pop_size, string_length)

    for generation in range(num_generations):
        fitness_scores = [calculate_fitness(ind) for ind in population]
        best_fitness = max(fitness_scores)
        best_individual = population[fitness_scores.index(best_fitness)]
        print(f"Generation {generation + 1}: Best Fitness = {best_fitness}, Best Individual = {best_individual}")
        if best_fitness == string_length:
            print("Optimal solution found!")
            break
        new_population = []
        for _ in range(pop_size):
            parent1, parent2 = select_parents(population, fitness_scores)
            offspring = crossover(parent1, parent2)
            offspring = mutate(offspring, mutation_rate)
            new_population.append(offspring)

        population = new_population 
string_length = 10  
pop_size = 20       
num_generations = 50  
mutation_rate = 0.1  
genetic_algorithm(string_length, pop_size, num_generations, mutation_rate)
