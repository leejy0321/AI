import random
import matplotlib.pyplot as plt

answer = [3, 1, 4, 3, 4, 1, 4, 1, 2, 2]

init_population = []
random.seed()
population_size = 10

for i in range(population_size):
    new_chromosome = []
    for j in range(population_size):
        new_chromosome.append(random.randint(1, 4))
    init_population.append(new_chromosome)

def fitness_function(individual):
    fitness = 0
    for i in range(0, len(individual)):
        if individual[i] == answer[i]:
            fitness += 10
    return fitness

def fitness_proportionate_selection(population):
    fitness_list = []
    fitness_sum = 0
    for i in range(0, len(population)):
        fitness_list.append(fitness_function(population[i]))
        fitness_sum += fitness_function(population[i])
    probability_list = []
    cumulative_value = 0
    for i in range(0, len(fitness_list)):
        try:
            cumulative_value += fitness_list[i] / fitness_sum
            probability_list.append(cumulative_value)
        except ZeroDivisionError:
            pass
    probability_list[-1] = 1.0
    random_number = random.random()
    for i, probability in enumerate(probability_list):
        if random_number <= probability:
            return population[i]
            break

def tournament_selection(population):
    fitness_list = []
    fitness_with_index = []
    probability_list = []
    cumulative_value = 0
    p = 0.7
    max = 0
    for i in range(0, len(population)):
        fitness_list.append(fitness_function(population[i]))
    for i, fitness in enumerate(fitness_list):
        fitness_with_index.append((i, fitness))
    fitness_with_index.sort(key=lambda tuple: tuple[1])
    random_number = random.random()
    for i in range(0, len(population)):
        cumulative_value += p*(1-p)**i
        probability_list.append(cumulative_value)
    probability_list[-1] = 1.0
    for i, probability in enumerate(probability_list):
        if random_number <= probability:
            (index, index_fitness) = fitness_with_index[len(population)-i-1]
            break
    return population[index]

def crossover(parent1, parent2):
    crosspoint = random.randint(0, 9)
    child = parent1[crosspoint:] + parent2[:crosspoint]
    return child

def mutate(individual):
    random_index = random.randint(0, 9)
    individual[random_index] = random.randint(1, 4)
    return individual

def genetic_algorithm(population):
    crossover_rate = 0.15
    mutation_rate = 0.005
    generation = 0
    avg_fitness_list = []
    while(generation < 1000):
        new_population = []
        fitness_sum = 0
        generation += 1
        n = 0
        for i in range(0, len(population)):
            x = fitness_proportionate_selection(population) # fitness proportionate selection을 사용하는 경우
            # x = tournament_selection(population) # tournament selection을 사용하는 경우
            child = x
            y = child
            while(x==y):
                y = fitness_proportionate_selection(population) # fitness proportionate selection을 사용하는 경우
                # y = tournament_selection(population) # tournament selection을 사용하는 경우
                n += 1
                if n > 100:
                    break
            if random.random() < crossover_rate:
                child = crossover(x, y)
            if random.random() < mutation_rate:
                child = mutate(child)
            new_population.append(child)
        population = new_population
        for i in range(0, len(population)):
            fitness_sum += fitness_function(population[i])
            if fitness_function(population[i]) == 100:
                print("정답이 포함된 population:", population)
                print("정답이 찾아진 generation:", generation)
                break
        avg_fitness = fitness_sum / 10
        avg_fitness_list.append(avg_fitness)
    plt.plot(range(len(avg_fitness_list)), avg_fitness_list)
    plt.xlabel('Generation')
    plt.ylabel('average fitness')
    plt.title('Result')
    plt.show()

genetic_algorithm(init_population)