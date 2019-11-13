import numpy as np
from matplotlib import pyplot as plt

minim = 0
maxim = 128
population_size = 64
variables = 2

y = []
x = []


def rosenbrock(a):
    return ((100 * (((a[0] ** 2) - a[1]) ** 2)) + ((a[0] - 1) ** 2))


def selectOne(a):
    max = sum([i for i in a])
    selection_probs = [i/max for i in a]
    return a[np.random.choice(len(a), p=selection_probs)]


def selectPool(population, ratio):
    max = sum([i for i in ratio])
    selection_probs = [i / max for i in ratio]
    a = population[np.random.choice(population.shape[0], population_size, p=selection_probs, replace=True)]
    return a


def cross(ind1, ind2):
    cx = np.random.randint(1, 7)
    ans1 = ind1[:cx]+ind2[cx:]
    return ans1


def mutate(a):
    cx = np.random.randint(0, len(a))
    if (a[cx] == '1'):
        a = a[:cx] + '0' + a[cx+1:]
    else:
        a = a[:cx] + '1' + a[cx + 1:]
    return a


def nextGen(population):
    population_fx = np.array([rosenbrock(i) for i in population])
    population_fx_sum = 0
    for i in population_fx:
        population_fx_sum += i
    population_fx_average = population_fx_sum / population_size
    y.append(population_fx_average)
    population_fx_ratio = np.array([i / population_fx_average for i in population_fx])
    pool = selectPool(population, population_fx_ratio)
    mate = np.random.permutation(population_size)
    pool_mate = np.array([pool[i] for i in mate])
    child = []
    for i, j in zip(pool, pool_mate):
        child_x = cross("{0:07b}".format(i[0]), "{0:07b}".format(j[0]))
        child_y = cross("{0:07b}".format(i[1]), "{0:07b}".format(j[1]))
        child.append([int(child_x, 2), int(child_y, 2)])
    child = np.array(child)
    for i in child:
        if (np.random.random() <= 0.01):
            i[0] = int(mutate("{0:07b}".format(i[0])), 2)
            i[1] = int(mutate("{0:07b}".format(i[1])), 2)

    return child


initial_population = np.array([np.random.randint(minim, maxim, size=variables) for i in range(population_size)])

for i in range(100):
    x.append(i)
    initial_population = nextGen(initial_population)
print(initial_population[0])
plt.plot(x, y)
plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.title('Fitness over time')
plt.savefig("my_graph.png")
