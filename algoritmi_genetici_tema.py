import math
import random
import matplotlib.pyplot as plt

#input

input_file = open("aa\input_file.txt", "r")
output_file = open("aa\output_file.txt", "w")

nr_crom = int(input_file.readline().strip())
s, d = input_file.readline().strip().split()
s, d = int(s), int(d)
a, b, h = input_file.readline().strip().split()
a, b, h = int(a), int(b), int(h)
precision = int(input_file.readline().strip())
crossover = float(input_file.readline().strip())
mutation = float(input_file.readline().strip())
nr_etape = int(input_file.readline().strip())
best_c = "" #the best cromosome in every generation

#generating the population and the translation
initial_c = [random.random() for _ in range(nr_crom)]
cromozomi = [(c * (d - s) + s) for c in initial_c]


#global variables and utilitarian functions
bit_length = math.ceil(math.log2((d - s) * 10 ** precision))
discrection_step = (d - s) / 2 ** bit_length
max_points = []  #will be used for the visual aid 
avg_values = []

def function(x):
    return a * x**2 + b * x + h

def selection_prob(x, sum_values):
    return function(x) / sum_values

def get_average():
    return sum([function(x) for x in cromozomi]) / nr_crom


#codification and decodification
def codif(x):
    global s
    global d
    
    ci = int(round((x - s) / discrection_step))
    
    codified_x = ""
    while ci > 0:
        rest = ci % 2
        codified_x = str(rest) + codified_x
        ci = ci // 2
        
    while len(codified_x) < bit_length:
        codified_x = "0" + codified_x
        
    return codified_x

def decodif(x):
    return s + int(x, 2) * discrection_step

def codification_step():
    return [codif(crom) for crom in cromozomi]


#selection step 
def cautare_binara(cum_sums, u):
    stanga = 0
    dreapta = nr_crom - 1
    while stanga <= dreapta:
        m = (stanga + dreapta) // 2
        if u >= cum_sums[m] and u < cum_sums[m + 1]:
            return m
        elif u < cum_sums[m]:
            dreapta = m - 1
        else:
            stanga = m + 1
    return nr_crom - 1

def selection_step(cum_sums, values, selection_probs):
    global best_c
    
    cum_sums.clear()
    cum_sums.append(0)
    for i in range(nr_crom):
        cum_sums.append(cum_sums[-1] + selection_probs[i])

    new_population = []
    new_crom = []
    probabilites = []
    
    best_c = values.index(max(values)) # elitist selection 
    best_c = cromozomi[best_c]
    
    for _ in range(nr_crom - 1):
        u = random.random()
        probabilites.append(u)
        
        idx = cautare_binara(cum_sums, u)
        
        new_population.append(cromozomi[idx])
        new_crom.append(idx + 1)
    
    return new_population, new_crom, probabilites

#crossover step
def crossing_2(x, y):
    l = bit_length
    i = random.randint(0, bit_length - 1)
    cx = x[0 : i] + y[i : l]
    cy = y[0 : i] + x[i : l]
    return cx, cy, i

def crossing_3(x, y, z):
    l = bit_length
    i = random.randint(0, bit_length - 1)
    cx = x[0 : i] + y[i : l]
    cy = y[0 : i] + z[i : l]
    cz = z[0 : i] + x[i : l]
    
    return cx, cy, cz, i
    
def crossover_step(codified_c, values):
    probabilites = []
    codif = codified_c.copy() # for the detailed output of the first run
    pairs = []
    for_crossing = []
    
    for i in range(nr_crom - 1):
            u = random.uniform(0, 1)
            probabilites.append(u)
            if u < crossover: 
                for_crossing.append([codified_c[i], i])
                
    i = 0
    if(len(for_crossing) % 2 == 1 and len(for_crossing) != 1):
        x, y, z = for_crossing[i][0], for_crossing[i + 1][0], for_crossing[i + 2][0]
        idx_x, idx_y, idx_z = for_crossing[i][1], for_crossing[i + 1][1], for_crossing[i + 2][1]
        codified_c[idx_x], codified_c[idx_y], codified_c[idx_z], split = crossing_3(x, y, z)
        
        pairs.append([idx_x, idx_y, idx_z, split])
        cromozomi[idx_x], cromozomi[idx_y], cromozomi[idx_z] = decodif(codified_c[idx_x]), decodif(codified_c[idx_y]), decodif(codified_c[idx_z])
        values[idx_x], values[idx_y], values[idx_z] = function(cromozomi[idx_x]), function(cromozomi[idx_y]), function(cromozomi[idx_z])
        
        i = 3
        
    while i < len(for_crossing) and len(for_crossing) != 1:
        x, y = for_crossing[i][0], for_crossing[i + 1][0]
        idx_x, idx_y = for_crossing[i][1], for_crossing[i + 1][1]
        codified_c[idx_x], codified_c[idx_y], split = crossing_2(x, y)
        
        pairs.append([idx_x, idx_y, split])
        cromozomi[idx_x], cromozomi[idx_y] = decodif(codified_c[idx_x]), decodif(codified_c[idx_y])
        values[idx_x], values[idx_y] = function(cromozomi[idx_x]), function(cromozomi[idx_y])
        
        i += 2
    
    return codif, probabilites, codified_c, pairs


#mutation step

def mutation_step(codified_c, values):
    mutated_c = []
    for i in range(nr_crom - 1): 
        has_mutated = False
        for j in range(bit_length): #the probability of mutation is applied for every gene
            u = random.random()
            if u < mutation:
                codified_c[i] = codified_c[i][0 : j] + str(int(not(int(codified_c[i][j])))) + codified_c[i][j + 1 :] 
                has_mutated = True
                
        if has_mutated:
            cromozomi[i] = decodif(codified_c[i])
            values[i] = function(cromozomi[i])
            mutated_c.append(i)
            
    return codified_c, mutated_c

def mutation_step_rare(codified_c, values): #currently using this one
    mutated_c = []
    for i in range(nr_crom - 1): 
        u = random.random()
        if u < mutation:
            position = random.randint(0, bit_length - 1)
            codified_c[i] = codified_c[i][0 : position] + str(int(not(int(codified_c[i][position])))) + codified_c[i][position + 1 :] 
            cromozomi[i] = decodif(codified_c[i])
            values[i] = function(cromozomi[i])
            mutated_c.append(i)
    
    return codified_c, mutated_c


#first run with detailed outputs

def first_run():
    global cromozomi
    codified_c = codification_step()
    values = [function(x) for x in cromozomi]
    sum_values = sum(values)

    selection_probs = [selection_prob(crom, sum_values) for crom in cromozomi]
    cum_sums = []
    
    output_file.write("Populatie initiala\n")
    for i in range(nr_crom):
        output_file.write(f"{i + 1}: {codified_c[i]} x = {cromozomi[i]} f = {values[i]}\n")
    
    output_file.write("\nProbabilitati de selectie\n")
    for i in range(nr_crom):
        output_file.write(f"cromozom {i + 1} probabilitate {selection_probs[i]}\n")
    
    cromozomi, new_crom, probabilities = selection_step(cum_sums, values, selection_probs)
    
    values = [function(x) for x in cromozomi] #after selectionm, the values of the function have to be recalculated
    sum_values = sum(values)
    codified_c = codification_step() #the codifications for the new population (post selection)
    
    output_file.write("\nIntervale probabilitati selectie\n")
    output_file.write(str(cum_sums))
    
    output_file.write("\nSelectii\n")
    for i in range(nr_crom - 1):
        output_file.write(f"u = {probabilities[i]} selectam cromozomul {new_crom[i]}\n")
        
    output_file.write("\nDupa selectie\n")
    for i in range(nr_crom - 1):
        output_file.write(f"{i + 1}: {codified_c[i]} x = {cromozomi[i]} f = {values[i]}\n")
        
    output_file.write(f"\nProbabilitatea de incrucisare {crossover}\n")
    copy_codified, probabilities, codified_c, pairs = crossover_step(codified_c, values)
    
    for i in range(nr_crom - 1):
        output_file.write(f"{i + 1}: {copy_codified[i]} u = {probabilities[i]}")
        if probabilities[i] < crossover:
            output_file.write(f" < {crossover} participa la selectie\n")
        else: output_file.write("\n")
        
    for p in pairs:
        if len(p) == 4:
            output_file.write(f"Recombinare cromozomii {p[0]} si {p[1]} si {p[2]} in punctul {p[3]}\n")
            output_file.write(f"{copy_codified[p[0]]} {copy_codified[p[1]]} {copy_codified[p[2]]}\n rezultat {codified_c[p[0]]} {codified_c[p[1]]} {codified_c[p[2]]}\n")
        
        else: 
            output_file.write(f"Recombinare cromozomii {p[0]} si {p[1]} in punctul {p[2]}\n")
            output_file.write(f"{copy_codified[p[0]]} {copy_codified[p[1]]} \n rezultat {codified_c[p[0]]} {codified_c[p[1]]} \n")
        
    output_file.write(f"\nDupa recombinare\n")
    for i in range(nr_crom - 1):
        output_file.write(f"{i + 1}: {codified_c[i]} x = {cromozomi[i]} f = {function(cromozomi[i])}\n")
        
    output_file.write(f"\nProbabilitatea de mutatie {mutation}\n")
    codified_c, mutated_c = mutation_step_rare(codified_c, values)
    
    output_file.write("Au fost modificati cromozomii:\n")
    for m in mutated_c:
        output_file.write(str(m) + " ")
    
    output_file.write("\nDupa mutatie:\n")
    for i in range(nr_crom - 1):
        output_file.write(f"{i + 1}: {codified_c[i]} x = {cromozomi[i]} f = {function(cromozomi[i])}\n")
    
    cromozomi.append(best_c) 
    codified_c.append(codif(best_c))          
    
    output_file.write(f"{nr_crom}: {codified_c[nr_crom - 1]} x = {cromozomi[nr_crom - 1]} f = {function(cromozomi[nr_crom - 1])}\n") #elistist selection
    
    return max(values)


#a general run

def run():
    codified_c = codification_step()
    
    global cromozomi
    
    values = [function(x) for x in cromozomi]
    sum_values = sum(values)
    selection_probs = [selection_prob(crom, sum_values) for crom in cromozomi]
    cum_sums = []
    
    cromozomi, new_crom, probabilities = selection_step(cum_sums, values, selection_probs) #some of the values are not used as there isn't a detailed output anymore
    
    values = [function(x) for x in cromozomi]
    sum_values = sum(values)                    #the reactualisation of the values of the function and the recodifications
    codified_c = codification_step()
    
    copy_codified, probabilities, codified_c, pairs = crossover_step(codified_c, values)
    codified_c, mutated_c = mutation_step_rare(codified_c, values)
    
    cromozomi.append(best_c)
    codified_c.append(codif(best_c))          
        
    return max(values)


#a graphic interface

def visual_aid():
    plt.plot(max_points, color ="purple", label="Max fitness")
    plt.plot(avg_values, color="magenta", label="Average fitness")
    plt.xlabel("Generatie")
    plt.ylabel("Fitness")
    plt.title("Evolutia algoritmului genetic")
    plt.legend()
    plt.grid(True)
    plt.show()


#the "main" of the programme

max_ = first_run()
avg_values.append(get_average())
max_points.append(max_)
output_file.write(f"\nEvolutia maximului\nmaxim {max_} - average {get_average()}\n")

for _ in range(nr_etape - 1):
    max_ = run()
    avg_values.append(get_average())
    max_points.append(max_)
    output_file.write(f"maxim {max_} - average {get_average()}\n")

visual_aid()

input_file.close()
output_file.close()