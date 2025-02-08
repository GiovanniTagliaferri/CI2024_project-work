from functions import *

# set the parameters according to the problem
depth = 3
pop_size = 100
num_epochs = 100
problem_id = 1

best_formula = run_evolution(num_epochs, pop_size, problem_id, depth)
print(f"Best formula found: {best_formula}"
      f"\nFitness: {best_formula.fitness:4e}")
