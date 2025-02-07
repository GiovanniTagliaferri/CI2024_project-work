from functions import *

depth = 3
pop_size = 100
num_epochs = 50
tournament_size = 5
problem_id = 2

# Load Problem data
data = np.load(f"data/problem_{problem_id}.npz")

X = data['x']  # input array
y = data['y']  # output array

X_max = X.max(axis=1).max()
X_min = X.min(axis=1).min()

num_variables = X.shape[0]
variables = [f"x{i}" for i in range(0, num_variables)]
X_dicts = [dict(zip(variables, row)) for row in X.T]

best_formula = run_evolution(num_epochs, pop_size, problem_id, depth, tournament_size)
print(f"Best formula found: {best_formula}"
      f"\nFitness: {best_formula.fitness:4e}")
