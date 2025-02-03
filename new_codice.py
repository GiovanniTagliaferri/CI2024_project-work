from new_functions_copy import *

depth = 3
pop_size = 500
num_epochs = 100
tournament_size = 5
problem_id = 6

# Load Problem data
data = np.load(f"data/problem_{problem_id}.npz")

X = data['x']  # input array
y = data['y']  # output array

X_max = X.max(axis=1).max()
X_min = X.min(axis=1).min()

num_variables = X.shape[0]
variables = [f"x{i}" for i in range(0, num_variables)]
X_dicts = [dict(zip(variables, row)) for row in X.T]

best_formula = run_evolution(num_epochs, pop_size, variables, X_dicts, y, X_min, X_max, depth, tournament_size)
print(f"Best formula found: {best_formula}"
      f"\nFitness: {best_formula.fitness}")

# plot formula
plot_predictions(best_formula, variables, X, y, problem_id)





# # Example Trees
# tree1 = ExpressionTree(
#     Node(np.add, 
#          Node(np.sin, Node("x0")), 
#          Node(np.multiply, Node("x1"), Node(2))), X_dicts, y
# )

# tree2 = ExpressionTree(
#     Node(np.subtract, 
#          Node(np.cos, Node("x1")), 
#          Node(np.add, Node("x0"), Node(3))), X_dicts, y
# )

# # Perform crossover
# offspring1 = crossover(tree1, tree2, X_dicts, y)

# # Display structures before and after crossover
# print("Parent 1:")
# print(tree1.to_formula(), tree1.fitness, tree1.num_nodes)
# print("\nParent 2:")
# print(tree2.to_formula(), tree2.fitness, tree2.num_nodes)
# print("\nOffspring:")
# print(offspring1.to_formula(), offspring1.fitness, offspring1.num_nodes)

# # Perform mutation
# offspring1 = hoist_mutation(offspring1, X_dicts, y)
# print("\nMutated Offspring:")
# print(offspring1.to_formula(), offspring1.fitness, offspring1.num_nodes)


