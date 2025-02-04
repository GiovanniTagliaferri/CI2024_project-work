from functions import *

depth = 6
pop_size = 50
num_epochs = 100
tournament_size = 5
problem_id = 5

# Load Problem data
data = np.load(f"data/problem_{problem_id}.npz")

X = data['x']  # input array
y = data['y']  # output array
# print(f"X shape: {X.shape}, y shape: {y.shape}")

# X_max = X.max(axis=1).max()
# X_min = X.min(axis=1).min()

num_variables = X.shape[0]
variables = [f"x{i}" for i in range(0, num_variables)]
X_dicts = [dict(zip(variables, row)) for row in X.T]

# best_formula = run_evolution(num_epochs, pop_size, problem_id, depth, tournament_size)
# print(f"Best formula found: {best_formula}"
#       f"\nFitness: {best_formula.fitness:4e}")

best_formula = ExpressionTree(
    Node(np.divide,
        Node(np.divide,
            Node(np.divide,
                Node(np.power, Node(np.arctan, Node("x0")), Node("x0")),
                Node(np.subtract, Node("x0"), Node(np.sinh, Node(2.767577882313156)))
            ),
            Node(np.add,
                Node(np.multiply,
                    Node(np.arccos, Node(np.cos, Node("x0"))),
                    Node(np.arccos,
                        Node(np.divide,
                            Node(np.arctan,
                                Node(np.divide,
                                    Node(3.793637705721765),
                                    Node(np.arcsin, Node(0.8377975771957574))
                                )
                            ),
                            Node(np.arctan,
                                Node(np.power,
                                    Node(2.646831671472668),
                                    Node(np.power, Node("x0"), Node(np.sqrt, Node(1.667586240087433)))
                                )
                            )
                        )
                    )
                ),
                Node(np.add,
                    Node(np.power,
                        Node(np.arcsin, Node(np.power, Node("x0"), Node("x0"))),
                        Node(np.exp, Node(np.sin, Node("x1")))
                    ),
                    Node(np.sinh, Node(np.sinh, Node(2.538429135351558)))
                )
            )
        ),
        Node(np.divide,
            Node(np.log, Node(np.cos, Node(3.6566196517772416))),
            Node(np.add,
                Node(np.power,
                    Node(np.divide,
                        Node("x0"),
                        Node(np.subtract,
                            Node(np.add,
                                Node(2.819701944034699),
                                Node(3.6726338594193213)
                            ),
                            Node(np.add, Node("x1"), Node(1.091559517201353))
                        )
                    ),
                    Node(np.multiply, Node(0.2011057505016626), Node(3.6566196517772416))
                ),
                Node(np.tanh, Node("x0"))
            )
        )
    ),
    X_dicts,
    y
)





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


