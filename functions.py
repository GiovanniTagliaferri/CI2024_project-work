import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt


# Define Binary and Unary Operations
BINARY_OP = [np.add, np.subtract, np.multiply, np.divide, np.power]
UNARY_OP = [np.sin, np.cos, np.log, np.exp, np.sqrt, np.tan, np.sinh, 
            np.cosh, np.tanh, np.arcsin, np.arccos, np.arctan]

# Mapping Operations to Symbols
OPERATIONS = {
    np.add: "+", np.subtract: "-", np.multiply: "*", np.divide: "/", 
    np.power: "^",
    np.sin: "sin", np.cos: "cos", np.log: "log", np.exp: "exp", 
    np.sqrt: "sqrt",
    np.tan: "tan", np.sinh: "sinh", np.cosh: "cosh", np.tanh: "tanh", 
    np.arcsin: "arcsin", np.arccos: "arccos", np.arctan: "arctan"
}


class Node:
    "Class that represents a component of a formula, such as variable, constant or operation."
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right  # Only used for binary operations

    def is_leaf(self):
        return self.left is None and self.right is None

    def is_unary(self):
        return self.value in UNARY_OP and self.left is not None and self.right is None

    def is_binary(self):
        return self.value in BINARY_OP and self.left is not None and self.right is not None

    def evaluate(self, variables):
        if self.is_leaf():
            return variables[self.value] if isinstance(self.value, str) else self.value
        
        if self.is_unary():
            return self.value(self.left.evaluate(variables))
        
        if self.is_binary():
            left_value = self.left.evaluate(variables)
            right_value = self.right.evaluate(variables)

            return self.value(left_value, right_value)
        
        try:
            result = self.value(self.left.evaluate(variables))
            if np.any(np.isinf(result)) or np.any(np.isnan(result)):
                return None
            return np.clip(result, -1e6, 1e6)
        except (ZeroDivisionError, ValueError, FloatingPointError):
            return None


    def to_formula(self):
        if self.is_leaf():
            return str(self.value)  # Return variable name or constant

        if self.is_unary():  # unary operation
            return f"{OPERATIONS[self.value]}({self.left.to_formula()})"

        if self.is_binary():  # binary operation
            return f"({self.left.to_formula()} {OPERATIONS[self.value]} {self.right.to_formula()})"

        raise ValueError(f"Invalid node structure: {self.value}")
    
    def count_nodes(self):
        left_count = self.left.count_nodes() if self.left else 0
        right_count = self.right.count_nodes() if self.right else 0
        return 1 + left_count + right_count  # count current node + children

class Formula:
    "Class that represents a formula as a tree of Nodes where each one is an operation, a constant or a variable."
    def __init__(self, root=None, X_dicts=None, y=None):
        self.root = root
        self.fitness = None
        self.num_nodes = 0

        self.compute_nodes()  # update node count

        # Compute fitness if dataset is provided (otherwise fitness is None)
        if X_dicts is not None and y is not None:
            self.compute_fitness(X_dicts, y)


    def evaluate(self, X_values):
        return self.root.evaluate(X_values)

    def to_formula(self):
        return self.root.to_formula() if self.root else "Empty Tree"

    def compute_nodes(self):
        self.num_nodes = self.root.count_nodes() if self.root else 0

    def compute_fitness(self, X_dicts, y):
        if self.root is None:  # Avoid computing fitness for empty trees
            self.fitness = float('inf')
            return self.fitness

        predictions = np.array([self.evaluate(X_dict) for X_dict in X_dicts])

        # check for invalid predictions: even if one is NaN or inf, fitness is inf
        if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
            self.fitness = float('inf')
            return self.fitness
        
        # compute the mean squared error
        self.fitness = 100* np.mean((predictions - y) ** 2)
        return self.fitness

    def update(self, X_dicts, y):
        self.compute_nodes()  # Ensure node count is up-to-date
        self.compute_fitness(X_dicts, y)  # Ensure fitness is accurate

# =============== FUNCTIONS ===============
def select_random_node(node, parent=None):
    "Select a random Node in a Formula by passing the root node, returning both the chosen node and its parent."
    if node is None or (node.is_leaf() and random.random() < 0.5):  # Select leaf nodes less frequently
        return node, parent

    if random.random() < 0.5 and node.left is not None:  # Randomly go left
        return select_random_node(node.left, node)
    elif node.right is not None:  # Otherwise, go right
        return select_random_node(node.right, node)

    return node, parent  # Return the node and its parent

# =============== GENERATION ===============
def generate_random_tree(variables, X_min, X_max, max_depth=3, p_const=0.2):
    "Return the root node of a tree (Formula) generated randomly, ensuring that it is valid."

    if max_depth == 0 or (random.random() < p_const):  
        # Generate a leaf node (a variable or a constant)
        if random.random() < 0.5:
            return Node(random.choice(variables))  # Variable node
        return Node(np.random.uniform(X_min, X_max))  # Constant node

    if random.random() < 0.5:  # Binary operation (must have two children)
        op = random.choice(BINARY_OP)
        
        left_child = generate_random_tree(variables, X_min, X_max, max_depth - 1, p_const)
        right_child = generate_random_tree(variables,X_min, X_max, max_depth - 1, p_const)
        
        return Node(op, left_child, right_child)

    # Unary operation (must have exactly one child)
    op = random.choice(UNARY_OP)
    return Node(op, generate_random_tree(variables, X_min, X_max, max_depth - 1, p_const), None)


def generate_population(population_size, variables, X_dicts, y, X_min, X_max, max_depth=3):
    "Generate a population of random valid generated Formula elements."
    population = []
    while len(population) < population_size:
        tree = generate_random_tree(variables, X_min, X_max, max_depth)
        formula = Formula(tree, X_dicts, y)
        if formula.fitness is not None and not np.isinf(formula.fitness):
            population.append(formula)
    
    return population

# =============== SELECTION ==================
def tournament_selection(population, tournament_size=5, elite_perc=0.8, size_bias_prob=0.3):
    "Perform tournament selection to select a parent."

    # Sort population by fitness (lower fitness is better)
    sorted_population = sorted(population, key=lambda tree: tree.fitness)

    # Determine elite and non-elite sizes
    elite_size = max(1, int(elite_perc * tournament_size))  # Ensure at least 1 elite individual
    non_elite_size = tournament_size - elite_size

    # Select elite candidates (best fitness individuals)
    elite_candidates = sorted_population[:elite_size]  

    # Select non-elite candidates randomly from the rest of the population
    non_elite_candidates = random.sample(sorted_population[elite_size:], non_elite_size) if len(sorted_population) > elite_size else []

    # Combine both groups into the tournament pool
    tournament_pool = elite_candidates + non_elite_candidates

    # Introduce size bias: select the individual with the least nodes with probability `size_bias_prob`
    if random.random() < size_bias_prob:
        return min(tournament_pool, key=lambda tree: tree.num_nodes)  # Select the smallest tree
    else:
        return random.choice(tournament_pool)  # Default: Random selection from the tournament

# =============== MUTATION ===============
def subtree_mutation(tree, variables, X_dicts, y, X_min, X_max):
    "Perform subtree mutation of a Formula by mutating a random part of it."
    tree_copy = copy.deepcopy(tree)
    node_to_mutate, parent = select_random_node(tree_copy.root)

    if node_to_mutate is None:
        return tree_copy

    new_subtree = generate_random_tree(variables, X_min, X_max)

    if parent is None:
        tree_copy.root = new_subtree
    else:
        if parent.left == node_to_mutate:
            parent.left = new_subtree
        elif parent.right == node_to_mutate:
            parent.right = new_subtree

    tree_copy.update(X_dicts, y)
    return tree_copy


def point_mutation(tree, variables, X_dicts, y, X_min, X_max):
    "Perform point mutation of a Formula by mutation a random node of it."
    tree_copy = copy.deepcopy(tree)
    node_to_mutate, _ = select_random_node(tree_copy.root)

    if node_to_mutate is None:
        return tree_copy  # No mutation happened

    # Apply mutation based on node type
    if node_to_mutate.is_binary():
        # Change to a different binary operator
        node_to_mutate.value = random.choice([op for op in BINARY_OP if op != node_to_mutate.value])
    
    elif node_to_mutate.is_unary():
        # Change to a different unary operator
        node_to_mutate.value = random.choice([op for op in UNARY_OP if op != node_to_mutate.value])

    elif node_to_mutate.is_leaf():
        if isinstance(node_to_mutate.value, str):  # Variable node
            # Change to another variable (ensuring it’s different)
            node_to_mutate.value = random.choice([var for var in variables if var != node_to_mutate.value])
        
        elif isinstance(node_to_mutate.value, (int, float)):  # Constant node
            # Slightly modify the constant while keeping it within `X_min` and `X_max`
            constant_shift = random.uniform(-0.5, 0.5) * (X_max.max() - X_min.min())  # Scale change to data range
            new_value = node_to_mutate.value + constant_shift
            node_to_mutate.value = np.clip(new_value, X_min.min(), X_max.max())  # Ensure valid range

    # Compute fitness after mutation
    tree_copy.update(X_dicts, y)

    return tree_copy


# ================== CROSSOVER ==================
def crossover(parent1, parent2, X_dicts, y):
    "Perform crossover between two Formula to generate a child one."
    parent1_copy = copy.deepcopy(parent1)
    parent2_copy = copy.deepcopy(parent2)

    # Select random subtrees
    parent_subtree, parent_parent = select_random_node(parent1_copy.root)
    donor_subtree, _ = select_random_node(parent2_copy.root)

    if parent_subtree is None or donor_subtree is None:
        return parent1_copy  # No crossover if selection fails

    # Perform the subtree replacement
    if parent_parent is None:
        parent1_copy.root = donor_subtree  # Replace entire tree if needed
    else:
        if parent_parent.left == parent_subtree:
            parent_parent.left = donor_subtree
        elif parent_parent.right == parent_subtree:
            parent_parent.right = donor_subtree

    # Update node count
    parent1_copy.update(X_dicts, y)

    return parent1_copy  # Returns a single offspring

# ================== EVOLUTION ==================
def offspring_generation(population, variables, X_dicts, y, X_min, X_max, tournament_size=5, elite_perc=0.5, mutation_rate=0.3, crossover_rate=0.7):
    "Function that wraps the process to generate the offspring."

    # Select first parent
    parent1 = tournament_selection(population, tournament_size)

    # Perform crossover (with probability)
    if random.random() < crossover_rate:
        parent2 = tournament_selection(population, tournament_size)
        offspring1 = crossover(parent1, parent2, X_dicts, y)  # Apply crossover
    else:
        offspring1 = copy.deepcopy(parent1)  # No crossover, just copy the parent

    # change mutation depending on the probability
    if random.random() < mutation_rate:
        if random.random() < 0.5:
            offspring1 = point_mutation(offspring1, variables, X_dicts, y, X_min, X_max)
        else:
            offspring1 = subtree_mutation(offspring1, variables, X_dicts, y, X_min, X_max)


    # evaluate the offspring: if the fitness is inf
    if np.isinf(offspring1.fitness) or np.isnan(offspring1.fitness):
        return None
    
    return offspring1

def plot_predictions(best_tree, variables, X, y_true, problem_id):
    "Plot together the distribution of the original output and the one obtained by applying the generated Formula on the inputs."
    X_dicts = [dict(zip(variables, row)) for row in X.T]
    y_pred = np.array([best_tree.evaluate(X_dict) for X_dict in X_dicts])

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    if problem_id != 1:
        ax.scatter(X[0], X[1], y_true, color="red", label="True Values", alpha=0.6)
        ax.scatter(X[0], X[1], y_pred, color="blue", label="Predicted Values", alpha=0.6)
        ax.set_ylabel("X1")
    else:
        ax.scatter(X[0], y_true, color="red", label="True Values", alpha=0.6)
        ax.scatter(X[0], y_pred, color="blue", label="Predicted Values", alpha=0.6)

    # Labels
    ax.set_xlabel("X0")
    ax.set_zlabel("Y")
    ax.set_title(f"Problem {problem_id} - Predictions vs True Values")
    ax.legend()

    # plt.savefig(f"predictions/problem_{problem_id}_predictions.png")
    plt.show()

def run_evolution(num_epochs, population_size, problem_id, max_depth=4, 
                  tournament_size=5, elite_perc=0.8, mutation_rate=0.7, crossover_rate=0.3):
    "Function that performs the whole evolution process and returns the final Formula."
    
    print(f"Finding formula for problem {problem_id}")

    data = np.load(f"data/problem_{problem_id}.npz")

    X = data['x']  # input array
    y = data['y']  # output array
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    X_max = X.max(axis=1).max()
    X_min = X.min(axis=1).min()

    num_variables = X.shape[0]
    variables = [f"x{i}" for i in range(0, num_variables)]
    X_dicts = [dict(zip(variables, row)) for row in X.T]
    
    # Track the best formula across all generations
    best_overall = None
    best_fitness = float('inf')

    # Initialize Population
    population = generate_population(population_size, variables, X_dicts, y, X_min, X_max, max_depth)
    print(f"Initial Population Size: {len(population)}")

    # number of children to generate at each generation (60% of the population)
    num_children = int(0.6 * population_size)

    # Run the evolution
    for epoch in tqdm(range(num_epochs), desc="Evolving Population", unit="gen"):
        # Dynamic mutation & crossover adjustments: prioritize mutation early, shift to crossover later
        if epoch == int(0.4 * num_epochs):
            mutation_rate = 0.3
            crossover_rate = 0.7
            print(f"\n-- Adjusted Mutation Rate: {mutation_rate}, Crossover Rate: {crossover_rate}")

        # Ensure valid offspring and maintain population size
        new_population = []

        while len(new_population) < num_children:
            offspring = offspring_generation(population, variables, X_dicts, y, X_min, X_max, tournament_size, elite_perc, mutation_rate, crossover_rate)
            if offspring is not None:
                new_population.append(offspring)
        
        # complete the population generating random trees to balance exploration and exploitation
        if len(new_population) < population_size:
            new_population.extend(list(generate_population(population_size - len(new_population), variables, X_dicts, y, X_min, X_max, max_depth)))
        
        # Replace old population with new one
        population = new_population

        best_individual = min(population, key=lambda tree: tree.fitness)

        # Print progress
        tqdm.write(
                   f"\n-- Best Formula: {best_individual.root.to_formula()} "
                   f"\n-- Best Fitness: {best_individual.fitness:.4e}")

        # Update best solution found so far
        if best_individual.fitness < best_fitness:
            best_fitness = best_individual.fitness
            best_overall = copy.deepcopy(best_individual)

    plot_predictions(best_overall, variables, X, y, problem_id)

    # return the best formula found
    return best_overall
