import numpy as np
import random
import warnings
import copy
from tqdm import tqdm
from joblib import Parallel, delayed
import torch

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
    def __init__(self, value, left=None, right=None):
        """
        Represents a node in the expression tree.

        :param value: Can be a binary/unary operator (from OPERATIONS) or a variable/constant.
        :param left: Left child node (None if leaf).
        :param right: Right child node (None for unary ops and leaves).
        """
        self.value = value
        self.left = left
        self.right = right  # Only used for binary operations

    def is_leaf(self):
        """Checks if the node is a leaf (variable or constant)."""
        return self.left is None and self.right is None

    def is_unary(self):
        """Checks if the node is a unary operator."""
        return self.value in UNARY_OP and self.left is not None and self.right is None

    def is_binary(self):
        """Checks if the node is a binary operator."""
        return self.value in BINARY_OP and self.left is not None and self.right is not None

    def evaluate(self, variables):
        """
        Recursively evaluates the expression tree while suppressing runtime warnings and handling overflows.

        :param variables: Dictionary mapping variable names to numerical values.
        :return: Computed numerical value (with safety checks).
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # Suppress NumPy runtime warnings

            if self.is_leaf():
                return variables[self.value] if isinstance(self.value, str) else self.value

            if self.is_unary():
                result = self.value(self.left.evaluate(variables))
            elif self.is_binary():
                left_value = self.left.evaluate(variables)
                right_value = self.right.evaluate(variables)

                # Prevent division by zero
                if self.value == np.divide and abs(right_value) < 1e-6:
                    return 1  # Safe default value

                result = self.value(left_value, right_value)
            else:
                raise ValueError(f"Invalid node structure: {self}")

            # **Handle overflow and extreme values**
            if np.isinf(result) or np.isnan(result):  # If result is infinity or NaN, replace with a safe value
                return float(1e6)  # Safe upper bound for large values

            result = np.clip(result, -1e6, 1e6)
            return float(result)

    def to_formula(self):
        """
        Recursively converts the tree into a human-readable mathematical formula as a string.
        """
        if self.is_leaf():
            return str(self.value)  # Return variable name or constant

        if self.is_unary():  # Unary operation (e.g., sin, cos)
            return f"{OPERATIONS[self.value]}({self.left.to_formula()})"

        if self.is_binary():  # Binary operation (e.g., +, -, *, /)
            return f"({self.left.to_formula()} {OPERATIONS[self.value]} {self.right.to_formula()})"

        raise ValueError(f"Invalid node structure: {self.value}")
    
    def count_nodes(self):
        """Recursively counts the total number of nodes in the tree."""
        left_count = self.left.count_nodes() if self.left else 0
        right_count = self.right.count_nodes() if self.right else 0
        return 1 + left_count + right_count  # Count current node + children

class ExpressionTree:
    def __init__(self, root=None, X_dicts=None, y=None):
        """
        Represents the expression tree and automatically computes fitness if dataset is provided.

        :param root: The root node of the tree.
        :param X_dicts: List of dictionaries containing variable values for each sample.
        :param y: Array of actual output values.
        """
        self.root = root
        self.fitness = None
        self.num_nodes = 0  # Initialize node count

        self.compute_nodes()  # Update node count

        # Compute fitness if dataset is provided
        if X_dicts is not None and y is not None:
            self.compute_fitness(X_dicts, y)


    def evaluate(self, X_values):
        """
        Evaluates the expression tree for given input values.

        :param X_values: Dictionary mapping variable names to numerical values.
        :return: Computed value.
        """
        return self.root.evaluate(X_values)

    def to_formula(self):
        return self.root.to_formula() if self.root else "Empty Tree"

    def compute_nodes(self):
        """Updates the total number of nodes in the tree."""
        self.num_nodes = self.root.count_nodes() if self.root else 0

    # def compute_fitness(self, X_dicts, y):
    #     """Computes the fitness using Mean Squared Error, handling edge cases."""
    #     if self.root is None:  # Avoid computing fitness for empty trees
    #         self.fitness = float('inf')
    #         return self.fitness

    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore", category=RuntimeWarning)
    #         predictions = np.array([self.evaluate(X_dict) for X_dict in X_dicts])
    #         self.fitness = 100 * np.mean((predictions - y) ** 2)  # Mean Squared Error

    #     return self.fitness
    def compute_fitness(self, X_dicts, y):
        """Computes the fitness using Mean Squared Error (MSE) and handles numerical issues."""
        
        if self.root is None:  # Avoid computing fitness for empty trees
            self.fitness = float('inf')
            return self.fitness

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            try:
                predictions = np.array([self.evaluate(X_dict) for X_dict in X_dicts])

                # Handle invalid predictions (NaN or Inf)
                if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                    self.fitness = float('inf')  # Penalize invalid trees
                else:
                    # Use np.nan_to_num to ensure fitness does not become NaN/inf
                    mse = np.mean((predictions - y) ** 2)
                    self.fitness = np.nan_to_num(100 * mse, nan=1e6, posinf=1e6, neginf=1e6)  # Clip extreme values
                
            except Exception as e:
                self.fitness = float('inf')  # If evaluation crashes, set high fitness
                print(f"Warning: Fitness evaluation error: {e}")

        return self.fitness

    def update(self, X_dicts, y):
        """Ensures node count and fitness are updated correctly."""
        self.compute_nodes()  # Ensure node count is up-to-date
        self.compute_fitness(X_dicts, y)  # Ensure fitness is accurate

# =============== FUNCTIONS ===============
# Select a random node in the tree
def select_random_node(node, parent=None):
    if node is None or (node.is_leaf() and random.random() < 0.5):  # Select leaf nodes less frequently
        return node, parent

    if random.random() < 0.5 and node.left is not None:  # Randomly go left
        return select_random_node(node.left, node)
    elif node.right is not None:  # Otherwise, go right
        return select_random_node(node.right, node)

    return node, parent  # Return the node and its parent

# =============== GENERATION ===============
import random
import numpy as np

def generate_random_tree(variables, max_depth=3, p_const=0.2):
    """
    Recursively generates a valid symbolic regression tree.

    :param variables: List of variable names (e.g., ["x0", "x1", ...]).
    :param max_depth: Maximum depth of the generated tree.
    :param p_const: Probability of generating a constant at leaf nodes.
    :return: A valid expression tree (Node).
    """
    if max_depth == 0 or (random.random() < p_const):  
        # Generate a leaf node (either a variable or a constant)
        if random.random() < 0.5:
            return Node(random.choice(variables))  # Variable node
        return Node(np.random.uniform(-5, 5))  # Constant node in range [-5, 5]

    if random.random() < 0.5:  # Binary operation (must have two children)
        op = random.choice(BINARY_OP)
        
        left_child = generate_random_tree(variables, max_depth - 1, p_const)
        right_child = generate_random_tree(variables, max_depth - 1, p_const)
        
        # Ensure the right child of a division operation is never zero
        if op == np.divide:
            max_attempts = 10  # Avoid infinite loops
            attempts = 0
            while isinstance(right_child.value, (int, float)) and abs(right_child.value) < 1e-3:
                if attempts >= max_attempts:
                    right_child = Node(1.0)  # Fallback to a safe constant
                    break
                right_child = generate_random_tree(variables, max_depth - 1, p_const)
                attempts += 1

        return Node(op, left_child, right_child)

    # Unary operation (must have exactly one child)
    op = random.choice(UNARY_OP)
    return Node(op, generate_random_tree(variables, max_depth - 1, p_const), None)


def generate_population(population_size, variables, X_dicts, y, max_depth=3):
    return Parallel(n_jobs=4)(
        delayed(ExpressionTree)(generate_random_tree(variables, max_depth), X_dicts, y) 
        for _ in range(population_size)
    )

# =============== SELECTION ===============
# def standard_tournament_selection(population, tournament_size=5, elite_perc=0.5):
#     # perform normal tournament selection
#     tournament_pool = random.sample(population, tournament_size)
#     return min(tournament_pool, key=lambda tree: tree.fitness)

def tournament_selection(population, tournament_size=5, elite_perc=0.7, size_bias_prob=0.3):
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
def subtree_mutation(tree, variables, X_dicts, y):
    tree_copy = copy.deepcopy(tree)
    node_to_mutate, parent = select_random_node(tree_copy.root)

    if node_to_mutate is None:
        return tree_copy

    new_subtree = generate_random_tree(variables)

    if parent is None:
        tree_copy.root = new_subtree
    else:
        if parent.left == node_to_mutate:
            parent.left = new_subtree
        elif parent.right == node_to_mutate:
            parent.right = new_subtree

    tree_copy.update(X_dicts, y)
    return tree_copy


def point_mutation(tree, variables, X_dicts, y):
    tree_copy = copy.deepcopy(tree)
    node_to_mutate, _ = select_random_node(tree_copy.root)

    if node_to_mutate is None:
        return tree_copy  # No mutation happened

    # Apply mutation based on node type
    if node_to_mutate.is_binary():
        node_to_mutate.value = random.choice(BINARY_OP)  # Change to a different binary operator
    elif node_to_mutate.is_unary():
        node_to_mutate.value = random.choice(UNARY_OP)  # Change to a different unary operator
    elif node_to_mutate.is_leaf():
        if isinstance(node_to_mutate.value, str):  # Variable
            node_to_mutate.value = random.choice(variables)  # Change to another variable
        elif isinstance(node_to_mutate.value, (int, float)):  # Constant
            node_to_mutate.value += random.uniform(-1.0, 1.0)  # Slightly modify the constant

    # Compute fitness after mutation
    tree_copy.update(X_dicts, y)

    return tree_copy

def hoist_mutation(tree, X_dicts, y):
    tree_copy = copy.deepcopy(tree)
    
    # Select a random subtree (node and parent)
    subtree, parent = select_random_node(tree_copy.root)

    if subtree is None or subtree.is_leaf():  # Ensure mutation does something meaningful
        return tree_copy

    # Replace the entire tree with the selected subtree
    tree_copy.root = subtree

    # Compute fitness after mutation
    tree_copy.compute_fitness(X_dicts, y)

    return tree_copy


# ================== CROSSOVER ==================
def crossover(parent1, parent2, X_dicts, y):
    """
    Performs subtree crossover between two parents.
    
    :param parent1: First parent ExpressionTree.
    :param parent2: Second parent ExpressionTree.
    :return: One offspring ExpressionTree.
    """
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
def offspring_generation(population, variables, X_dicts, y, tournament_size=4, elite_perc=0.5, mutation_rate=0.3, crossover_rate=0.7):
    
    # Select first parent
    parent1 = tournament_selection(population, tournament_size)

    # Perform crossover (with probability)
    if random.random() < crossover_rate:
        parent2 = tournament_selection(population, tournament_size)
        offspring1 = crossover(parent1, parent2, X_dicts, y)  # Apply crossover
    else:
        offspring1 = copy.deepcopy(parent1)  # No crossover, just copy the parent

    # Perform mutation (with probability)
    if random.random() < mutation_rate:
        mutation_function = random.choice([subtree_mutation, point_mutation])
        offspring1 = mutation_function(offspring1, variables, X_dicts, y)


    return offspring1

# def adjust_rates(epoch, num_epochs, min_mutation=0.3, mutation_rate_start=0.7):
#     mutation_rate = max(mutation_rate_start * (1 - epoch / num_epochs), min_mutation)
#     crossover_rate = 1 - mutation_rate
#     return mutation_rate, crossover_rate

def run_evolution(num_epochs, population_size, variables, X_dicts, y, max_depth=4, 
                  tournament_size=4, elite_perc=0.5, mutation_rate=0.7, crossover_rate=0.3, n_jobs=4):
    
    # Track the best formula across all generations
    best_overall = None
    best_fitness = float('inf')

    # Step 1: Initialize Population
    population = generate_population(population_size, variables, X_dicts, y, max_depth)

    # Run the evolution
    for epoch in tqdm(range(num_epochs), desc="Evolving Population", unit="gen"):
        # Dynamic mutation & crossover adjustments: prioritize mutation early, shift to crossover later
        if epoch > 40:
            mutation_rate = 0.3
            crossover_rate = 0.7

        new_population = []

        # Generate offspring in parallel
        offspring_list = Parallel(n_jobs=n_jobs)(
            delayed(offspring_generation)(population, variables, X_dicts, y, tournament_size, elite_perc, mutation_rate, crossover_rate)
            for _ in range((population_size // 2))  # Generate half the population as offspring
        )

        # Ensure valid offspring and maintain population size
        new_population.extend([child for child in offspring_list if child is not None])
        
        # complete the population generating random trees to balance exploration and exploitation
        new_population.extend(generate_population(population_size - len(new_population), variables, X_dicts, y, max_depth))

        print(len(new_population))

        # Replace old population with new one
        population = new_population

        best_individual = min(population, key=lambda tree: tree.fitness)

        # Print progress
        tqdm.write(
                   f"\n-- Best Formula: {best_individual.root.to_formula()} "
                   f"\n-- Best Fitness: {best_individual.fitness:.4f} "
                   f"\n-- Mutation Rate: {mutation_rate:.2f}, Crossover Rate: {crossover_rate:.2f}")

        # Update best solution found so far
        if best_individual.fitness < best_fitness:
            best_fitness = best_individual.fitness
            best_overall = copy.deepcopy(best_individual)

    # Step 5: Return the best individual found
    return best_overall
