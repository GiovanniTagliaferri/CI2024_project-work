from functions import *

def f1(depth = 3, pop_size = 50, num_epochs = 50, tournament_size = 5, problem_id = 1):
    best_formula = run_evolution(num_epochs, pop_size, problem_id, depth, tournament_size)

    print(f"Best formula found: {best_formula}"
          f"\nFitness: {best_formula.fitness:4e}")

    return best_formula

def f2(depth = 6, pop_size = 100, num_epochs = 100, tournament_size = 5, problem_id = 2):
    best_formula = run_evolution(num_epochs, pop_size, problem_id, depth, tournament_size)

    print(f"Best formula found: {best_formula}"
          f"\nFitness: {best_formula.fitness:4e}")

    return best_formula

def f3(depth = 3, pop_size = 100, num_epochs = 100, tournament_size = 5, problem_id = 3):
    best_formula = run_evolution(num_epochs, pop_size, problem_id, depth, tournament_size)

    print(f"Best formula found: {best_formula}"
          f"\nFitness: {best_formula.fitness:4e}")

    return best_formula

def f4(depth = 3, pop_size = 100, num_epochs = 100, tournament_size = 5, problem_id = 4):
    best_formula = run_evolution(num_epochs, pop_size, problem_id, depth, tournament_size)

    print(f"Best formula found: {best_formula}"
          f"\nFitness: {best_formula.fitness:4e}")

    return best_formula

def f5(depth = 6, pop_size = 100, num_epochs = 100, tournament_size = 5, problem_id = 5):
    best_formula = run_evolution(num_epochs, pop_size, problem_id, depth, tournament_size)

    print(f"Best formula found: {best_formula}"
          f"\nFitness: {best_formula.fitness:4e}")

    return best_formula

def f6(depth = 6, pop_size = 100, num_epochs = 100, tournament_size = 5, problem_id = 6):
    best_formula = run_evolution(num_epochs, pop_size, problem_id, depth, tournament_size)

    print(f"Best formula found: {best_formula}"
          f"\nFitness: {best_formula.fitness:4e}")

    return best_formula

def f7(depth = 6, pop_size = 100, num_epochs = 100, tournament_size = 5, problem_id = 7):
    best_formula = run_evolution(num_epochs, pop_size, problem_id, depth, tournament_size)

    print(f"Best formula found: {best_formula}"
          f"\nFitness: {best_formula.fitness:4e}")

    return best_formula

def f8(depth = 6, pop_size = 100, num_epochs = 50, tournament_size = 5, problem_id = 8):
    best_formula = run_evolution(num_epochs, pop_size, problem_id, depth, tournament_size)

    print(f"Best formula found: {best_formula}"
          f"\nFitness: {best_formula.fitness:4e}")

    return best_formula

