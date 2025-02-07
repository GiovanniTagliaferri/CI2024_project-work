from s324286 import *

def symreg(problem_id):
    data = np.load(f"data/problem_{problem_id}.npz")
    X = data['x']
    y = data['y']
    
    if problem_id == 1:
        f = f1
    elif problem_id == 2:
        f = f2
    elif problem_id == 3:
        f = f3
    elif problem_id == 4:
        f = f4
    elif problem_id == 5:
        f = f5
    elif problem_id == 6:
        f = f6
    elif problem_id == 7:
        f = f7
    elif problem_id == 8:
        f = f8
    else:
        raise ValueError("Invalid problem number.")
    
    print(f"{100 * np.mean((f(x=X) - y) ** 2):4e}")


for i in range(1, 9):
    symreg(i)


