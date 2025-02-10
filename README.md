# Project Work
The focus of this project was to solve the problem of **Symbolic Regression**.

The formulas found for each given problem are contained in the file `s324286.py`, and they can be run using the `symreg.py` script.  
This GitHub repository is structured as follows:

- `/data` contains the given problems in the format *problem_N.npz*.
- `/results` contains the final results:
  - `/predictions` contains the plots comparing the true output values with the predicted ones from the formulas.
  - `results.md` includes the final formulas for each problem, along with the best parameter and fitness values.
- `/src` contains the algorithm's code:
  - `functions.py` includes all the functions used to develop the algorithm.
  - `main.py` allows running the evolution process for a problem, specifying its ID and initial parameters.

Further details can be found in the file `CI2024_final-report_s324286.pdf`.
