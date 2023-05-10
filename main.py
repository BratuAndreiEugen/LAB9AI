import numpy as np

from solvers.DigitSolver import digits
from solvers.IRISSolver import iris
from solvers.MNISTSolver import mnist
from solvers.SepiaSolver import sepia
from solvers.ToolSepiaSolver import tool_sepiaCNN, tool_sepiaANN
from solvers.ToolSepiaSolverV2 import tool_sepia
from utils.data import get_mnist
from sklearn.datasets import load_digits

c = True
while c:
    print("\n")
    print("1. Sklearn Digits")
    print("2. MNIST Digits")
    print("3. IRIS")
    print("4. Sepia CNN")
    print("5. Sepia ANN")
    print("6. Sepia CNN that works :)")

    inp = int(input("Alege : "))
    if inp == 1:
        digits()
    elif inp == 2:
        mnist()
    elif inp == 3:
        iris()
    elif inp == 4:
        tool_sepiaCNN()
    elif inp == 5:
        tool_sepiaANN()
    elif inp == 6:
        tool_sepia()
    else:
        c = False
