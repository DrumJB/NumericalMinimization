import numpy as np

def RastriginFunction(x, y):
    # x, y ∈ [-1, 1]
    # global minimum at (0, 0) with value 0

    # scale the input according to the search domain x, y ∈ [-5.12, 5.12]
    x = x * 5.12
    y = y * 5.12

    A = 10
    return A * 2 + (x ** 2 - A * np.cos(2 * np.pi * x)) + (y ** 2 - A * np.cos(2 * np.pi * y))

def HimmelblauFunction(x, y):
    # x, y ∈ [-1, 1]
    # global minima at (3, 2), (-2.805118, 3.131312), (-3.779310, -3.283186), and (3.584428, -1.848126) with value 0

    # scale the input according to the search domain x, y ∈ [-5, 5]
    x = x * 5
    y = y * 5

    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

def EggholderFunction(x, y):
    # x, y ∈ [-1, 1]
    # global minimum at (512, 404.2319) with value about -959.6407

    # scale the input according to the search domain x, y ∈ [-512, 512]
    x = x * 512
    y = y * 512
    
    return -(y + 47) * np.sin(np.sqrt(np.abs(0.5 * x + y + 47))) - x * np.sin(np.sqrt(np.abs(x - y - 47)))