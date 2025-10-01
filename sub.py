from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from numpy import array
import math


def prediction(X, W, b):
    if (np.matmul(X, W) + b)[0] >= 0:
        return 1
    return 0

def perceptronStep(Points, colorIndex, Weight, bias, learn_rate=0.01):
    for i in range(len(Points)):
        y_hat = prediction(Points[i], Weight, bias)
        # A red point in the blue area
        if colorIndex[i] - y_hat == 1:
            Weight[0] += Points[i][0] * learn_rate
            Weight[1] += Points[i][1] * learn_rate
            bias += learn_rate
            # Move line closer to the misclassified red point

        # A blue point in the red area
        elif colorIndex[i] - y_hat == -1:
            Weight[0] -= Points[i][0] * learn_rate
            Weight[1] -= Points[i][1] * learn_rate
            bias -= learn_rate
            # Move line closer to the misclassified blue point

    return Weight, bias
