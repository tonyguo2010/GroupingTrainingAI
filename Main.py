from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from numpy import array
import math
import sub
import time

raw_data = np.loadtxt("datapoints.txt", delimiter=",")
points = list(zip(raw_data.T[0], raw_data.T[1]))

colors = []
for x in raw_data.T[2]:
    if x == 1:
        colors.append("r")
    else:
        colors.append("blue")
# Sorting the red and blue colours

np.random.seed(123456)
x_min, x_max = min(raw_data.T[0]), max(raw_data.T[0])
y_min, y_max = min(raw_data.T[1]), max(raw_data.T[1])

weights = np.array(np.random.rand(2, 1))
bias = np.random.rand(1)[0] + x_max
# Adding the x_max to the bias gives more accuracy, try without it, the lines will be messed up but will still work.

trained_boundary_lines = []

print(weights, bias) # Comparison 1

# 25 is fitful for training, you can try different times
for i in range(25):
    learning_output = sub.perceptronStep(
        points, # X, Y of a point
        raw_data.T[2], # The True/False Data
        weights, # Random weights
        bias # Random bias based on x_max
    )
    weights, bias = learning_output # The updated values
    trained_boundary_lines.append((-weights[0] / weights[1], -bias / weights[1]))

weights = list(weights[0]) + list(weights[1]) # Simplifying

print(weights, bias) # Comparison 2

plt.ion()

plt.scatter(
    x = raw_data.T[0],
    y = raw_data.T[1],
    vmin=-0.5,
    s=20,
    c=colors,
    edgecolors='black'
) # Scatter Plot

plt.xlim(-0.5,1.5) # Set a veiw range
plt.ylim(-0.5,1.5)

for y in trained_boundary_lines:
    x = np.linspace(start=-10, stop=10, num=5)
    plt.plot(x, y[0] * x + y[1], linestyle="--", color="c", alpha=0.3)
    # This shows how the line adjusted.
    plt.pause(0.1)

slop = -weights[0] / weights[1] # Calculating slope of the line
intercept = -bias / weights[1] # Calculating intercept of the line

x = np.linspace(
    start=-10, # How big the line'll be.
    stop=10,
    num=5 # Amount of dots that will be connected to generate
)
y = slop * x + intercept # Yeah same thing for above above

plt.pause(2)
plt.plot(x, y, "k") # Drawing the line, the last trained result

# The labels
plt.title("Randomized Dataset")
plt.xlabel('X', color='#1C2833')
plt.ylabel('Y', color='#1C2833')

plt.ioff()
plt.show()

