import numpy as np
import matplotlib.pyplot as plt

# There are 3 lines
num_of_lines = 3

# Defines a coluor for each line
colours = ['c', 'crimson', 'chartreuse']

# Defines a marker for each line
markers = ['o', 'v', '*']

# Creates x array with numbers ranged from 0 to 10(exclusive)
# Creates an empty list for y co-ordinates in each line
x = np.arange(10)
y = []

# For each line
for i in range(num_of_lines):
    # Adds to y according to y=ix+1 function
    y.append(x*i+1)

# This is where plotting happens!!!
# For each line
for i in range(num_of_lines):
    # Scatter plot with point_size^2 = 75, and with respective colors
    plt.scatter(x, y[i], marker=markers[i], s=75, c=colours[i])
    # Connects points with lines, and with respective colours
    plt.plot(x, y[i], c=colours[i])

# Show grid in the plot
plt.grid()
# Finally, display the plot
plt.show()