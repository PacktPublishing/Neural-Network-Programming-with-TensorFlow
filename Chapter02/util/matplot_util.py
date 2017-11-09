import numpy as np
import matplotlib.pyplot as plt


def draw_plot(x,y, title, label):

    colours = ['blue', 'crimson', 'orange', 'red', 'black', 'green']
    i = 0
    # Defines a marker for each line
    markers = ['o', 'v', '*', '+', 'o', "x"]

    num_of_lines = len(y)

    for i in range(num_of_lines):
    # Scatter plot with point_size^2 = 75, and with respective colors
      plt.scatter(x, y[i], marker=markers[i], s=25, c=colours[i])
    # Connects points with lines, and with respective colours
      line1, = plt.plot(x, y[i], c=colours[i], label = label[i])
      import matplotlib.patches as mpatches
      from matplotlib.legend_handler import HandlerLine2D
      plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)}, loc =4)
      plt.title(title,y=1.05)


    plt.grid()
    plt.show()

def main():
    x = np.arange(10)
    y = []
    num_of_lines = 2
    # For each line
    for i in range(num_of_lines):
        # Adds to y according to y=ix+1 function
        y.append(x * i + 1)
    title = 'Test Plot'
    label = ['Test Accuracy', 'Train Accuracy']
    draw_plot(x,y,title, label)

if __name__ == '__main__':
    main()