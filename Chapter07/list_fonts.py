import matplotlib.pyplot as plt
import warnings

for font in plt.rcParams['font.sans-serif']:
    print font
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        plt.rcParams['font.family'] = font
        plt.text(0,0,font)
        plt.savefig(font+'.png')

        if len(w):
            print "Font {} not found".format(font)