import Image
import numpy as np
import os
import scipy
from scipy import misc
from scipy.misc import imsave
def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data
train_data = []
base = "/home/ubuntu/work/github/rajdeepd/neuralnetwork-programming/ch02/data/"
base_path = base + "/test-100/Type_3/"
base_red_path = base + "/test-100-reduced/Type_3/"



list_files = os.listdir(base_path)
for l in list_files:
    p = base_path + l
    try:
        arr = load_image(p)
        arr_resized = misc.imresize(arr, 10)
        print(l)
        imsave(base_red_path + l, arr_resized)
    except IOError as err:
        print("IO error: {0}".format(err))
