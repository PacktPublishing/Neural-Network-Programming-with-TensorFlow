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
label_data = []
base = "/home/ubuntu/work/github/rajdeepd/neuralnetwork-programming/ch02/data/train-100-reduced"
base_path_1ist = [base + "/Type_1/", base + "/Type_2/", base + "/Type_3/"]



for base_path in base_path_1ist:
    print(base_path)
    label_type1 = np.asarray([1.,0.,0.])
    label_type2 = np.asarray([0., 1., 0.])
    label_type3 = np.asarray([0., 0., 1.])
    label = np.asarray([])
    if base_path.endswith("1/"):
        label = label_type1
    elif base_path.endswith("2/"):
        label = label_type2
    else:
        label = label_type3

    list_files = os.listdir(base_path)

    for l in list_files:
        p = base_path + l
        try:
            arr = load_image(p)
            train_data.append(arr)
            label_data.append(label)
            print(l)

        except IOError as err:
            print("IO error: {0}".format(err))

    print("train_data len:" + str(train_data.__len__()))
    print("label_data len:" + str(label_data.__len__()))
    np.savetxt("labels-100.csv", label_data, delimiter=",", fmt='%1.1f')

train_data_ndarray = np.asarray(train_data)
label_data_ndarray = np.asarray(label_data)

print("Total Size:" + str(train_data.__len__()))
