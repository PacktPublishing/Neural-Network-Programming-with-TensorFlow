import numpy
a = numpy.asarray([ [1,2,3], [4,5,6], [7,8,9] ])
#numpy.savetxt("foo.csv", a, delimiter=",", fmt='%1.1f')

x = numpy.loadtxt("labels-100.csv", delimiter=',')
label_array = []
for y in x:
    y_ndarray = numpy.asarray(y)
    label_array.append(y_ndarray)
print(label_array)
#print(x)
