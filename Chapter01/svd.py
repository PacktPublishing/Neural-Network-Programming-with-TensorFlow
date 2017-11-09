import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plts

path = "/Users/manpreet.singh/Sandbox/codehub/github/science/neuralnetwork-programming/ch01/plots"

text = ["I", "like", "enjoy",
         "deep", "learning", "NLP", "flying", "."]
xMatrix = np.array([[0,2,1,0,0,0,0,0],
              [2,0,0,1,0,1,0,0],
              [1,0,0,0,0,0,1,0],
              [0,1,0,0,1,0,0,0],
              [0,0,0,1,0,0,0,1],
              [0,1,0,0,0,0,0,1],
              [0,0,1,0,0,0,0,1],
              [0,0,0,0,1,1,1,0]], dtype=np.float32)

X_tensor = tf.convert_to_tensor(xMatrix, dtype=tf.float32)

# tensorflow svd
with tf.Session() as sess:
    s, U, Vh = sess.run(tf.svd(X_tensor, full_matrices=False))

for i in range(len(text)):
    plts.text(U[i,0], U[i,1], text[i])

plts.ylim(-0.8,0.8)
plts.xlim(-0.8,2.0)
plts.savefig(path + '/svd_tf.png')

# numpy svd
la = np.linalg
U, s, Vh = la.svd(xMatrix, full_matrices=False)

print(U)
print(s)
print(Vh)

# write matrices to file (understand concepts)
file = open(path + "/matx.txt", 'w')
file.write(str(U))
file.write("\n")
file.write("=============")
file.write("\n")
file.write(str(s))
file.close()

for i in range(len(text)):
    plts.text(U[i,0], U[i,1], text[i])

plts.ylim(-0.8,0.8)
plts.xlim(-0.8,2.0)
plts.savefig(path + '/svd_np.png')
