import tensorflow as tf
import numpy as np

"""
Usage:
    python tensors_example.py
"""

m1 = [[1.0, 2.0],
      [3.0, 4.0]]

m2 = np.array([[1.0, 2.0],
               [3.0, 4.0]], dtype=np.float32)

m3 = tf.constant([[1.0, 2.0],
                  [3.0, 4.0]])

print(type(m1))
print(type(m2))
print(type(m3))


print("----------")
t1 = tf.convert_to_tensor(m1, dtype=tf.float32)
t2 = tf.convert_to_tensor(m2, dtype=tf.float32)
t3 = tf.convert_to_tensor(m3, dtype=tf.float32)

print(type(t1))
print(type(t2))
print(type(t3))



print("----------")
matrix1 = tf.constant([[1.0, 2.0]])
matrix2 = tf.constant([[1], [2]])

myTensor = tf.constant([
    [[1,2 ],
     [3,4],
     [5,6]],
    [[7,8],
     [9,10],
     [11,12]]
])

matrix_one = tf.ones((3,4))
matrix_zeros = tf.zeros((3,4))
matrix_ten = tf.ones((3, 4)) * 10

print(matrix1)
print(matrix2)
print(myTensor)
print(matrix_one)
print(matrix_zeros)
print(matrix_ten)


print("----------")
x = tf.constant([[1, 2]])
neg_x = tf.negative(x)
print(neg_x)

print("---- Session ---- ")
with tf.Session() as sess:
    result = sess.run(neg_x)
print(result)

sess = tf.InteractiveSession()

matrix = tf.constant([[1., 2.]])
negMatrix = tf.negative(matrix)

result = negMatrix.eval()

print(result)
sess.close()