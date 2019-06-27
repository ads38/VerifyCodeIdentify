import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
'''
m1 = tf.constant(222)
m2 = tf.constant(4)
pro = tf.multiply(m1, m2)
sess = tf.Session()
res = sess.run(pro)
print(res)
'''

arr = []
for i in range(10):
    arr.append(i)
print(arr[5])