import tensorflow as tf
import numpy as np

w = tf.Variable(tf.constant(5,dtype=tf.float32))
lr = 0.6
epoch = 40

#for epoch in range(epoch):
#    with tf.GradientTape()as tape:
#        loss = tf.square(w+1)
#    grads = tape.gradient(loss, w)
#
#    w.assign_sub(lr*grads)
#    print("After %s epoch,w is %f,loss is %f" %(epoch, w.numpy(),loss))

a = np.arange(15)
b = a[5:]
c = a[:-5]
print(b)
print(c)