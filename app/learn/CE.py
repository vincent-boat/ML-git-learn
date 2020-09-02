import tensorflow as tf


loss_c1 = tf.losses.categorical_crossentropy([1,0],[0.8,0.2])
loss_c2 = tf.losses.categorical_crossentropy([1,0],[0.4,0.6])
print(loss_c1)
print(loss_c2)

#一般在使用时先用softmax将参数转换成符合概率分布的形式，再进行交叉熵计算
#tf.softmax_cross_entropy_with_logits(y_,y)
