#神经网络搭建流程 使用keras实现鸢尾花分类
#1、import所有需要的包
import tensorflow as tf
from sklearn import datasets
from tensorflow.keras.layers import Dense
from  tensorflow.keras import Model
import numpy as np

#2、说明训练集和测试集
x_train = datasets.load_iris().data
y_train = datasets.load_iris().target

np.random.seed(116)
np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(y_train)
tf.random.set_seed(116)
#class model
class IrisModel(Model):
    def __init__(self):
        super(IrisModel,self).__init__()
        self.d1 = Dense(3,activation='sigmoid',kernel_regularizer=tf.keras.regularizers.l2())

    def call(self,x):
        y = self.d1(x)
        return y
model = IrisModel()


#3、用model = tf.keras.models.Sequential 描述前向传播流程
#model = tf.keras.models.Sequential([
#    tf.keras.layers.Dense(3,activation='softmax',kernel_regularizer=tf.keras.regularizers.l2())
#])

#4、用model.compile说明训练方法：优化器、激活函数等
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

#5、用model.fit说明训练迭代多少次、学习率、batch大小等。
model.fit(x_train,y_train,batch_size=32,epochs=500,validation_split=0.2,validation_freq=20)

#6、用model.summary打印出网络结构和参数统计
model.summary()
