import tensorflow as tf
import tensorflow.keras.datasets.cifar10 as cifar
#卷积计算过程

#CBAPD

#卷积在机器学习中的作用就是特征提取器
#搭建卷积神经网络的五步
#1、卷积
#Conv2D(filter=6,kernal_size = (5,5),padding = 'same'),
#2、批标准化
#BatchNormalization(),
#3、激活层
#Activation('relu'),
#4、池化层
#MaxPool2D(pool_size=(2,2),strides=2,padding='same')
#AveragePooling2D(pool_size=(2,2),strides=2.padding='same')
#5、舍弃层
#Dropout(0.2)