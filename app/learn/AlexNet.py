import tensorflow as tf
import tensorflow.keras.datasets.cifar10 as cifar
import numpy as np
from matplotlib import pyplot as plt
import os
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D,BatchNormalization,Activation,MaxPool2D,Dropout,Flatten,Dense
np.set_printoptions(threshold=np.inf)

(x_train,y_train),(x_test,y_test) = cifar.load_data()
x_train,x_test = x_train/255.0,x_test/255.0
#LeNet
class AlexLine(Model):
    def __init__(self):
        super(AlexLine,self).__init__()
        self.c1 = Conv2D(filters=96, kernel_size=(3,3),padding='valid')#卷积层
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')#激活函数
        self.p1 = MaxPool2D(pool_size=(3,3),strides=2,padding='valid')#池化层

        self.c2 = Conv2D(filters=256,kernel_size=(3,3),padding='valid')#第二层卷积层
        self.b2 = BatchNormalization()
        self.a2 = Activation('relu')
        self.p2 = MaxPool2D(pool_size=(3,3),strides=2,padding='valid')

        self.c3 = Conv2D(filters=384,kernel_size=(3,3),padding='same',activation='relu')

        self.c4 = Conv2D(filters=384,kernel_size=(3,3),padding='same',activation='relu')

        self.c5 = Conv2D(filters=256,kernel_size=(3,3),padding='same',activation='relu')
        self.p3 = MaxPool2D(pool_size=(3,3),strides=2)

        self.flatten = Flatten()#将输入特征拉成直线
        self.f1 = Dense(2048,activation='relu')#第一层神经网络
        self.d1 = Dropout(0.5)
        self.f2 = Dense(2048,activation='relu')#第二层神经网络
        self.d2 = Dropout(0.2)
        self.f3 = Dense(10,activation='softmax')


    def call(self,x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)

        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.p2(x)

        x = self.c3(x)

        x = self.c4(x)

        x = self.c5(x)
        x = self.p3(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d1(x)
        x = self.f2(x)
        x = self.d2(x)
        y = self.f3(x)
        return y
model = AlexLine()

model.compile(
    optimizer = "adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['sparse_categorical_accuracy']
)
checkpoint_save_path = "./checkpoint/AlexLine.ckpt"
if os.path.exists(checkpoint_save_path + 'index'):
    print('----------load model-----------')
    model.load_weights(checkpoint_save_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)
history = model.fit(x_train, y_train, batch_size=32, epochs =5, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])
model.summary()
file = open('./weights.txt','w')
for v in model.trainable_variables:
    file.write(str(v.name)+'\n')
    file.write(str(v.shape)+'\n')
    file.write(str(v.numpy)+'\n')
file.close()

#************show******************

acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.subplot(1,2,1)
plt.plot(acc , label= 'Training Accuracy')
plt.plot(val_acc, label = 'Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(loss,label = 'Training Loss')
plt.plot(val_loss,label = 'Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()


