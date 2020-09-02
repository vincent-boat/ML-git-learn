import tensorflow as tf
from  tensorflow.keras.layers import Flatten,Dense
from  tensorflow.keras import Model
from  tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt


fashion = tf.keras.datasets.fashion_mnist
(x_train,y_train) , (x_test,y_test) = fashion.load_data()
x_train, x_test = x_train/255.0, x_test/255.0
#图像预处理增强数据集
x_train = x_train.reshape(x_train.shape[0],28,28,1)#增加一个维度使得数据和网络结构匹配

image_gen_train = ImageDataGenerator(
    rescale=1 ,
    rotation_range=15,
    width_shift_range=.1,
   height_shift_range=.1,
    horizontal_flip=True,
    zoom_range=0.2
)
image_gen_train.fit(x_train)

class FashionModel(Model):
    def __init__(self):
        super(FashionModel, self).__init__()
        self.flatten = Flatten()
        self.d1 = Dense(255,activation='relu')
        self.d2 = Dense(10,activation='softmax')
    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        y = self.d2(x)
        return y
model = FashionModel()

model.compile(optimizer = 'adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy']
)
history = model.fit(image_gen_train.flow( x_train,y_train,batch_size=32),epochs=5,validation_data=(x_test,y_test),validation_freq=5)
model.summary()

acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1,2,1)
plt.plot(acc, label = 'Training Accuracy')
plt.plot(val_acc , label = 'Validation Accuracy')
plt.title('Training Validation Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(loss, label = 'Training Loss')
plt.plot(val_loss, label = 'Validation Loss')
plt.title('Training Validation Loss')
plt.legend()

plt.show()