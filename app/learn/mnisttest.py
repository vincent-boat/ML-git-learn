import tensorflow as tf
import numpy as np
from PIL  import Image

model_save_path = './checkpoint/mnist.ckpt'

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')])

model.load_weights(model_save_path)

preNum = int(input('input the number of test pictures:'))

for i in range(preNum):
    image_path =input('input the name of test pictures:')
    img = Image.open(image_path)
    img = img.resize((28,28),Image.ANTIALIAS)
    img_arr = np.array(img.convert('L'))

    for i in range(28):
        for j in range(28):
            if img_arr[i][j] < 50:
                img_arr[i][j] = 255
            else:
                img_arr[i][j] = 0

    img_arr = img_arr/255.0
    x_prdict = img_arr[tf.newaxis,...]
    result = model.predict(x_prdict)
    pred = tf.argmax(result, axis=1)
    print('\n')
    tf.print(pred)
