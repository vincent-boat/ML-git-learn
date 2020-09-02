import numpy as np
import tensorflow as tf
from  tensorflow.keras.layers import  Dense,SimpleRNN
import matplotlib.pyplot as plt
import os

input_word = 'abcde'
w_to_id = { 'a':0 , 'b':1, 'c':2, 'd':3, 'e':4}#单词映射到数值id的词典
id_to_onthot = {0 : [1.,0.,0.,0.,0.], 1: [0.,1.,0.,0.,0.], 2: [0.,0.,1.,0.,0.], 3: [0.,0.,0.,1.,0.,], 4: [0.,0.,0.,0.,1.]}

x_train = [id_to_onthot[w_to_id['a']],id_to_onthot[w_to_id['b']],id_to_onthot[w_to_id['c']],id_to_onthot[w_to_id['d']],id_to_onthot[w_to_id['e']]]
y_train = [w_to_id['b'],w_to_id['c'],w_to_id['d'],w_to_id['e'],w_to_id['a']]

np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)

x_train = np.reshape(x_train,(len(x_train),1,5))
y_train = np.array(y_train)

model = tf.keras.Sequential([
        SimpleRNN(3),
        Dense(5,activation='softmax')
])

model.compile(
                optimizer = tf.optimizers.Adam(0.01),
                loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics = ['sparse_categorical_accuracy']
)

checkpoint_save_path = './checkpoint/rnn_onehot_1prel.ckpt'

if os.path.exists(checkpoint_save_path + '.index'):
    print('-----------load model------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
                                                    filepath=checkpoint_save_path,
                                                    save_weights_only=True,
                                                    save_best_only=True,
                                                    monitor='loss'
)

history = model.fit(x_train,y_train,batch_size=32,epochs=100,callbacks=[cp_callback])

model.summary()

file = open('./weights.txt','w')
for v in model.trainable_variables:
    file.write(str(v.name)+'\n')
    file.write(str(v.shape)+'\n')
    file.write(str(v.numpy())+'\n')
file.close()

#####################show##########################

acc = history.history['sparse_categorical_accuracy']
loss = history.history['loss']

plt.subplot(1,2,1)
plt.plot(acc , label = 'Training Accuarcy')
plt.title('Training Accuarcy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(loss, label = 'Training Loss')
plt.title('Training loss')
plt.legend()
plt.show()

############predict##########
preNum = int(input('input the number of the test alphabet'))
for i in range(preNum):
    alphabet1 = input('input the test alphabet:')
    alphabet = [id_to_onthot[w_to_id[alphabet1]]]
    #为了使alphabet符合输入Simple RNN的要求进行reshape
    alphabet = np.reshape(alphabet,(1,1,5))
    result = model.predict([alphabet])
    pred = tf.argmax(result, axis = 1)
    pred = int(pred)
    tf.print(alphabet1 + '->' + input_word[pred])

