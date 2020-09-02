from sklearn import datasets
import tensorflow as tf
import numpy as np
from   matplotlib import  pyplot as plt

#数据读入
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

#数据乱序
np.random.seed(116) #后面记得使用相同的seed值这样打乱后仍可以对齐
np.random.shuffle(x_data)
np.random.seed(116)#使用相同的seed值
np.random.shuffle(y_data)
tf.random.set_seed(116)

#数据集拆分
#训练集
x_train = x_data[:-30]
y_train = y_data[:-30]
#测试集
x_test = x_data[-30:]
y_test = y_data[-30:]
#数据类型转换
x_train = tf.cast(x_train,tf.float32)
x_test = tf.cast(x_test,tf.float32)

#配对打包
train_db = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(32)

#定义神经网络所有可训练参数
w1 = tf.Variable(tf.random.truncated_normal([4,3],stddev=0.1,seed=1))#一层神经网络，期中4表示输入特征类数，3表示输出分类数
b1 = tf.Variable(tf.random.truncated_normal([3],stddev=0.1,seed=1))#b1的维度必须与w1输出的分类维度相同

epoch = 300 #迭代次数epoch
lr = 0.25 #学习率
train_loss_result = [] #记录每轮loss值
test_acc = [] #记录每轮acc
loss_all = 0  #每轮4个step，loss_all记录每轮的loss和

#嵌套循环迭代，用with结构更新参数，显示当前loss

for epoch in range(epoch):
    for step ,(x_train,y_train) in enumerate(train_db): #batch 级迭代
        with tf.GradientTape() as tape:
            y = tf.matmul(x_train,w1)+b1
            y = tf.nn.softmax(y)
            y_ = tf.one_hot(y_train,depth=3)
            loss = tf.reduce_mean(tf.square(y_ - y))
            loss_all += loss.numpy()
        grads = tape.gradient(loss,[w1,b1])
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])

    print('Epoch {} , loss:{}'.format(epoch,loss_all/4))
    train_loss_result.append(loss_all/4)
    loss_all = 0

    #测试部分
    total_correct , total_number = 0,0
    for x_test ,y_test in test_db:
        y = tf.matmul(x_test,w1) + b1
        y = tf.nn.softmax(y)
        pred = tf.argmax(y,axis=1)
        pred = tf.cast(pred , dtype=y_test.dtype)
        correct = tf.cast(tf.equal(pred , y_test) , dtype=tf.int32)
        correct = tf.reduce_sum(correct)
        total_correct += int(correct)
        total_number += x_test.shape[0]

    acc = total_correct / total_number
    test_acc.append(acc)
    print('test acc:',acc)
    print('——————————————————')

    #绘制loss 和 acc曲线
plt.title('loss function curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(train_loss_result, label= '$Loss$')
plt.legend()
plt.show()

plt.title('Acc curve')
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.plot(test_acc,label = '$Accuracy$')
plt.legend()
plt.show()




