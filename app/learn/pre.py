from sklearn import datasets
import tensorflow as tf
import numpy as np

a = tf.constant([1,2,3,1,1])
b = tf.constant([0,1,3,4,5])
#where(（条件语句）,a,b)   如果为真则返回a 如果为假则返回b
c = tf.where(tf.greater(a,b),a,b)#greater 函数对应元素比较是否a>b
#print('c:',c)


#np.random.RandomState.rand() 可以返回一个[0,1)之间的随机数

rdm= np.random.RandomState(seed=11212)
a = rdm.rand()
b = rdm.rand(2,3)
#print(a)
#print(b)

#np.vstack可以将两个数组按垂直方向叠加
a = np.array([1,2,3])
b = np.array([4,5,6])
c = np.vstack((a,b))
#print('c:\n',c)


#np.mgrid[起始值：结束值：步长，起始值：结束值：步长]  .ravel(数组)将其中的数组拉伸成为一维数组    np.c_[数组1，数组2.。]将数组配对
# 经常一起使用生成网格坐标点  a的每个值都对应b的每个值
a,b = np.mgrid[1:4:1,0:2:0.5]
grid =np.c_[a.ravel(),b.ravel()]
#print(grid)

#激活函数

