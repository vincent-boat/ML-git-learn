import  tensorflow as tf
#features = tf.constant([1, 2, 23, 37])
#labels = tf.constant([1, 0, 0, 1])
#dataset = tf.data.Dataset.from_tensor_slices((features,labels))
#print(dataset)
#for element in dataset:
#    print(element)
#不能显示id值 未解决

#枚举需要用两个变量去接收
#seq = ['one', 'two', 'three', 'four']
#for i,element in enumerate(seq):
#    print(i,element)

y = tf.constant([1.07, 2.33, -2])
y_pro = tf.nn.softmax(y)
print(y_pro)