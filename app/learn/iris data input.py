from sklearn import datasets
from pandas import DataFrame
import pandas as pd
x_data = datasets.load_iris().data#获取鸢尾花测量数据集
y_data = datasets.load_iris().target#获取鸢尾花对应标签
print("x_data from datasets: \n",x_data)#打印测量数据
print("y_data from datasets: \n",y_data)#打印标签

x_data = DataFrame(x_data,columns=['花萼长度','花萼宽度','花瓣长度','花瓣宽度'])#整理数据格式成为表格形式
pd.set_option('display.unicode.east_asian_width',True)#设置列名对齐
print('x_data add index:\n',x_data)#打印整理厚的数据表

x_data['     类别'] = y_data#添加类别栏数据为y_data
print('x_data add column: \n',x_data)#打印添加对应标签列


