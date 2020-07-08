"""
汽车耗油预测实战
"""
import tensorflow as tf
import pandas as pd
from tensorflow import keras
import matplotlib
from tensorflow.keras import layers, losses

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt



dataset_path = keras.utils.get_file('auto-mpg.data',"http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
col =  ['MPG','Cylinders','Displacement','Horsepower','Weight','Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path,names=col,na_values="?",
                          comment='\t',sep=" ",skipinitialspace=True)

dataset = raw_dataset.copy()
#统计数据的特征
blank_num = dataset.isna().sum()
# print(blank_num)
#delete balnk data
dataset = dataset.dropna()
blank_num = dataset.isna().sum()
# print(blank_num)

#progress data 预处理数据
origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japen'] = (origin == 3)*1.0
print(dataset.tail())

#切分训练集 数据集
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_labels = train_dataset.pop("MPG")
test_labels = test_dataset.pop("MPG")

#通过标准差和均值 对数据进行规范化
train_stats = train_dataset.describe()
#转置
train_stats = train_stats.transpose()
"""
print(train_stats)
              count         mean         std  ...     50%      75%     max
Cylinders     314.0     5.477707    1.699788  ...     4.0     8.00     8.0
Displacement  314.0   195.318471  104.331589  ...   151.0   265.75   455.0
Horsepower    314.0   104.869427   38.096214  ...    94.5   128.00   225.0
Weight        314.0  2990.251592  843.898596  ...  2822.5  3608.00  5140.0
Acceleration  314.0    15.559236    2.789230  ...    15.5    17.20    24.8
Model Year    314.0    75.898089    3.675642  ...    76.0    79.00    82.0
USA           314.0     0.624204    0.485101  ...     1.0     1.00     1.0
Europe        314.0     0.178344    0.383413  ...     0.0     0.00     1.0
Japen         314.0     0.197452    0.398712  ...     0.0     0.00     1.0
"""
def formater(x):
    return (x-train_stats['mean'])/train_stats['std']

#标准化
format_train_data = formater(train_dataset)
format_test_data = formater(test_dataset)

"""
314行 输入特征长度为9
print(format_train_data.shape,train_labels.shape)
(314, 9) (314,)
"""
#构建dateset对象，并随机打乱之后分批次
train_db = tf.data.Dataset.from_tensor_slices((format_train_data.values,train_labels.values)).shuffle(100).batch(32)

#查看属性对于 MPG的影响
"""
fig,(ax1,ax2,ax3) = plt.subplots(1,3,sharey=True)
ax1.scatter(train_dataset["Cylinders"],train_labels)
ax2.scatter(train_dataset["Displacement"],train_labels)
ax3.scatter(train_dataset["Model Year"],train_labels)
plt.show()
"""

#定义网络
class Network(keras.Model):
    def __init__(self):
        super(Network, self).__init__()
        self.lay1 = layers.Dense(64,activation='relu')
        self.lay2 = layers.Dense(64,activation='relu')
        self.lay3 = layers.Dense(1)

    def call(self,inputs,training=None,mask=None):
        x = self.lay1(inputs)
        x = self.lay2(x)
        x = self.lay3(x)

        return x

model = Network()
#9 为特征值长度
model.build(input_shape=(4,9))
print(model.summary())
optimizer = tf.keras.optimizers.RMSprop(0.001)
for epoch in range(200):
    #遍历一次训练集
    for step,(x,y) in enumerate(train_db):
        with tf.GradientTape() as tape:
            out = model(x)
            #计算MSE均方差
            loss = tf.reduce_mean(losses.MSE(y,out))
            mae_loss = tf.reduce_mean(losses.MAE(y,out))
        if step % 10 == 0:
            print("eopch:",epoch,"step:",step,"loss:",float(mae_loss))
        grads = tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(grads,model.trainable_variables))





