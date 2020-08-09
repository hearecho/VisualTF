"""
实战kaggle 房价预测
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from tensorflow import initializers as init

train_data = pd.read_csv('../data/train.csv')
test_data = pd.read_csv('../data/test.csv')

# print(train_data.shape)

# print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
#标准化数据,只处理是数据类型的字段进行标准化
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
# print(all_features.dtypes)
#标准化一般都是减去均值除以标准差
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
all_features = all_features.fillna(0)
#对于数据类型是 object的 他们中可能有很多都是集中离散值，那么就可以删除该列，并新增列用来表示不同的参数，使用0，1来进行表示
# dummy_na=True将缺失值也当作合法的特征值并为其创建指示特征
all_features = pd.get_dummies(all_features, dummy_na=True)
# print(all_features.shape)
# print(all_features)
#转换为np数组，方便训练
#训练数据有多少
n_train = train_data.shape[0]
train_features = np.array(all_features[:n_train].values,dtype=np.float)
test_features = np.array(all_features[n_train:].values,dtype=np.float)
train_labels = np.array(train_data.SalePrice.values.reshape(-1, 1),dtype=np.float)
print(train_labels)


def get_net():
    net = keras.models.Sequential()
    net.add(keras.layers.Dense(1))
    return net

log_rmse=tf.keras.losses.mean_squared_logarithmic_error

def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = tf.concat([X_train, X_part], axis=0)
            y_train = tf.concat([y_train, y_part], axis=0)
    return X_train, y_train, X_valid, y_valid

def k_fold(k, X_train, y_train, num_epochs,learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        # create model
        data = get_k_fold_data(k, i, X_train, y_train)
        net=get_net()
        # Compile model
        net.compile(loss=tf.keras.losses.mean_squared_logarithmic_error, optimizer=tf.keras.optimizers.Adam(learning_rate))
        # Fit the model
        history=net.fit(data[0], data[1],validation_data=(data[2], data[3]), epochs=num_epochs, batch_size=batch_size,validation_freq=1,verbose=0)
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        print('fold %d, train rmse %f, valid rmse %f'
              % (i, loss[-1], val_loss[-1]))
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='train')
    plt.plot(val_loss, label='valid')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
# k_fold(k, train_features, train_labels, num_epochs,lr, weight_decay, batch_size)






