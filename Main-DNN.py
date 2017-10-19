### Import the data ###
import numpy as np  # 用于数学计算，比如矩阵，向量计算
import pandas as pd  # 用于读取数据，做预处理
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# 合并所有的文件
df = pd.read_csv("T-1.csv")
for i in range(1, 22):
    # print ("T-" + str(i+1) + ".csv")
    tmp = pd.read_csv("T-" + str(i + 1) + ".csv")
    df = pd.concat([df, tmp])

# df = df.drop('Unnamed: 12', 1)

# 构造特征数据和 label
cols = [col for col in df.columns if col not in ['Output']]
X = df[cols]
df.iloc[:, 11:12]=df.iloc[:, 11:12]
#print(df)
y = df.iloc[:, 11:12]/10000
#print(y)

# separate training and testing data
lines_per_file = 105000
X_train = X[0:lines_per_file * 18]
y_train = y[0:lines_per_file * 18]
X_test = X[lines_per_file * 18: lines_per_file * 22]
y_test = y[lines_per_file * 18: lines_per_file * 22]

## convert to numpy format
X_train = X_train.as_matrix()
y_train = y_train.as_matrix()
X_test = X_test.as_matrix()
y_test = y_test.as_matrix()

# normalize the data
# 做归一化 (feature normalization)
m = np.mean(X_train, axis=0)
s = np.std(X_train, axis=0)

X_train = (X_train - m) / s
X_test = (X_test - m) / s

# 输出训练数据的大小
print (np.shape(X_train))
print (np.shape(y_train))
print (np.shape(X_test))
print (np.shape(y_test))

# copy the original documents
for i in range(0, 22):
    fname = "T-" + str(i + 1) + ".csv"
    tmp = pd.read_csv(fname)
    # tmp = tmp.drop('Unnamed: 12', 1)
    tmp.to_csv('outputD3_' + fname, header=True)

# 导入深度学习库 - keras相关部分
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt


def DNN(nHiddenLayers, dropout, hidden_dim):
    model = Sequential()
    model.add(Dense(hidden_dim, input_dim=11, activation='sigmoid'))  # 输入层
    for i in range(0, nHiddenLayers):  # construct hidden layers
        if dropout == True:
            model.add(Dropout(0.3))
        model.add(Dense(hidden_dim, activation='sigmoid'))
    model.add(Dense(1, activation='linear'))  # The last linear layer
    return model


# model = DNN(1, True)
# 构建优化器，用来优化模型
# rms = RMSprop(lr=0.02, rho=0.9, epsilon=1e-08, decay=0.0)
# model.compile(loss='mse',
#                optimizer=rms,
#                metrics=['mse'])

# 训练模型 epoch: 多少次循环迭代  batch_size: 每次训练取出的batch size
# model.fit(X_train, y_train, epochs=10, batch_size=100)


# copy the original documents
# for i in range(0,22):
#    fname = "T-" + str(i+1) + ".csv"
#    tmp = pd.read_csv(fname)
#    tmp.to_csv('output_' + fname, header=True)

# 训练 1层，2层，3层神经网络
#for i in range(1, 4):
for dropout in [False]:  # control the dropout
        model = DNN(3, dropout, 35)
        # 构建优化器，用来优化模型
        rms = RMSprop(lr=0.02, rho=0.9, epsilon=1e-08, decay=0.0)
        model.compile(loss='mse',
                      optimizer=rms,
                      metrics=['mse'])

        # 训练模型 epoch: 多少次循环迭代  batch_size: 每次训练取出的batch size
        model.fit(X_train, y_train, epochs=10, batch_size=100, verbose=1)
        #print model.evaluate(X_test, y_test, batch_size=100)
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        # append the results to the file
        pred = np.concatenate([pred_train, pred_test])
        # print np.shape(pred)

        # 加一个prediction colume
        for j in range(0, 22):
            fname = "outputD3_T-" + str(j + 1) + ".csv"
            tmp = pd.read_csv(fname)
            tmp['DNN_layer_'] = pd.DataFrame(pred[j * lines_per_file: (j + 1) * lines_per_file])
            tmp.to_csv(fname, header=True)


