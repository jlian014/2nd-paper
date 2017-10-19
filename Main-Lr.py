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
    tmp.to_csv('output4_' + fname, header=True)

from sklearn.metrics import mean_squared_error
from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(X_train, y_train)

pred_train = model.predict(X_train)
pred_test = model.predict(X_test)
pred = np.concatenate([pred_train, pred_test])
        # 加一个prediction colume
for j in range(0, 22):
            fname = "output4_T-" + str(j + 1) + ".csv"
            tmp = pd.read_csv(fname)
            tmp['Linear'] = pd.DataFrame(pred[j * lines_per_file: (j + 1) * lines_per_file])
            tmp.to_csv(fname, header=True)
