import pandas as pd   # 用于读取数据，做预处理
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt


#DNN-1 layer
dfx1 = pd.read_csv("outputD_T-19.csv")
for i in range(19,22):
    tmpx = pd.read_csv("outputD_T-" + str(i+1) + ".csv")
    dfx1 = pd.concat([dfx1, tmpx],ignore_index=True)
    x1 = dfx1['DNN_layer_']
    x1 = x1.where(x1 > 0, 0)
    #print(x)

#DNN-2 layer
dfx2 = pd.read_csv("outputD2_T-19.csv")
for i in range(19,22):
    tmpx = pd.read_csv("outputD2_T-" + str(i+1) + ".csv")
    dfx2 = pd.concat([dfx2, tmpx],ignore_index=True)
    x2 = dfx2['DNN_layer_']
    x2 = x2.where(x2 > 0, 0)

#DNN-3 layer
dfx3 = pd.read_csv("outputD_T-22.csv")
for i in range(19,22):
    tmpx = pd.read_csv("outputD_T-" + str(i+1) + ".csv")
    dfx3 = pd.concat([dfx3, tmpx],ignore_index=True)
    x3 = dfx3['DNN_layer_']
    x3 = x3.where(x3 > 0, 0)

#Support Vector
dfx4 = pd.read_csv("output2_T-22.csv")
for i in range(19,22):
    tmpx = pd.read_csv("output2_T-" + str(i+1) + ".csv")
    dfx4 = pd.concat([dfx4, tmpx],ignore_index=True)
    x4 = dfx4['SVM']
    x4 = x4.where(x4 > 0, 0)

#Linear Regression
dfx5 = pd.read_csv("output4_T-22.csv")
for i in range(19,22):
    tmpx = pd.read_csv("output4_T-" + str(i+1) + ".csv")
    dfx5 = pd.concat([dfx5, tmpx],ignore_index=True)
    x5 = dfx5['Linear']
    x5 = x5.where(x5 > 0, 0)

#KNN-3
dfx6 = pd.read_csv("output3_T-22.csv")
for i in range(19,22):
    tmpx = pd.read_csv("output3_T-" + str(i+1) + ".csv")
    dfx6 = pd.concat([dfx6, tmpx],ignore_index=True)
    x6 = dfx6['KNN']
    x6 = x6.where(x6 > 0, 0)

#KNN-5
dfx7 = pd.read_csv("output32_T-22.csv")
for i in range(19,22):
    tmpx = pd.read_csv("output32_T-" + str(i+1) + ".csv")
    dfx7 = pd.concat([dfx7, tmpx],ignore_index=True)
    x7 = dfx7['KNN']
    x7 = x7.where(x7 > 0, 0)

#KNN-7
dfx8 = pd.read_csv("output33_T-22.csv")
for i in range(19,22):
    tmpx = pd.read_csv("output33_T-" + str(i+1) + ".csv")
    dfx8 = pd.concat([dfx8, tmpx],ignore_index=True)
    x8 = dfx8['KNN']
    x8 = x8.where(x8 > 0, 0)

#Read PBM data
dfy = pd.read_csv("T-19.csv")
for i in range(19,22):
    tmpy = pd.read_csv("T-" + str(i+1) + ".csv")
    dfy = pd.concat([dfy, tmpy],ignore_index=True)
    y = dfy['Output']/10000


def indices(y,func):
 return [j for (j,val) in enumerate(y) if func(val)]
inds=indices(y,lambda x:x>0)
#print (inds)

y=y[inds]

x1=x1[inds]
x2=x2[inds]
x3=x3[inds]
x4=x4[inds]
x5=x5[inds]
x6=x6[inds]
x7=x7[inds]
x8=x8[inds]
#print(y)

Err1 = 2*((x1)-(y))/(y+x1)
Err2 = 2*((x2)-(y))/(y+x2)
Err3 = 2*((x3)-(y))/(y+x3)
Err4 = 2*((x4)-(y))/(y+x4)
Err5 = 2*((x5)-(y))/(y+x5)
Err6 = 2*((x6)-(y))/(y+x6)
Err7 = 2*((x7)-(y))/(y+x7)
Err8 = 2*((x8)-(y))/(y+x8)
#print (Err1)

#Calculate the rmse
rms1 = sqrt(mean_squared_error(y, x1))
rms2 = sqrt(mean_squared_error(y, x2))
rms3 = sqrt(mean_squared_error(y, x3))
rms4 = sqrt(mean_squared_error(y, x4))
rms5 = sqrt(mean_squared_error(y, x5))
rms6 = sqrt(mean_squared_error(y, x6))
rms7 = sqrt(mean_squared_error(y, x7))
rms8 = sqrt(mean_squared_error(y, x8))

#Calculate the mae
mae1=mean_absolute_error(y, x1)
mae2=mean_absolute_error(y, x2)
mae3=mean_absolute_error(y, x3)
mae4=mean_absolute_error(y, x4)
mae5=mean_absolute_error(y, x5)
mae6=mean_absolute_error(y, x6)
mae7=mean_absolute_error(y, x7)
mae8=mean_absolute_error(y, x8)

#Calculate the MBE
x1=np.array(x1)
x2=np.array(x2)
x3=np.array(x3)
x4=np.array(x4)
x5=np.array(x5)
x6=np.array(x6)
x7=np.array(x7)
x8=np.array(x8)
y=np.array(y)
forecast_errors1 = [x1[i]-y[i] for i in range(len(x1))]
bias1 = sum(forecast_errors1) * 1.0/len(x1)
forecast_errors2 = [x2[i]-y[i] for i in range(len(x2))]
bias2 = sum(forecast_errors2) * 1.0/len(x2)
forecast_errors3 = [x3[i]-y[i] for i in range(len(x3))]
bias3 = sum(forecast_errors3) * 1.0/len(x3)
forecast_errors4 = [x4[i]-y[i] for i in range(len(x4))]
bias4 = sum(forecast_errors4) * 1.0/len(x4)
forecast_errors5 = [x5[i]-y[i] for i in range(len(x5))]
bias5 = sum(forecast_errors5) * 1.0/len(x5)
forecast_errors6 = [x6[i]-y[i] for i in range(len(x6))]
bias6 = sum(forecast_errors6) * 1.0/len(x6)
forecast_errors7 = [x7[i]-y[i] for i in range(len(x7))]
bias7 = sum(forecast_errors7) * 1.0/len(x7)
forecast_errors8 = [x8[i]-y[i] for i in range(len(x8))]
bias8 = sum(forecast_errors8) * 1.0/len(x8)


NS1= 1 - sum((y-x1)**2)/sum((y-np.mean(y))**2)
NS2= 1 - sum((y-x2)**2)/sum((y-np.mean(y))**2)
NS3= 1 - sum((y-x3)**2)/sum((y-np.mean(y))**2)
NS4= 1 - sum((y-x4)**2)/sum((y-np.mean(y))**2)
NS5= 1 - sum((y-x5)**2)/sum((y-np.mean(y))**2)
NS6= 1 - sum((y-x6)**2)/sum((y-np.mean(y))**2)
NS7= 1 - sum((y-x7)**2)/sum((y-np.mean(y))**2)
NS8= 1 - sum((y-x8)**2)/sum((y-np.mean(y))**2)
#print(r2_score(x1, y))

print(NS1,NS2,NS3,NS4,NS5,NS6,NS7,NS8)



plt.rc('font',family='Times New Roman')


data = [Err1, Err2,Err3,Err4,Err5,Err6,Err7,Err8]


plt.boxplot(data,sym=".", vert=True, whis=1.5)
#plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], ['DNN-1', 'DNN-2', 'DNN-3','SVM','Linear','kNN-3','kNN-5','kNN-7'],fontname='Times new Roman',fontsize='large')
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8],[])
plt.ylim((-2.5,2.5))
plt.margins(0.2)
plt.ylabel('Relative Difference',fontname='Times new Roman',fontsize='x-large')


#Creat table place under the plot
columns = ('DNN-1', 'DNN-2', 'DNN-3','SVM','Linear','kNN-3','kNN-5','kNN-7')
rows = ['RMSE','MAE','MBE','EF']

Tdata=[[rms1,rms2,rms3,rms4,rms5,rms6,rms7,rms8],
       [mae1,mae2,mae3,mae4,mae5,mae6,mae7,mae8],
       [bias1,bias2,bias3,bias4,bias5,bias6,bias7,bias8],
       [NS1,NS2,NS3,NS4,NS5,NS6,NS7,NS8]]
print(Tdata)
n_rows=len(Tdata)
y_offset = np.array([0.0] * len(columns))
print(y_offset)
cell_text=[]
for row in range(n_rows):
    y_offset = Tdata[row]
    cell_text.append(['%1.2f' % x for x in y_offset])
print(y_offset)
the_table = plt.table(cellText=cell_text,
                      rowLabels=rows,
                      colLabels=columns,
                      loc='bottom',cellLoc='center')
the_table.scale(1,1.5)
plt.subplots_adjust(left=0.2, bottom=0.2)
plt.show()



#xd=x[:,0]
#yd=y[:,0]
