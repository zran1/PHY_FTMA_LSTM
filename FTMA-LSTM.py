import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from keras.layers import Multiply
from keras.layers.core import *
from keras.layers.rnn.lstm import LSTM
from keras.models import *
from keras import backend as K
import matplotlib.pyplot as plt
from keras import  callbacks

def R2(ym, ys):
    R = np.corrcoef(ym, ys)[0,1]
    R2 = R ** 2  
    return R2

def NSE(ym, ys):
    x_mean = ys.mean()
    SST = np.sum((ys - x_mean)**2)
    SSRes = np.sum((ys - ym)**2)
    NSE = 1 - (SSRes / SST)
    return NSE

def KGE(ym, ys):
    ymAve = np.mean(ym)
    ysAve = np.mean(ys)
    COV = np.cov(ym, ys)
    CC = COV[0, 1] / np.std(ym) / np.std(ys)
    BR = ymAve / ysAve
    RV = (np.std(ym) / ymAve) / (np.std(ys) / ysAve)
    KGE = 1 - np.sqrt((CC - 1) ** 2 + (BR - 1) ** 2 + (RV - 1) ** 2)
    return KGE


df1=pd.read_excel(r'D:/Pycharm/data.xlsx')


feat = df1.iloc[:, 1:5]  
date = df1.iloc[:, 0]   

from sklearn import preprocessing  
min_max_scaler = preprocessing.MinMaxScaler()
df0=min_max_scaler.fit_transform(feat)
maxs = min_max_scaler.data_max_[0]  
mins = min_max_scaler.data_min_[0]  


df = pd.DataFrame(df0)
train_num1 = 201  
train_num2 = 272
train_num3 = 361
train_num4 = 457
train_num5 = 558
train_num6 = 617
train_num7 = 700
train_num8 = 869
train_num9 = 910
train_num10 = 989
train_num11 = 1271
train_num12 = 1392
train_num13 = 1733
train_num14 = 2066
train_num15 = 2195
train_num16 = 2288
train_num17 = 2501
train_num18 = 2698
train_num19 = 2871
train_num20 = 3060
val_num1 = 3121
val_num2 = 3289
val_num3 = 3414  
val_num4 = 3503
val_num5 = 3582
val_num6 = 3771
test_num1 = 3916
test_num2 = 4040
test_num3 = 4179
test_num4 = 4284
test_num5 = 4468
test_num6 = 4565
targets = df.iloc[:,0]  


def TimeSeries(dataset, start_index, history_size, end_index, step,
               target_size, true):
    data = []  
    labels = []  

    start_index = start_index + history_size  

    
    for i in range(start_index, end_index - target_size):
        
        index = range(i - history_size, i, step)  
        
        data.append(dataset.iloc[index])
        labels.append(true[i + target_size])
    
    return np.array(data), np.array(labels)

history_size = 12  
target_size = 0  
step = 1  
batchsize = 64
units = 256
lr = 0.001
epochs = 200  

x_train1, y_train1 = TimeSeries(dataset=df, start_index=0, history_size=history_size, end_index=train_num1, step=step, target_size=target_size, true=targets)
x_train2, y_train2 = TimeSeries(dataset=df, start_index=train_num1, history_size=history_size, end_index=train_num2, step=step, target_size=target_size, true=targets)
x_train3, y_train3 = TimeSeries(dataset=df, start_index=train_num2, history_size=history_size, end_index=train_num3, step=step, target_size=target_size, true=targets)
x_train4, y_train4 = TimeSeries(dataset=df, start_index=train_num3, history_size=history_size, end_index=train_num4, step=step, target_size=target_size, true=targets)
x_train5, y_train5 = TimeSeries(dataset=df, start_index=train_num4, history_size=history_size, end_index=train_num5, step=step, target_size=target_size, true=targets)
x_train6, y_train6 = TimeSeries(dataset=df, start_index=train_num5, history_size=history_size, end_index=train_num6, step=step, target_size=target_size, true=targets)
x_train7, y_train7 = TimeSeries(dataset=df, start_index=train_num6, history_size=history_size, end_index=train_num7, step=step, target_size=target_size, true=targets)
x_train8, y_train8 = TimeSeries(dataset=df, start_index=train_num7, history_size=history_size, end_index=train_num8, step=step, target_size=target_size, true=targets)
x_train9, y_train9 = TimeSeries(dataset=df, start_index=train_num8, history_size=history_size, end_index=train_num9, step=step, target_size=target_size, true=targets)
x_train10, y_train10 = TimeSeries(dataset=df, start_index=train_num9, history_size=history_size, end_index=train_num10, step=step, target_size=target_size, true=targets)
x_train11, y_train11 = TimeSeries(dataset=df, start_index=train_num10, history_size=history_size, end_index=train_num11, step=step, target_size=target_size, true=targets)
x_train12, y_train12 = TimeSeries(dataset=df, start_index=train_num11, history_size=history_size, end_index=train_num12, step=step, target_size=target_size, true=targets)
x_train13, y_train13 = TimeSeries(dataset=df, start_index=train_num12, history_size=history_size, end_index=train_num13, step=step, target_size=target_size, true=targets)
x_train14, y_train14 = TimeSeries(dataset=df, start_index=train_num13, history_size=history_size, end_index=train_num14, step=step, target_size=target_size, true=targets)
x_train15, y_train15 = TimeSeries(dataset=df, start_index=train_num14, history_size=history_size, end_index=train_num15, step=step, target_size=target_size, true=targets)
x_train16, y_train16 = TimeSeries(dataset=df, start_index=train_num15, history_size=history_size, end_index=train_num16, step=step, target_size=target_size, true=targets)
x_train17, y_train17 = TimeSeries(dataset=df, start_index=train_num16, history_size=history_size, end_index=train_num17, step=step, target_size=target_size, true=targets)
x_train18, y_train18 = TimeSeries(dataset=df, start_index=train_num17, history_size=history_size, end_index=train_num18, step=step, target_size=target_size, true=targets)
x_train19, y_train19 = TimeSeries(dataset=df, start_index=train_num18, history_size=history_size, end_index=train_num19, step=step, target_size=target_size, true=targets)
x_train20, y_train20 = TimeSeries(dataset=df, start_index=train_num19, history_size=history_size, end_index=train_num20, step=step, target_size=target_size, true=targets)


x_val1, y_val1 = TimeSeries(dataset=df, start_index=train_num20, history_size=history_size, end_index=val_num1, step=step, target_size=target_size, true=targets)
x_val2, y_val2 = TimeSeries(dataset=df, start_index=val_num1, history_size=history_size, end_index=val_num2, step=step, target_size=target_size, true=targets)
x_val3, y_val3 = TimeSeries(dataset=df, start_index=val_num2, history_size=history_size, end_index=val_num3, step=step, target_size=target_size, true=targets)
x_val4, y_val4 = TimeSeries(dataset=df, start_index=val_num3, history_size=history_size, end_index=val_num4, step=step, target_size=target_size, true=targets)
x_val5, y_val5 = TimeSeries(dataset=df, start_index=val_num4, history_size=history_size, end_index=val_num5, step=step, target_size=target_size, true=targets)
x_val6, y_val6 = TimeSeries(dataset=df, start_index=val_num5, history_size=history_size, end_index=val_num6, step=step, target_size=target_size, true=targets)

x_test1, y_test1 = TimeSeries(dataset=df, start_index=val_num6, history_size=history_size, end_index=test_num1, step=step, target_size=target_size,  true=targets)
x_test2, y_test2 = TimeSeries(dataset=df, start_index=test_num1, history_size=history_size, end_index=test_num2, step=step, target_size=target_size,  true=targets)
x_test3, y_test3 = TimeSeries(dataset=df, start_index=test_num2, history_size=history_size, end_index=test_num3, step=step, target_size=target_size,  true=targets)
x_test4, y_test4 = TimeSeries(dataset=df, start_index=test_num3, history_size=history_size, end_index=test_num4, step=step, target_size=target_size,  true=targets)
x_test5, y_test5 = TimeSeries(dataset=df, start_index=test_num4, history_size=history_size, end_index=test_num5, step=step, target_size=target_size,  true=targets)
x_test6, y_test6 = TimeSeries(dataset=df, start_index=test_num5, history_size=history_size, end_index=test_num6, step=step, target_size=target_size,  true=targets)



x_train=np.concatenate((x_train1,x_train2,x_train3,x_train5,x_train7,x_train8,x_train11,x_train12,x_train13,x_train15,x_train17,x_train19,x_train20,x_val2,x_val3,x_val4,x_val6,x_test1,x_test2,x_test6))
y_train=np.concatenate((y_train1,y_train2,y_train3,y_train5,y_train7,y_train8,y_train11,y_train12,y_train13,y_train15,y_train17,y_train19,y_train20,y_val2,y_val3,y_val4,y_val6,y_test1,y_test2,y_test6))
x_test=np.concatenate((x_train4,x_train6,x_train9,x_train14,x_train18,x_val1,x_val5,x_test3,x_test5,x_test4))
y_test=np.concatenate((y_train4,y_train6,y_train9,y_train14,y_train18,y_val1,y_val5,y_test3,y_test5,y_test4))




train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))  
train_ds = train_ds.batch(batchsize)  


test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))  
test_ds = test_ds.batch(batchsize)


sample = next(iter(train_ds))  
inputs_shape = sample[0].shape[1:]  
inputs = keras.Input(shape=inputs_shape)  


input_dim = int(inputs.shape[2])
TIME_STEPS = history_size

def multihead_attention_3d_block(inputs, n_heads=3):
    input_dim = int(inputs.shape[2])
    head_outputs = []
    for i in range(n_heads):
        a1 = Permute((2, 1))(inputs)
        a1 = Dense(TIME_STEPS, activation='softmax')(a1)
        a1 = Permute((2, 1))(a1)
        a2 = Dense(input_dim, activation='softmax')(inputs)
        a_probs = Multiply()([a1, a2])
        head_outputs.append(a_probs)
    attention_output = head_outputs[0] if n_heads == 1 else keras.layers.Add()(head_outputs) / n_heads
    attention_output = Multiply()([attention_output, inputs])
    return attention_output

def model_attention_applied_before_lstm():
    K.clear_session() 
    attention_mul = multihead_attention_3d_block(inputs)
    attention_mul = LSTM(units, return_sequences=False)(attention_mul)
    dense1 = Dense(units, activation='relu')(attention_mul)
    output = Dense(1)(dense1)
    model = Model(inputs=[inputs], outputs=output)
    return model



model = model_attention_applied_before_lstm()
model.summary()


model.compile(optimizer=keras.optimizers.Adam(lr), loss='mean_squared_error') 

es = callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=20,
                                 restore_best_weights=True)

history = model.fit(train_ds, epochs=epochs, callbacks= [es])

model.evaluate(test_ds)  



history_dict = history.history  
train_loss = history_dict['loss']  


y_moni = model.predict(x_train)  
y_t = y_moni[:,0]
y_t = y_t*(maxs-mins)+mins
ytzhen = y_train*(maxs-mins)+mins

mapet = np.mean(np.abs((y_t-ytzhen)/(ytzhen)))*100
print("NSEt = ",NSE(y_t, ytzhen)) 
print("KGEt = ",KGE(y_t, ytzhen)) 
print("R2t = ",R2(y_t, ytzhen)) 
print("RMSEt = ", np.sqrt(mean_squared_error(ytzhen, y_t)))
print("MAEt = ",mean_absolute_error(y_t, ytzhen))
print("MAPEt = ",mapet,'%')
train = {"train":ytzhen,"moni":y_t}
train = pd.DataFrame(train)


x_predict = x_test  
y_true = y_test  
y_predict = model.predict(x_predict)  
y_predict = y_predict[:,0]
y_true = y_true*(maxs-mins)+mins
y_predict = y_predict*(maxs-mins)+mins


result={"test":y_true,"pred":y_predict}
result=pd.DataFrame(result)

mape1 = np.mean(np.abs((y_predict-y_true)/(y_true)))*100
print("NSE1 = ",NSE(y_predict, y_true)) 
print("KGE1 = ",KGE(y_predict, y_true)) 
print("R21 = ",R2(y_predict, y_true)) 
print("RMSE1 = ", np.sqrt(mean_squared_error(y_true, y_predict)))
print("MAE1 = ",mean_absolute_error(y_predict, y_true))
print("MAPE1 = ",mape1,'%')



plt.figure()
draw=pd.concat([pd.DataFrame(y_true),pd.DataFrame(y_predict)],axis=1);
draw.iloc[:,0].plot(figsize=(12,6))  
draw.iloc[:,1].plot(figsize=(12,6))  
plt.legend(('real', 'predict'),loc='upper right',fontsize='15')
plt.title("Test Data",fontsize='30') 
plt.savefig(r"D:\Pycharm\FTMA-LSTM\对比.png",dpi=300)
train.to_excel(r"D:\Pycharm\FTMA-LSTM训练结果.xlsx")
result.to_excel(r"D:\Pycharm\FTMA-LSTM预测结果.xlsx")
