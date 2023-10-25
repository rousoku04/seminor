import numpy as np
import matplotlib.pyplot as plt	

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

import warnings
warnings.simplefilter('ignore', FutureWarning)

from data_organize import data_org as org
from loss_function import pinball_loss_alpha
from rnn_predict import prediction as prd

# constant
# elements : list
# must : '日付', '始値'.
elements = ['日付', '始値']

# input dimention (elementsから日付を抜いた分)
input_dim = len(elements) - 1

# output dimention ('始値')
output_dim = 1

# --------------------------------- Config ------------------------------------- #
# 以下のパラメータはすべて好きにいじってもらって大丈夫です。

# company : str
# list : NTT, softbank, Chuden, Kanden, KDDI, NTTData, TodenHD, ZHD
company = 'NTT'


# start : str
# 日付の開始日時を決定する。
# 形式は「年/月/日」。なお数字頭の0の有無は関係なし。
# 選択可能なのは2007/1/4以降
# ただし、コードの都合上一応上記よりも前の日付であっても動きはする。
start = '2018/1/4'

# end : str
# 2022/8/31日に固定済み

# delay : int
# 時系列データとして扱う際に何日分の幅を選ぶかのパラメータ
delay = 50

# quantile
alpha = (np.arange(9) + 1) / 10


# activation function
activation = 'relu'

# Dropout
dropout = 0.2

# ------------------------------------------------------------------------------ #

# ここをいじるのは基本的に避けたほうがいいです。
# loading data
data = org(company=company, elements=elements, start=start)

## ------------------------------------------------------------------------------------------------- ##
# ここから通常のNNにて回帰を行っている部分
# normal data
data.time_variable()

x_train = data.input
y_train = data.output_nor

x_test = data.input_test
y_true = data.output_test_nor

y_est_train = np.zeros([len(alpha), len(x_train)])
y_est_test = np.zeros([len(alpha), len(x_test)])

# Normal NN
for index, alpha_temp in enumerate(alpha):
    model = Sequential()
    ## ------------------------ 好きにいじってどうぞ ------------------------ ##
    model.add(Dense(32, input_dim=input_dim, activation=activation))



    model.add(Dense(output_dim))
    model.compile(loss=pinball_loss_alpha(alpha_temp), optimizer=Adam(0.00001))

    early_stopping = EarlyStopping(monitor='val_loss', patience=200)
    model.fit(
                x_train, y_train,
                epochs = 1000,
                batch_size = 128,
                validation_split = 0.1,
                callbacks = [early_stopping]
    )

    # NNの構造を出力する
    # model.summary()

    ## -------------------------------------------------------------------- ##


    y_est_train[index] = model.predict(x_train).reshape(-1)
    y_est_test[index] = model.predict(x_test).reshape(-1)

fig = plt.figure(figsize=(12, 16))

# 訓練データ
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(x_train[:, 0], y_train * data.output_max, label='observations', linewidth=4, linestyle='dashdot')

for i in range(len(alpha)):
    ax1.plot(x_train[:,0], y_est_train[i] * data.output_max, label='estimation' + str(alpha[i]), linewidth=6)

ax1.grid()
ax1.legend()

# テストデータ
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(x_test[:, 0], y_true * data.output_max, label='observations', linewidth=4, linestyle='dashdot')

for i in range(len(alpha)):
    ax2.plot(x_test[:,0], y_est_test[i] * data.output_max, label='estimation' + str(alpha[i]), linewidth=6)

ax2.grid()
ax2.legend()


plt.savefig('result/' + company + '_nn.png', bbox_inches='tight')
plt.clf()
plt.close()

# 通常のNNはここまで    
## ------------------------------------------------------------------------------------------------- ##

## ------------------------------------------------------------------------------------------------- ##
# RNNはここから
# time sequence data
# FIX
elements = ['日付', '始値']

data.time_sequence(delay=delay)

x_train = data.input_seq_nor
y_train = data.output_seq_nor

y_test = data.output_test_nor

# Recurrent NN
y_est_train = np.zeros([len(alpha), len(x_train)])
y_est_test = np.zeros([len(alpha), data.num_days_test])
input = np.append(x_train[-1, :, 0], y_train[-1, 0]).reshape(1, -1, 1)

for index, alpha_temp in enumerate(alpha):
    model = Sequential()
    ## ------------------------ 好きにいじってどうぞ ------------------------ ##

    model.add(LSTM(32, batch_input_shape=(None, delay, input_dim), activation='tanh', return_sequences=True))


    model.add(Dense(output_dim))
    model.compile(loss=pinball_loss_alpha(alpha_temp), optimizer=Adam(0.00001))


    early_stopping = EarlyStopping(monitor='val_loss', patience=20)
    model.fit(
                x_train, y_train,
                epochs = 100,
                batch_size = 64,
                validation_split = 0.1,
                callbacks = [early_stopping]
    )

    # NNの構造を出力する
    # model.summary()

    ## -------------------------------------------------------------------- ##


    y_est_train[index] = model.predict(x_train).reshape(-1)
    y_est_test[index] = prd(model=model, input=input, output_length=data.num_days_test)

fig = plt.figure(figsize=(12, 8))

# 訓練データ
ax1 = fig.add_subplot(2, 1, 1)

ax1.plot(np.arange(len(y_train[:, 0])), y_train[:, 0].reshape(-1) * data.output_max, label='observations', linewidth=4, linestyle='dashdot')

# TEST
for i in range(len(alpha)):
    ax1.plot(np.arange(len(y_train[:, 0])), y_est_train[i].reshape(-1) * data.output_max, label='estimation' + str(alpha[i]), linewidth=6)

ax1.grid()
ax1.legend()


# テストデータ
ax2 = fig.add_subplot(2, 1, 2)

ax2.plot(np.arange(len(y_test[:, 0])), y_test[:, 0].reshape(-1) * data.output_max, label='observations', linewidth=4, linestyle='dashdot')

# TEST
for i in range(len(alpha)):
    ax2.plot(np.arange(len(y_test[:, 0])), y_est_test[i].reshape(-1) * data.output_max, label='estimation' + str(alpha[i]), linewidth=6)

ax2.grid()
ax2.legend()

plt.savefig('result/' + company + '_rnn.png', bbox_inches='tight')
plt.clf()
plt.close()

# ここまで    
## ------------------------------------------------------------------------------------------------- ##
