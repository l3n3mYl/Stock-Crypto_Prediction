import time
import random
import numpy as np
import pandas as pd
import datetime as dt
import tensorflow as tf
import pandas_datareader as web

from collections import deque
from sklearn import preprocessing
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM, BatchNormalization
from keras.callbacks import TensorBoard, ModelCheckpoint

SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3
RATIO_TO_PREDICT = "BTC-USD"
EPOCHS = 13
BATCH_SIZE = 64
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"

def classify(current, future):
	if float(future) > float(current):
		return 1
	else:
		return 0

def preprocess_df(df):
	df = df.drop('future', 1)

	for col in df.columns:
		if col != 'target':
			df[col] = df[col].pct_change()
			df.dropna(inplace=True)
			df[col] = preprocessing.scale(df[col].values)

	df.dropna(inplace=True)

	sequential_data = []
	prev_days = deque(maxlen=SEQ_LEN)

	for i in df.values:
		prev_days.append([n for n in i[:-1]])#all but target
		if len(prev_days) == SEQ_LEN:
			sequential_data.append([np.array(prev_days), i[-1]])

	random.shuffle(sequential_data)

	buys, sells = [], []

	for seq, target in sequential_data:
		if target == 0:
			sells.append([seq, target])
		else:
			buys.append([seq, target])

	random.shuffle(buys)
	random.shuffle(sells)

	lower = min(len(buys), len(sells))

	sequential_data = buys+sells
	random.shuffle(sequential_data)

	x, y = [], []

	for seq, target in sequential_data:
		x.append(seq)
		y.append(target)

	return np.array(x), y

main_data = pd.DataFrame()

ratios = ["BTC-USD", "LTC-USD", "BCH-USD", "ETH-USD"]

price_start_date = dt.datetime(2015, 1, 1)
price_end_date = dt.datetime.now()

scaler = MinMaxScaler(feature_range=(0,1))

for ratio in ratios:
	data = web.DataReader(f'{ratio}', 'yahoo', price_start_date, price_end_date)

	data.rename(columns={'Close': f'{ratio}_close', 'Volume': f'{ratio}_volume'}, inplace=True)

	data = data[[f"{ratio}_close", f"{ratio}_volume"]]

	if len(main_data)==0:
		main_data = data
	else:
		main_data = main_data.join(data)

main_data.fillna(method='ffill', inplace=True)
main_data.dropna(inplace=True)

main_data['future'] = main_data[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PERIOD_PREDICT)
main_data['target'] = list(map(classify, main_data[f'{RATIO_TO_PREDICT}_close'], main_data['future']))

main_data.dropna(inplace=True)

times = sorted(main_data.index.values)
last_5pct = sorted(main_data.index.values)[-int(0.05*len(times))]

validation_main_data = main_data[(main_data.index >= last_5pct)]
main_data = main_data[(main_data.index < last_5pct)]

train_x, train_y = preprocess_df(main_data)
validation_x, validation_y = preprocess_df(validation_main_data)

print(f"train data: {len(train_x)} validation: {len(validation_x)}")
print(f"Don't buys: {train_y.count(0)}, buys: {train_y.count(1)}")
print(f"VALIDATION Don't buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")

train_x	= np.asarray(train_x)
train_y = np.asarray(train_y)
validation_x = np.asarray(validation_x)
validation_y = np.asarray(validation_y)

model = Sequential()
model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))

opt = tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-6)

model.compile(
	loss='sparse_categorical_crossentropy',
	optimizer=opt,
	metrics=['accuracy']
)

tensorboard = TensorBoard(log_dir=f"logs/{NAME}")

filepath = "models/RNN_Final-{epoch:02d}-{val_accuracy:.3f}.hd5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

history = model.fit(
	train_x, train_y,
	batch_size=BATCH_SIZE,
	epochs=EPOCHS,
	validation_data=(validation_x, validation_y),
	callbacks=[tensorboard, checkpoint]
)

score = model.evaluate(validation_x, validation_y, verbose=0)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])

model.save(f"models/{NAME}")