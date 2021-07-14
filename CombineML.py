import time
import math
import random
import calendar
import yfinance # Fixes Yahoo Finance
import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yfin
import tensorflow as tf
import matplotlib.pyplot as plt

from collections import deque # Double-ended queue
from sklearn import preprocessing
from keras.models import Sequential
from pandas_datareader import data as pdr # Data pulling
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM, BatchNormalization
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

yfin.pdr_override() # Minimise code changes

DAYS_TO_PREDICT = 20 # How many days to predict
FUTURE_PERIOD_PREDICT = 3 # Feature creation, shifting days downwards
# RATIO_TO_PREDICT = "AAPL"
RATIO_TO_PREDICT = "BTC-USD" # Stock/Crypto prediction symbol
EPOCHS = 1
BATCH_SIZE = 64
# NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{RATIO_TO_PREDICT}-{int(time.time())}" # Model name
# RATIOS = ["AAPL", "MSFT"] # Len must be dividable by 2
RATIOS = ["BTC-USD", "BCH-USD", "ETH-USD", "XRP-USD"] # Which Stock/Cryto to pull for model training
# RATIOS = ["BTC-USD", "BCH-USD"]

# SEQ_LEN = 120 # Prediction based on how many days
# SEQ_LEN = 60
# LAST_DIM = 8
# print(len(RATIOS))
# l = 10/0
# print(l)
if len(RATIOS) == 2:
	LAST_DIM = len(RATIOS)*2 # Calculate last dimension for reshaping
	SEQ_LEN = 120
elif len(RATIOS) == 4:
	LAST_DIM = len(RATIOS)*2
	SEQ_LEN = 60

NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{RATIO_TO_PREDICT}-{int(time.time())}" # Model name


scaler = MinMaxScaler(feature_range=(0,1))

### Function adds predictions at the end of the dict
# * prediction_round - dict which holds all Crypto/Stock data
# RETURN: dict with added predictions
def create_dict(prediction_round):
	new_dict = {}

	new_row = [
		{f'{ratio}_volume':'0', f'{ratio}_close':f'{prediction_round}'} if ratio == RATIO_TO_PREDICT 
		else {f'{ratio}_volume':'0', f'{ratio}_close':'0'} 
	for ratio in RATIOS]

	new_row.append({'future':'0', 'target':'0'})

	for i in range(0, len(new_row)):
		new_dict.update(new_row[i])

	return new_dict

### Compares current value to future value
# * current - current value
# * future - future value
# RETURN: 1 if stock worth buying/0 if not
def classify(current, future):
	if float(future) > float(current):
		return 1
	else:
		return 0

### Calculates the Stock/Crypto change in percentage
# * df - full data frame
# RETURN: values in np array(x) and labels(y)
def preprocess_df(df):
	df = df.drop('future', 1)

	# Normalise
	for col in df.columns:
		if col != 'target':
			df[col] = df[col].pct_change()
			df[col] = preprocessing.scale(df[col].values)

	df.dropna(inplace=True) # Delete missing data

	# Define deque to keep list the same len
	sequential_data = []
	prev_days = deque(maxlen=SEQ_LEN)

	# Add prev day values for prediction
	for i in df.values:
		prev_days.append([n for n in i[:-1]])#all but target
		if len(prev_days) == SEQ_LEN:
			sequential_data.append([np.array(prev_days), i[-1]]) # everything but target

	random.shuffle(sequential_data)

	buys, sells = [], []

	# Separate buys and sells
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

### Pulls real data from the API
# * none
# RETURN: data as dict
def get_initial_data():
	main_data = pd.DataFrame()

	price_start_date = dt.datetime(2017, 9, 14)
	price_end_date = dt.datetime.now()

	for ratio in RATIOS:
		# data = web.DataReader(f'{ratio}', 'yahoo', price_start_date, price_end_date)
		data = pdr.get_data_yahoo(f'{ratio}', start=price_start_date, end=price_end_date)

		data.rename(columns={'Close': f'{ratio}_close', 'Volume': f'{ratio}_volume'}, inplace=True)

		data = data[[f"{ratio}_close", f"{ratio}_volume"]]

		if len(main_data)==0:
			main_data = data
		else:
			main_data = main_data.join(data)

	main_data.fillna(method='ffill', inplace=True)

	main_data['future'] = main_data[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PERIOD_PREDICT)
	main_data['target'] = list(map(classify, main_data[f'{RATIO_TO_PREDICT}_close'], main_data['future']))
	
	return main_data

main_data = get_initial_data()

times = sorted(main_data.index.values) # Sort data by values
last_10pct = sorted(main_data.index.values)[-int(0.1*len(times))] # Take last 10pct data for validation

main_validation_data = main_data[(main_data.index >= last_10pct)] # Validation
main_test_data = main_data[(main_data.index < last_10pct)] # Train

train_x, train_y = preprocess_df(main_test_data)
validation_x, validation_y = preprocess_df(main_validation_data)

print('===========================================================')
print(f"train data: {len(train_x)} validation: {len(validation_x)}")
print(f"Don't buys: {train_y.count(0)}, buys: {train_y.count(1)}")
print(f"VALIDATION Don't buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")
print('===========================================================')

# Convert train and validation data to np arrays
train_x	= np.asarray(train_x)
train_y = np.asarray(train_y)
validation_x = np.asarray(validation_x)
validation_y = np.asarray(validation_y)


## Model creation

# Input Layer
model = Sequential()
model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

# Hidden Layers
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(LSTM(32, return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))

model.add(LSTM(16))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(8, activation='relu'))
model.add(Dropout(0.2))

# Output Layer
model.add(Dense(2, activation='softmax'))

opt = tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-6)

model.compile(
	loss='sparse_categorical_crossentropy',
	optimizer=opt,
	metrics=['accuracy']
)

tensorboard = TensorBoard(log_dir=f"logs/{NAME}") # Training and Validation visualisation

# Def checkpoint and model file storing paths
filepath = 'tmp/checkpoint'
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

history = model.fit(
	train_x, train_y,
	batch_size=BATCH_SIZE,
	epochs=EPOCHS,
	validation_data=(validation_x, validation_y),
	callbacks=[tensorboard, checkpoint]
)

# Load best model from this session
model.load_weights(filepath)

score = model.evaluate(validation_x, validation_y, verbose=0)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])

### Predict future days and add prediction to the final dict
# * future_day - how many days to predict into the future
# * test_data - list of Stock/Crypto values
# * prediction_days - prediction based on how many days
# RETURN: list of Stock/Crypto values, model inputs for plotting
def PredictTomorrow(future_day=1, test_data=[], prediction_days=SEQ_LEN):
	test_start = dt.datetime(2020, 1, 1)

	# Calculate next day
	year = dt.datetime.now().year
	month = dt.datetime.now().month
	day = dt.datetime.now().day + future_day
	last_month_day = calendar.monthrange(year, month)[1]

	if day > last_month_day:
		month = month +1
		day = day - last_month_day
		if month >= 13:
			month = 1
			year = year +1

	test_end = dt.datetime(year, month, day)

	# If there is no data, pull new data from API
	if len(test_data) == 0:
		test_data = get_initial_data()
		
	actual_price = test_data[f'{RATIO_TO_PREDICT}_close'].values

	# Concat real data list and prediction list
	total_dataset = pd.concat(
		(main_data[f'{RATIO_TO_PREDICT}_close'], test_data[f'{RATIO_TO_PREDICT}_close']),
		axis=0)

	model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
	model_inputs = model_inputs.reshape(-1, 1)
	model_inputs = scaler.fit_transform(model_inputs)

	real_data = [model_inputs[len(model_inputs)-
		480:len(model_inputs)+future_day, 0]]

	real_data = np.array(real_data)
	# real_data = np.reshape(real_data, (-1, SEQ_LEN, LAST_DIM))
	real_data = np.reshape(real_data, (-1, SEQ_LEN, LAST_DIM))# 4 crypto - 8


	prediction = model.predict(real_data)
	prediction = scaler.inverse_transform(prediction)

	prediction_round = float("{:.2f}".format(prediction[0][0]))
	new_row = create_dict(prediction_round)
	test_data.loc[dt.datetime(year, month, day)] = new_row

	return test_data, model_inputs

td = main_data
model_inputs = None

# Predict several days into future
for i in range(1, DAYS_TO_PREDICT):
	td, model_inputs = PredictTomorrow(future_day=i, test_data=td)

# Add features to the new dataset
td['future'] = main_data[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PERIOD_PREDICT)
td['target'] = list(map(classify, main_data[f'{RATIO_TO_PREDICT}_close'], main_data['future']))
print(td.tail(DAYS_TO_PREDICT))

# Split data into real and predicted
predicted_data = td[f'{RATIO_TO_PREDICT}_close'][len(td)-DAYS_TO_PREDICT:-1]
real_data = td[f'{RATIO_TO_PREDICT}_close'][len(td)-365:len(td)-DAYS_TO_PREDICT+1]

# Data plotting
plt.plot(predicted_data, color='pink', label='Predictions')
plt.plot(real_data, color='blue', label='Real Data')
plt.title(f'{RATIO_TO_PREDICT} price prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc='upper left')
plt.show()


# Create additional list for holding
x_test = []

prediction_days = SEQ_LEN
for x in range(prediction_days, len(model_inputs)):
	x_test.append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)

# Reshape for extended graph
# x_test = np.reshape(x_test, (x_test.shape[0], 30, LAST_DIM))

# 4 crypto - 8
x_test = np.reshape(x_test, -1)
missing_value = math.floor(x_test.shape[0]/SEQ_LEN/LAST_DIM)
needed_value = missing_value*SEQ_LEN*LAST_DIM
final_difference = x_test.shape[0]-needed_value
x_test = x_test[final_difference-1:-1]
x_test = np.reshape(x_test, (-1, SEQ_LEN, LAST_DIM))
x_test = np.reshape(x_test, (x_test.shape[0], SEQ_LEN, LAST_DIM))

# Reshape for shorter graph
# x_test = np.reshape(x_test, -1)
# missing_value = math.floor(x_test.shape[0]/120/LAST_DIM)
# needed_value = missing_value*120*LAST_DIM
# final_difference = x_test.shape[0]-needed_value
# x_test = x_test[final_difference-1:-1]
# x_test = np.reshape(x_test, (-1, 120, LAST_DIM))

# Predict test data for fitted graph
predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)
actual_price = td[f"{RATIO_TO_PREDICT}_close"].values

# Plot final graph
plt.plot(actual_price, color='black', label="Actual Price")
plt.plot(predicted_prices, color='pink', label="Predicted Prices")
plt.title(f"{RATIO_TO_PREDICT} price prediction")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc='upper left')
plt.show()