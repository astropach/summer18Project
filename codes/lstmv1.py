from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import median_absolute_error

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
import numpy

ttData = -30

# date-time parsing function for loading the dataset
def parser(x):
	return datetime.strptime(x, '%Y-%m-%d')

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# scale train and test data to [-1, 1]
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = numpy.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mse', 'mae', 'mape', 'cosine'])
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()
	return model

# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]

# load dataset
series = read_csv('infyDown.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser,usecols=[0,4])

# transform data to be stationary
raw_values = series.values
diff_values = difference(raw_values, 1)
print("transformed stationary")
# transform data to be supervised learning
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values
print("transformed supervised")
# split data into train and test-sets
train, test = supervised_values[0:ttData], supervised_values[ttData:]
print("split done")
print(test.size)
print(train.size)
# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)
print("transformed scale")

# fit the model
lstm_model = fit_lstm(train_scaled, 1, 1, 2)
print("fit done")
# forecast the entire training dataset to build up state for forecasting
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
lstm_model.predict(train_reshaped, batch_size=1)
print("reshaped")
# walk-forward validation on the test data
predictions = list()
for i in range(len(test_scaled)):
	# make one-step forecast
	X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
	yhat = forecast_lstm(lstm_model, 1, X)
	# invert scaling
	yhat = invert_scale(scaler, X, yhat)
	# invert differencing
	yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
	# store forecast
	predictions.append(yhat)
	expected = raw_values[len(train) + i + 1]
	print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))

# report performance
rmse = sqrt(mean_squared_error(raw_values[ttData:], predictions))
r2score = r2_score(raw_values[ttData:], predictions)
mae = mean_absolute_error(raw_values[ttData:], predictions)
msle = mean_squared_log_error(raw_values[ttData:], predictions)
evs = explained_variance_score(raw_values[ttData:], predictions)
mae2 = median_absolute_error(raw_values[ttData:], predictions)

print('Test RMSE: %.3f' % rmse)
print('Test r2Score: %.3f' % r2score)
print('Test ms log err: %.3f' % msle)
print('Test mean abs err: %.3f' % mae)
print('Test median abs err: %.3f' % mae2)
print('Test explained variance score: %.3f' % evs)

# line plot of observed vs predicted
pyplot.plot(raw_values[ttData:],label='Actual',linewidth=0.5)
pyplot.plot(predictions,label='Predicted',linewidth=0.5)
pyplot.legend()
pyplot.xlabel('Test RMSE: %.3f' % rmse)
pyplot.title("INFY")
pyplot.show()