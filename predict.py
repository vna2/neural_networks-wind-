import numpy as np
import pandas as pd
from keras import layers, optimizers, losses, metrics
from keras.models import load_model
import sys, csv



def mean_absolute_percentage_error(y_true, y_pred):
	y_true, y_pred = np.array(y_true), np.array(y_pred)
	y_true, y_pred = check_array(y_true), check_array(y_pred)
	return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def check_array(array):
	for i, value in enumerate(array):
		if value == 0:
			array[i] = my_mean(array)
	return array

def my_mean(array):
	sum = 0
	for a in array:
		sum += a
	return sum/(len(array) - 1)


# print(check_array([1,0,8]))


# exit()


if (len(sys.argv) != 7):
	print("usage: python3 predict.py –i <input file> -a <actual file> -c <config file>")
	exit()
else:
	for i, arg in enumerate(sys.argv):
		if arg == "-i":
			data_file = sys.argv[i + 1]
		elif arg == "-a":
			actual_file = sys.argv[i + 1]
		elif arg == "-c":
			config_file = sys.argv[i + 1]

model = load_model(config_file)
model.summary()


data = pd.read_csv(data_file, header=None).iloc[:, 1:]
labels = pd.read_csv(actual_file, header=None).iloc[:, 1:]

#mean_absolute_error
model.compile(optimizer=optimizers.RMSprop(0.01), loss='mean_absolute_error',
	metrics=[metrics.CategoricalAccuracy()])
[mae, x] = model.evaluate(data, labels, batch_size=32)

#mean_squared_error
model.compile(optimizer=optimizers.RMSprop(0.01), loss='mean_squared_error',
	metrics=[metrics.CategoricalAccuracy()])
[mse, x] = model.evaluate(data, labels, batch_size=32)

predictions = model.predict(data, batch_size=32)

#mean_absolute_percentage_error
with open(actual_file, "r") as actual:	
	actual_values = csv.reader(actual, delimiter=',')
	mape = 0
	for (pred_row, actual_row) in zip(predictions, actual_values):
		actual_row = list(map(float, actual_row[1:]))
		mape += mean_absolute_percentage_error(actual_row, pred_row)
	mape /= labels.shape[0]

with open('predicted.csv', 'w+') as f, open(data_file, "r") as nn_repr:	
	values = csv.reader(nn_repr, delimiter=',')
	f.write(f'MAE: {mae} MAPE: {mape} MSE: {mse}\n')
	for (result, value) in zip(predictions, values):
		f.write(value[0] + ' ')
		i = 0
		for res in result:
			f.write(str(res))
			if i != len(result) - 1:
				f.write(',')
			i = i + 1	
		f.write('\n')
	f.close()