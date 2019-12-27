import numpy as np
import pandas as pd
from keras import layers, optimizers, losses, metrics
from keras.models import load_model
import sys, csv

if (len(sys.argv) != 3):
	print("usage: python3 new_representation.py –i <input file>")
	exit()
else:
	for i, arg in enumerate(sys.argv):
		print(i, sys.argv[i])
		if arg == "-i":
			data_file = sys.argv[i + 1]
		elif arg == "-a":
			actual_file = sys.argv[i + 1]

model = load_model('./data/WindDenseNN.h5')
model.summary()

data = pd.read_csv(data_file).iloc[:, 1:]
labels = pd.read_csv('./data/actual.csv').iloc[:, 1:]

#mean_absolute_error
model.compile(optimizer=optimizers.RMSprop(0.01), loss='mean_absolute_error',
	metrics=[metrics.CategoricalAccuracy()])
[mae, x] = model.evaluate(data, labels, batch_size=32)

#mean_absolute_percentage_error
model.compile(optimizer=optimizers.RMSprop(0.01), loss='mean_absolute_percentage_error',
	metrics=[metrics.CategoricalAccuracy()])
[mape, x] = model.evaluate(data, labels, batch_size=32)

#mean_squared_error
model.compile(optimizer=optimizers.RMSprop(0.01), loss='mean_squared_error',
	metrics=[metrics.CategoricalAccuracy()])
[mse, x] = model.evaluate(data, labels, batch_size=32)

results = model.predict(data, batch_size=32)

with open('predicted.csv', 'w') as f, open(data_file, "r") as nn_repr:	
	values = csv.reader(nn_repr, delimiter=',')
	f.write(f'MAE: {mae} MAPE: {mape} MSE: {mse}\n')
	for (result, value) in zip(results, values):
		f.write(value[0] + ' ')
		i = 0
		for res in result:
			f.write(str(res))
			if i != len(result) - 1:
				f.write(',')
			i = i + 1	
		f.write('\n')
