import numpy as np
import pandas as pd
from keras import layers, optimizers, losses, metrics
from keras.models import load_model, Model
import os
import csv
import sys

if (len(sys.argv) != 5):
	print("usage: python3 new_representation.py –i <input file> -c <config file>")
	exit()
else:
	for i, arg in enumerate(sys.argv):
		print(i, sys.argv[i])
		if arg == "-i":
			data_file = sys.argv[i + 1]
		elif arg == "-c":
			config_file = sys.argv[i + 1]


model1 = load_model(config_file)
model2 = Model(model1.input, model1.layers[0].output)
model2.summary()

model2.compile(optimizer='rmsprop', loss='mse')

data = pd.read_csv(data_file, header=None).iloc[:, 1:]

results = model2.predict(data, batch_size=32)
#print(results.shape)
#print(results)

with open('new_representation.csv', 'w') as f, open(data_file, "r") as nn_repr:
	values = csv.reader(nn_repr, delimiter=',')
	for (result, value) in zip(results, values):
		f.write(value[0] + ' ')
		i = 0
		for res in result:
			f.write(str(res))
			if i != len(result) - 1:
				f.write(',')
			i = i + 1	
		f.write('\n')