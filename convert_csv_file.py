import csv
import sys

if (len(sys.argv) != 3):
	print("usage: python3 convert_csv_file.py –i <input file>")
	exit()
else:
	data_file = sys.argv[2]

with open(data_file, 'r') as vectors_64, open('clustering_64.txt', 'w') as file:
	rows = csv.reader(vectors_64, delimiter=',')
	file.write('vectors\n')
	for row in rows:
		i = 0
		for _string in row:
			if i == 0:
				file.write(row[0].replace(" ", "&") + ' ')
			else:
				file.write(str(_string) + ' ')
			i = i + 1
		file.write('\n')


