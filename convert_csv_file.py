import csv
import sys

if (len(sys.argv) != 5):
	print("usage: python3 convert_csv_file.py –i <input file> -o <output file>")
	exit()
else:
	for i, arg in enumerate(sys.argv):
		if arg == "-i":
			data_file = sys.argv[i + 1]
		elif arg == "-o":
			output_file = sys.argv[i + 1]

with open(data_file, 'r') as vectors_64, open(output_file, 'w+') as file:
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


