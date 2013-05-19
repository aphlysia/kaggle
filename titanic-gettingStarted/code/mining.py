#!/usr/bin/env python

if __name__ == '__main__':
	import M
	import csv
	import numpy as np
	rows = list(csv.reader(open('../csv/train.csv')))
	'''
	x, e = M.getColumns(rows, (0, 3, (int, lambda x: 1 if x=='male' else 0)))
	print(M.count(x))
	'''
	x, e = M.getColumns(rows, (0, 1, 3, 4), (int, float, lambda x: 1 if x=='male' else 0, float))
	x = np.matrix(x)
	y = x[0,:]   #class
	x = x[1:,:]  #attributes
	x0, x1 = M.split(x, y, lambda _y: _y==0)
	'''
	success, failure = M.crossValidation(x0, x1, M.Fisher)
	print(success, failure)
	'''

	d = M.Gauss(x0, x1)
	rows = list(csv.reader(open('../csv/test.csv')))[1:]
	for row in rows:
		try:
			x = []
			x.append(float(row[0]))
			x.append(1 if row[2] == 'male' else 0)
			x.append(float(row[3]))
			print(d.do(np.matrix(x).T))
		except ValueError:
			print("0,#")

