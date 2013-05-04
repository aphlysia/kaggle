#!/usr/bin/env python
'''
import csv
rows = list(csv.reader(open('../csv/train.csv')))
x, e = M.getColumns(rows, (0, 1), (float, float))
M.count(x)
'''

def getColumns(rows, indices, fs = None, printError = False):
	l = len(indices)
	xs = [[] for _ in range(l)]
	_xs = [1] * l
	errorRows = []
	if fs is None:
		fs = ((lambda x: x),) * l
	for row in rows:
		try:
			for i in range(l):
				_xs[i] = fs[i](row[indices[i]])
			for i in range(l):
				xs[i].append(_xs[i])
		except Exception as ex:
			if printError:
				print(ex)
				print('\t' + str(row))
			errorRows.append(row)
	return xs, errorRows

def count(xs):
	l = len(xs)
	c = {}
	for i in range(len(xs[0])):
		key = tuple([xs[j][i] for j in range(l)])
		c[key] = c.get(key, 0) + 1
	return c

