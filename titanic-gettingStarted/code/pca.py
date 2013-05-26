#!/usr/bin/env python

if __name__ == '__main__':
	import M
	import csv
	import numpy as np
	rows = list(csv.reader(open('../csv/train.csv')))
	x, e = M.getColumns(rows, (0, 1, 3, 4), (int, float, lambda x: 1 if x=='male' else 0, float))
	x = np.matrix(x)
	y = x[0,:]   #class
	x = x[1:,:]  #attributes
	pca = M.PCA(x)
	pca.plot(x, c = ['r' if i == 0 else 'b' for i in list(np.array(y)[0])])


