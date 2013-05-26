#!/usr/bin/env python

if __name__ == '__main__':
	import M
	import csv
	import numpy as np
	import matplotlib
	import matplotlib.pyplot as plt
	rows = list(csv.reader(open('../csv/train.csv')))
	x, e = M.getColumns(rows, (0, 1, 3, 4), (int, float, lambda x: 1 if x=='male' else 0, float))
	x = np.matrix(x)
	y = x[0,:]   #class
	x = x[1:,:]  #attributes
	pca = M.PCA(x)
	v = pca.do(x)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(list(np.array(v[0])[0]), list(np.array(v[1])[0]), c = ['r' if i == 0 else 'b' for i in list(np.array(y)[0])])
	plt.show()

