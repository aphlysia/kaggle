#!/usr/bin/env python
import numpy

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

def split(x, class0):
	'''
	split x into class0 and class1
	'''
	x = numpy.matrix(x)
	r, c = x.shape
	x0 = None
	x1 = None
	for i in range(c):
		if x[0,i] == class0:
			x0 = x[1:,i] if x0 is None else numpy.hstack((x0, x[1:,i]))
		else:
			x1 = x[1:,i] if x1 is None else numpy.hstack((x1, x[1:,i]))
	return x0, x1

def mean(x):
	'''
	x[i,j] is the i-th elements of the x at the j trial
	'''
	_x = []
	r, c = x.shape
	for i in range(r):
		_x.append(x[i].mean())
	return numpy.matrix(_x, float).T

def covariance(x):
	'''
	x[i,j] is the i-th elements of the x at the j trial
	'''
	a = numpy.matrix(x, float)
	r, c = a.shape
	for i in range(r):
		a[i] -= a[i].mean()
	return a.dot(a.T) / float((c - 1))

def MahalanobisD(x, mu, SI):
	_x = x - mu
	return (_x.T * SI * _x)[0,0]
	
class Discriminator:
	def __init__(self, x0 = None, x1 = None):
		if x1 is not None and x1 is not None:
			self.train(x0, x1)
	def train(x0, x1):
		raise NotImprementedError
	def do(x):
		raise NotImprementedError

class Fisher(Discriminator):
	def train(self, x0, x1):
		self.mean0 = mean(x0)
		self.mean1 = mean(x1)
		S0 = covariance(x0)
		S1 = covariance(x1)
		_, n0 = x0.shape
		_, n1 = x1.shape
		S = ((n0 - 1) * S0 + (n1 - 1) * S1) / (n0 + n1 - 2)
		self.SI = S.I
	def do(self, x):
		d0 = MahalanobisD(x, self.mean0, self.SI)
		d1 = MahalanobisD(x, self.mean1, self.SI)
		return 0 if d0 < d1 else 1
		
class Quadratic(Discriminator):
	def train(self, x0, x1):
		self.mean0 = mean(x0)
		self.mean1 = mean(x1)
		self.SI0 = covariance(x0).I
		self.SI1 = covariance(x1).I
	def do(self, x):
		d0 = MahalanobisD(x, self.mean0, self.SI0)
		d1 = MahalanobisD(x, self.mean1, self.SI1)
		return 0 if d0 < d1 else 1

def crossValidation(x0, x1, discriminator, bin = 1):
	success = 0
	false = 0
	r, c = x0.shape
	for i in range(0, c, bin):
		_x0 = numpy.hstack((x0[:,0:i], x0[:,i+bin:]))
		d = discriminator(_x0, x1)
		for j in range(bin):
			try:
				if d.do(x0[:,i+j]) == 0:
					success += 1
				else:
					false += 1
			except IndexError:
				pass
	r, c = x1.shape
	for i in range(0, c, bin):
		_x1 = numpy.hstack((x1[:,0:i], x1[:,i+bin:]))
		d = discriminator(x0, _x1)
		for j in range(bin):
			try:
				if d.do(x1[:,i+j]) == 1:
					success += 1
				else:
					false += 1
			except IndexError:
				pass
	return success, false

if __name__ == '__main__':
	import csv
	rows = list(csv.reader(open('../csv/train.csv')))
	x, e = M.getColumns(rows, (0, 3, (int, lambda x: 1 if x=='male' else 0)))
	print(M.count(x))
	x, e = M.getColumns(rows, (0, 1, 3, 4, 5, 6), (int, float, lambda x: 1 if x=='male' else 0, float, float, float))
	x0, x1 = split(x, 0)
	fisher = Fisher(x0, x1)
	fisher.do(numpy.matrix((1, 0, 38)).T)
	CV(x0, x1, Fisher)

