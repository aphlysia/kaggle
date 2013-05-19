#!/usr/bin/env python
import numpy as np
import math
import random

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

def split(x, y, isClass0):
	'''
	split x into class0 and class1
	'''
	x = np.matrix(x)
	r, c = x.shape
	x0 = None
	x1 = None
	for i in range(c):
		if isClass0(y[0,i]):
			x0 = x[:,i] if x0 is None else np.hstack((x0, x[:,i]))
		else:
			x1 = x[:,i] if x1 is None else np.hstack((x1, x[:,i]))
	return x0, x1

def mean(x):
	'''
	x[i,j] is the i-th elements of the x at the j trial
	'''
	_x = []
	r, c = x.shape
	for i in range(r):
		_x.append(x[i].mean())
	return np.matrix(_x, float).T

def covariance(x):
	'''
	x[i,j] is the i-th elements of the x at the j trial
	'''
	a = np.matrix(x, float)
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

class Gauss(Discriminator):
	def train(self, x0, x1):
		self.mean0 = mean(x0)
		self.mean1 = mean(x1)
		S0 = covariance(x0)
		self.SI0 = S0.I
		self.d0 = np.linalg.det(S0) ** 0.5
		S1 = covariance(x1)
		self.SI1 = S1.I
		self.d1 = np.linalg.det(S1) ** 0.5
		dim, _ = x0.shape
		self.k = (2. * math.pi)**(-dim / 2.)
	def do(self, x):
		d0 = MahalanobisD(x, self.mean0, self.SI0)
		d1 = MahalanobisD(x, self.mean1, self.SI1)
		p0 = self.k * math.exp(-d0 / 0.5) / self.d0
		p1 = self.k * math.exp(-d1 / 0.5) / self.d1
		return 0 if p0 > p1 else 1

class BayseGauss(Discriminator):
	def train(self, x0, x1):
		self.mean0 = mean(x0)
		self.mean1 = mean(x1)
		S0 = covariance(x0)
		self.SI0 = S0.I
		self.d0 = np.linalg.det(S0) ** 0.5
		S1 = covariance(x1)
		self.SI1 = S1.I
		self.d1 = np.linalg.det(S1) ** 0.5
		dim, _ = x0.shape
		self.k = (2. * math.pi)**(-dim / 2.)
		_, c0 = x0.shape
		_, c1 = x1.shape
		self.p0 = float(c0) / (c0 + c1)
		self.p1 = 1. - self.p0
	def do(self, x):
		d0 = MahalanobisD(x, self.mean0, self.SI0)
		d1 = MahalanobisD(x, self.mean1, self.SI1)
		p0 = self.k * math.exp(-d0 / 0.5) / self.d0 * self.p0
		p1 = self.k * math.exp(-d1 / 0.5) / self.d1 * self.p1
		return 0 if p0 > p1 else 1

def crossValidation(x0, x1, discriminator, bin = 1):
	success = 0
	failure = 0
	r, c0 = x0.shape
	r, c1 = x1.shape
	l = [0] * c0 + [1] * c1
	random.shuffle(l)
	l0 = 0
	l1 = 0
	for i in range(0, c0 + c1, bin):
		r0 = l0 + l[i:i+bin].count(0)
		r1 = l1 + l[i:i+bin].count(1)
		_x0 = np.hstack((x0[:,:l0], x0[:,r0:]))
		_x1 = np.hstack((x1[:,:l1], x1[:,r1:]))
		#_x0 = x0[:,120:139]
		#_x1 = x1[:,120:139]
		d = discriminator(_x0, _x1)
		for j in range(l0, r0):
			if d.do(x0[:,j]) == 0:
				success += 1
			else:
				failure += 1
		for j in range(l1, r1):
			if d.do(x1[:,j]) == 1:
				success += 1
			else:
				failure += 1
		l0 = r0
		l1 = r1
	return success, failure 

