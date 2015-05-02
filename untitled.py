from sklearn import svm
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import cross_val_score
from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge
import datetime
import csv
import json    # or `import simplejson as json` if on Python < 2.6
import math   
import pylab as pl
import numpy as np
from sklearn import datasets, linear_model
from sklearn.datasets.samples_generator import make_regression


strmaps = {}

def read_date(s='04/09/2014'):
	date=datetime.datetime.strptime(s, '%m/%d/%Y').date()	
	today=datetime.date.today()
	diff=today-date
	return diff.days


def read_city(s, idx):
	cities = strmaps.get(idx, {})
	l = len(cities)
	if cities.get(s, None) == None:
		cities[s] = l
	strmaps[idx] = cities
	return cities[s]

def isfloat(val):
	if type(val) is int:
		return True
	if val.isdigit():
		return True
	try:
		float(val)
		return True
	except ValueError:
		return False


#below is used for transfering character to number
def read_name_map(filname):
	import csv
	features=[]
	output=[]
	with open(filname) as csvfile:
		reader=csv.reader(csvfile)
		headers = reader.next()
		#print headers
		for row in reader:
			feat = []
			cnt = 0
			for c in row[1:-1]:
				#print c
				cnt = 1 + cnt
				if '/' in c:
					c=read_date(c)
					c=0
				if not isfloat(c):
					#print '~~~~', c, type(c)
					c=read_city(c, cnt)
				feat.append(c)
			features.append(feat)
			output.append(float(row[-1]))
			#print 'row',row
			#print 'feat',feat
	return (features, output)
	

def read_train_csv(filname):
	import csv
	features=[]
	output=[]
	with open(filname) as csvfile:
		reader=csv.reader(csvfile)
		headers = reader.next()
		#print headers
		for row in reader:
			feat = []
			cnt = 0
			for c in row[1:-1]:
				#print c
				cnt = 1 + cnt
				if '/' in c:
					c=read_date(c)
					c=0
				if not isfloat(c):
					#print '~~~~', c, type(c)
					c=read_city(c, cnt)
				feat.append(c)
			features.append(feat)
			output.append(math.log(1+float(row[-1]),20))
			#print 'row',row
			#print 'feat',feat
	return (features, output)

def get_category_arr(col, val):
	# train data has index column, which is removed.
	colmap = strmaps.get(col+1)	
	arr = []
	for i in range(len(colmap)):
		if (i == val):
			arr.append(1)
		else:
			arr.append(0)
	return arr


def write_train_csv(filname, features, output):
	#print "categories", json.dumps(strmaps)
	#print "<<<<<<<<<<<>>>>>>>>>>>>>"

	#print "category_arr", get_category_arr(1, 28)
	#print "category_arr", get_category_arr(2, 0)
	#print "category_arr", get_category_arr(3, 2)

	newfeatures = []
	with open(filname,'wb') as csvfile:
		a=csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
		rowidx = 0
		for line in features:
			row = []
			row.append(line[0])
			row= row + get_category_arr(1,line[1])
			row= row + get_category_arr(2,line[2])
			row= row + get_category_arr(3,line[3])
			row= row + line[4:]
			newfeatures.append(row[:])
			row.append(output[rowidx])
			rowidx += 1 
			#print row
			a.writerow(row)

	return newfeatures, output

	read_name_map('train.csv')

trainfeatures, trainoutput = read_train_csv('train.csv')

trainfeatures, trainoutput = write_train_csv("train.csv", trainfeatures, trainoutput)

regr = linear_model.LinearRegression()

regr.fit(trainfeatures, trainoutput)

print 'Coefficients: \n', regr.coef_

