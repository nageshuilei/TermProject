from sklearn import svm

def svm_train(features,output):
	clf=svm.SVC()
	clf.fit(features,output)
	return clf

def read_train_csv(filname):
	import csv
	features=[]
	output=[]
	with open(filname) as csvfile:
		reader=csv.reader(csvfile)
		for row in reader:
			feat = []
			for c in row[1:]:
				feat.append(ord(c))
			features.append(feat)
			output.append(row[0])
			print row
	return (features, output)



def read_test_csv(filname):
	import csv
	originfeatures = []
	features=[]
	output=[]
	with open(filname) as csvfile:
		reader=csv.reader(csvfile)
		for row in reader:
			originfeatures.append(row)
			feat = []
			for c in row:
				feat.append(ord(c))
			features.append(feat)
			print row
	return (features, originfeatures)


def svm_test(clf,testfeatures):
	predict=clf.predict(testfeatures)
	print predict
	return predict


def rf_train(features,output):
	from sklearn.ensemble import ExtraTreesClassifier
	clf=ExtraTreesClassifier(n_estimators=1000, criterion='entropy', max_depth=None, min_samples_split=1, random_state=0)
	clf.fit(features,output)
	return clf

def rf_test(clf,testfeatures):
	predict=clf.predict(testfeatures)
	#print predict
	return predict



trainfeatures, trainoutput = read_train_csv('train.csv')
#testfeatures, testoutput = read_train_csv('test1.csv')
testfeatures, testoriginfeatures = read_test_csv('test.csv')

print 'length of testfeatures', len(testfeatures)
#clf = svm_train(trainfeatures, trainoutput)
#testpredict = svm_test(clf, testfeatures)
clf = rf_train(trainfeatures, trainoutput)
testpredict = rf_test(clf, testfeatures)
print '----------------------------------'
print 'length of testpredict', len(testpredict)
# print testpredict

"""
print '=================================='
print 'length', len(testoutput)
print testoutput
error = 0

for i in range(len(testpredict)):
	if testpredict[i] != testoutput[i]:
		error += 1
print '++++++++++++++++++++++++++++++++++'
print 'Total ', len(testpredict), 'Error ', error, 'Accuracy', float(len(testpredict)-error)/len(testpredict)
"""
with open("testdatapredict",'wb') as fil:
	lines = []
	for i in range(0, len(testpredict)):
		lin = '' + testpredict[i]+','
		lin += ''.join(testoriginfeatures[i])
		lin += '\n'
		lines.append(lin)
	fil.writelines(lines)


