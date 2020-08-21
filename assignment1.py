import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

def checkAccuracy(set, pct):
	labels = []
	predicts = []
	for person in set:
		labels.append(person.getLabel())
		predicts.append(pct.predict(person))

	return metrics.accuracy_score(labels, predicts)

def sign(n):
	if (n >= 0):
		return 1
	else:
		return -1
	

class Perceptron():
	"""docstring for Perceptron"""
	weights = []
	lr = 0.1
	his = []
	def __init__(self):
		self.weights = np.random.rand(9)

	def predict(self, person):
		sum = 0
		data = person.getData()
		for i in range(len(data)):
			sum = sum + data[i] * self.weights[i]

		sum = sum + self.weights[-1]
		return sign(sum)

	def train(self, people):
		hien = 0
		accuracies = []
		for person in people:

			data = person.getData()
			label = person.getLabel()

			prediction = self.predict(person)
			error = label - prediction

			#adjust weights
			for i in range((len(data))):
				self.weights[i] = self.weights[i] + self.lr * error * data[i]
			#adjust bias
			self.weights[-1] = self.weights[-1] + self.lr * error


			accuracies.append(checkAccuracy(people, self))
		self.his = accuracies

class Person():
	"""docstring for Point"""
	personal_data = []
	label = 0
	def __init__(self, personal_data, label):
		super(Person, self).__init__()
		self.personal_data = personal_data
		self.label = label
	def showData(self):
		print(self.personal_data, self.label)

	def getData(self):
		return self.personal_data

	def getLabel(self):
		return self.label


file_path = '/home/quanghien/Documents/assignment/diabetes_scale.txt'

people = []
with open(file_path, 'r') as reader:
	data_row = reader.readline()
	cnt = 1

	while data_row:

		#label value is from the begining to the 1st space
		first_space_index = data_row.find(' ')
		label = data_row[0:first_space_index]
		label = int(label)

		#getting value: value is between a colon and a space
		data = data_row[first_space_index+1:]
		colon_index = data.find(':')
		values = []

		while colon_index > 0:
			next_space_index = data.find(' ')
			value = data[colon_index+1 : next_space_index]
			
			values.append(float(value))

			data = data[next_space_index+1:]
			colon_index = data.find(':')

		people.append(Person(values, label))

		data_row = reader.readline()
		cnt += 1


# create and training the perceptron
p = Perceptron()


train_set, test_set = train_test_split(people, test_size = 0.25)


p.train(train_set)

print(checkAccuracy(train_set,p))
print(checkAccuracy(test_set,p))

plt.plot(p.his)
plt.show()