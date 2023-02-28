# PDP2023 - GraCo Neural Network
import pandas
import numpy
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from tensorflow.keras.optimizers import SGD

#load dataset
dataset = loadtxt('dataset.csv', delimiter=',')
X_train = dataset[:,0:16]
Y_train = dataset[:,-7:]

#load dataval
dataset = loadtxt('dataval.csv', delimiter=',')
X_test = dataset[:,0:16]
 
# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(32, input_dim=16, activation='relu'))
	model.add(Dense(7, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.02, momentum=0.7), metrics=['accuracy'])
	return model
 
estimator = KerasClassifier(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
#kfold = KFold(n_splits=10, shuffle=True)
#results = cross_val_score(estimator, X_train, Y_train, cv=kfold)
#print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
estimator.fit(X_train, Y_train)
prediction = estimator.model.predict(X_test)
print(prediction)
prediction = pandas.DataFrame(prediction).to_csv('dataout.csv', float_format='%.9f')