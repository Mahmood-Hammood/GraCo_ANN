# Example of neural network in keras
# first neural network with keras tutorial
from numpy import loadtxt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

#load dataset
dataset = loadtxt('dataset.csv', delimiter=',')
#split into inpux(X) and output (y) variables
X = dataset[:,0:16]
Y = dataset[:,-7:]
# define the keras model
model = Sequential()
model.add(Dense(22, input_shape=(16,), activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(7, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, Y, epochs=150, batch_size=10)
# evaluate the keras model
#_, accuracy = model.evaluate(X, Y)
#print('Accuracy: %.2f' % (accuracy*100))
# make class predictions with the model
predictions = (model.predict(X) > 0.5).astype(int)
# summarize the first 5 cases
for i in range(5):
	print('%s => %s (expected %s)' % (X[i].tolist(), predictions[i], Y[i]))



