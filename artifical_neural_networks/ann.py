import tensorflow, tensorflow.keras as keras, os, random, numpy as np, pandas as pd, matplotlib.pyplot as plt

#setting seed values to ensure same sequence of random numbers generated on each run
os.environ['PYTHONHASHSEED']='2023'
random.seed(2023) 
np.random.seed(2023)
tensorflow.random.set_seed(2023)

# neural networks implementation for different valus of n
# setup training data
training_data = pd.read_csv("/Users/shruti/Desktop/alda/hw/alda hw4/ann_2023/train_2023.csv")

y_train = np.array(training_data.pop("Class"))
x_train = np.array(training_data)

#setup validation data
validation_data = pd.read_csv("/Users/shruti/Desktop/alda/hw/alda hw4/ann_2023/validation_2023.csv")

y_val = np.array(validation_data.pop("Class"))
x_val = np.array(validation_data)

#setup testing data
testing_data = pd.read_csv("/Users/shruti/Desktop/alda/hw/alda hw4/ann_2023/test_2023.csv")

y_test = np.array(testing_data.pop("Class"))
x_test = np.array(testing_data)

# iterate over a range of possible numbers of hidden neurons
neurons = [4,16,32,64,128]
val_acc = []
train_acc = []

for n in neurons:
    print("Checking for n = ", n)

    #defining the neural network
    model = keras.Sequential()

    #adding the hidden layer
    model.add(keras.layers.Dense(n, activation="relu"))

    #adding output layer with 10 nuerons, corresponding to 10 output class labels
    model.add(keras.layers.Dense(10, activation="softmax"))

    model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    #fit the neural network
    history = model.fit(
    x_train,
    y_train,
    batch_size=10,
    epochs=10,
    validation_data=(x_val, y_val))

    val_acc.append(history.history['val_accuracy'][9])
    train_acc.append(history.history['accuracy'][9])


# plot neurons and accuracy
plt.plot(neurons, val_acc, label='Validation')
plt.plot(neurons, train_acc, label='Training')

plt.title("Hidden neurons vs accuracy")
plt.xlabel("Neurons")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Evaluating the model on test data for n = 32
model = keras.Sequential()
model.add(keras.layers.Dense(32, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(
    x_train,
    y_train,
    batch_size=10,
    epochs=10,
    validation_data=(x_val, y_val))

results = model.evaluate(
    x_test,
    y_test,
    batch_size=10)

print("Results after evaluating the neural network for n=32: ",results)