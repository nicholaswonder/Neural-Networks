# import tensorflow and pandas
import keras as ks
import pandas as pd
import numpy as np

# Get data from training file
fileName = input("Enter filename for training data: ")
data = pd.read_csv(fileName, names=["Fahrenheit", "Kelvin"])

# Clean data to ensure no gaps that could mess with model training
data = data.dropna()

# Separate questions and answers for testing
degreesF = data["Fahrenheit"]
degreesK = data["Kelvin"]

# Get number of epochs for experimentation
epochs = int(input("Input number of training steps: "))

# Declare model with one layer and one neuron
model = ks.Sequential()
model.add(ks.layers.Dense(units=1, input_shape=[1]))

# Compile model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Train the model with given number of training steps
model.fit(x=degreesF, y=degreesK, epochs=epochs, batch_size=1000)

value = 0
while value != -999:
    value = int(input("Enter value to test or -999 to quit: "))
    if value != -999:
        testValue = np.array([value]).reshape(1,1)
        print(model.predict(testValue))