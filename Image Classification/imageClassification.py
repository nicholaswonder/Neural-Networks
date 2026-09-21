# Import libraries
import keras
import keras as ks
import numpy as np
import matplotlib.pyplot as plt

# Get the fashion data from fashion_mnist dataset
fashionData = ks.datasets.fashion_mnist
print("Got Data!")

# Load data into training and testing arrays
(train_images, train_labels), (test_images, test_labels) = fashionData.load_data()
print("Loaded Data!")

# Define the different types of clothing
clothingTypes = ['t-shirt/top', 'trousers', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'boot']

# Preprocess the data so that the data can be efficiently fed into the neural network
train_images = train_images / 255
test_images = test_images / 255
print("Preprocessed Data!")

# Build the model
model = ks.Sequential([
    ks.layers.Flatten(input_shape=(28,28)), # Each image is 28x28 pixels
    ks.layers.Dense(128, activation='relu'), # Creates 128 nodes
    ks.layers.Dense(10)
])
print("Model Built!")

# Compile the model - this is where the optimizer, loss function, and the metrics of which the model is tested
model.compile(optimizer='adam',
              loss=ks.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
print("Model Compiled!")

# Begin training the model
model.fit(train_images, train_labels, epochs=10)
print("Model Trained!")

# Test model for accuracy
testLoss, testAcc = model.evaluate(test_images, test_labels, verbose=2)
print("Model Tested!")

# Show predictions for the test cases
# Create a probability model to make it easier to interpret the results
predictModel = keras.Sequential([model,
                                 keras.layers.Softmax()])

# Store the predictions of the new model
predictions = predictModel.predict(test_images)

# Create function to plot images
plt.grid = False
def plotImage(i, predictArray, trueLabel, img):
    trueLabel, img = trueLabel[i], img[i]
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    # predictArray returns an array of the models confidence in each label
    # Get the label the model is most confident in
    predictedLabel = np.argmax(predictArray)
    if predictedLabel == trueLabel:
        color = 'green'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(clothingTypes[predictedLabel],
                                         100*np.max(predictArray),
                                         clothingTypes[trueLabel]),
                                         color=color)
    # END FUNCTION

# Define function to plot the value array
def plotValueArray(i, predictArray, trueLabel):
    trueLabel = trueLabel[i]
    plt.xticks(range(10))
    plt.yticks([])
    plot = plt.bar(range(10), predictArray, color="#777777")
    plt.ylim([0, 1])
    predictedLabel = np.argmax(predictArray)

    plot[predictedLabel].set_color('red')
    plot[trueLabel].set_color('green')
    # END FUNCTION

# Create a show of the test images and the models predictions
rows = 5
cols = 3
images = rows*cols
plt.figure(figsize=(2*2*cols, 2*rows))
for i in range(images):
    plt.subplot(rows, 2*cols, 2*i+1)
    plotImage(i, predictions[i], test_labels, test_images)
    plt.subplot(rows, 2*cols, 2*i+2)
    plotValueArray(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()