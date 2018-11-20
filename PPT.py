import numpy as np
import matplotlib.pyplot as plt
import sys


# ------------------------------------------- #
# ---------------- VARIABLES ---------------- #
# ------------------------------------------- #

# Numpy seed
np.random.seed(0)

# Training loop
learning_rate = 0.1
costs = []

# Number of trainings
train_time = 100000

# Number of tests
test_time = 100000


# ------------------------------------------- #
# ---------------- FUNCTIONS ---------------- #
# ------------------------------------------- #

# Sigmoid functions
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_p(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Create random dataset for training and testing the Perceptron
def random_data(loc, scale, size, group):
    dataset = np.empty((size, 3))
    dataset[:,:2] = np.random.normal(loc, scale, (size, 2))
    dataset[:,2] = group
    return dataset

# Estimate Perceptron with stochastic gradient descent
def train(data):
    # Random weight generation
    weight_1, weight_2, bias = np.random.randn(), np.random.randn(), np.random.randn()

    for i in range(train_time):
        random_i = np.random.randint(len(data))
        point = data[random_i]

        activation = point[0] * weight_1 + point[1] * weight_2 + bias
        prediction = sigmoid(activation)
        target = point[2]

        # Cost function
        cost = np.square(prediction - target)

        # Derivatives
        dcost_prediction = 2 * (prediction - target)
        dprediction_activation = sigmoid_p(activation)
        dactivation_w1 = point[0]
        dactivation_w2 = point[1]
        dactivation_b = 1

        # Slope of cost function
        dcost_activation = dcost_prediction * dprediction_activation

        # Change weight and bias
        weight_1 = weight_1 - (learning_rate * dcost_activation * dactivation_w1)
        weight_2 = weight_2 - (learning_rate * dcost_activation * dactivation_w2)
        bias = bias - (learning_rate * dcost_activation * dactivation_b)

    return [weight_1, weight_2, bias]


# Prediction with weights
def predict(data, weights):
    score = 0 # Store the number of time the Perceptron predicted the correct output
    for i in range(test_time):
        # Select a random point into the dataset
        random_i = np.random.randint(len(data))
        point = data[random_i]

        # Compute the activation of the selected point using the weights obtained and define the prediction made
        activation = point[0] * weights[0] + point[1] * weights[1] + weights[2]
        prediction = sigmoid(activation)

        # Compare the prediction with the target output
        target = point[2]
        if (target == 1 and prediction >= 0.5) or (target == 0 and prediction < 0.5):
            score += 1
    # Print out the mean accuracy from score
    score = 1.0*score/test_time
    print("Mean accuracy: " + '{percent:.2%}'.format(percent=score))



# ------------------------------------------- #
# ----------------- TESTING ----------------- #
# ------------------------------------------- #

# Generate training data
loc1 = 1
loc2 = 3
scale = 0.5
size = 5000
training_data_1 = random_data(loc1, scale, size, 0)
training_data_2 = random_data(loc2, scale, size, 1)
training_data = np.concatenate([training_data_1, training_data_2], axis = 0)

# Print training_data
#plt.scatter(training_data_1[:, 0],training_data_1[:, 1], color = 'red', alpha = 0.1, label = "Group 1")
#plt.scatter(training_data_2[:, 0], training_data_2[:, 1], color = 'blue', alpha = 0.1, label = "Group 2")
#plt.show()

# Generate test data
test_data_1 = random_data(loc1, scale, size, 0)
test_data_2 = random_data(loc2, scale, size, 1)
test_data = np.concatenate([test_data_1, test_data_2], axis = 0)

weights = train(training_data)
predict(test_data, weights)

# Compute weight function
x1 = -1
y1 = (1 - weights[2] - weights[0] * x1) / weights[1]
x2 = 5
y2 = (1 - weights[2] - weights[0] * x2) / weights[1]


# Print test_data
plt.scatter(test_data_1[:, 0],training_data_1[:, 1], color = 'red', alpha = 0.1, label = "Group 0")
plt.scatter(test_data_2[:, 0], training_data_2[:, 1], color = 'blue', alpha = 0.1, label = "Group 1")
plt.plot([x1, x2], [y1, y2], color = 'green')
plt.show()
