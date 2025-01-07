import numpy as np
import matplotlib.pyplot as plt

# Step 2: Initialize weights and biases
def initialize_parameters(input_size, hidden_size, output_size):
    weights = {
        "W1": np.random.randn(hidden_size, input_size) * 0.01,
        "b1": np.zeros((hidden_size, 1)),
        "W2": np.random.randn(output_size, hidden_size) * 0.01,
        "b2": np.zeros((output_size, 1)),
    }
    return weights

# Step 3: Implement forward propagation
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward_propagation(X, weights):
    Z1 = np.dot(weights["W1"], X) + weights["b1"]
    A1 = np.tanh(Z1)  # Hidden layer activation
    Z2 = np.dot(weights["W2"], A1) + weights["b2"]
    A2 = sigmoid(Z2)  # Output layer activation
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache

# Step 4: Compute the loss
def compute_loss(y_true, y_pred):
    m = y_true.shape[1]
    loss = -(1 / m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

# Step 5: Implement backward propagation
def backward_propagation(X, y, weights, cache):
    m = X.shape[1]
    A1, A2 = cache["A1"], cache["A2"]
    dZ2 = A2 - y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(weights["W2"].T, dZ2) * (1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return gradients

# Step 6: Update weights
def update_parameters(weights, gradients, learning_rate):
    weights["W1"] -= learning_rate * gradients["dW1"]
    weights["b1"] -= learning_rate * gradients["db1"]
    weights["W2"] -= learning_rate * gradients["dW2"]
    weights["b2"] -= learning_rate * gradients["db2"]
    return weights

# Step 7: Training loop
def train_network(X, y, hidden_size, learning_rate, epochs):
    np.random.seed(42)
    input_size = X.shape[0]
    output_size = 1
    weights = initialize_parameters(input_size, hidden_size, output_size)
    losses = []

    for epoch in range(epochs):
        y_pred, cache = forward_propagation(X, weights)
        loss = compute_loss(y, y_pred)
        losses.append(loss)
        gradients = backward_propagation(X, y, weights, cache)
        weights = update_parameters(weights, gradients, learning_rate)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs} - Loss: {loss:.4f}")

    return weights, losses

# Step 8: Plot decision boundary
def plot_decision_boundary(X, y, weights):
    x_min, x_max = X[0, :].min() - 0.1, X[0, :].max() + 0.1
    y_min, y_max = X[1, :].min() - 0.1, X[1, :].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    grid = np.c_[xx.ravel(), yy.ravel()].T
    _, cache = forward_propagation(grid, weights)
    Z = cache["A2"].reshape(xx.shape)

    plt.contourf(xx, yy, Z > 0.5, alpha=0.6, cmap=plt.cm.coolwarm)
    plt.scatter(X[0, :], X[1, :], c=y.ravel(), cmap=plt.cm.coolwarm, edgecolors="k")
    plt.title("Decision Boundary")
    plt.show()

# Dataset setup
X = np.array([
    [0.1, 0.15, 0.25, 0.35, 0.5, 0.6, 0.65, 0.8],
    [0.6, 0.71, 0.8, 0.45, 0.5, 0.2, 0.3, 0.35]
])
y = np.array([[1, 1, 1, 1, 0, 0, 0, 0]])

# Parameters
hidden_size = 4
learning_rate = 0.1
epochs = 1000

# Train the model
trained_weights, losses = train_network(X, y, hidden_size, learning_rate, epochs)

# Plot loss curve
plt.plot(losses)
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# Visualize decision boundary
plot_decision_boundary(X, y, trained_weights)
