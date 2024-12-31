import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data = {
    "X1": [0.1, 1.2, 1.5, 2.0, 2.5, 0.5, 1.8, 0.2, 1.9, 0.8],
    "X2": [1.1, 0.9, 1.6, 1.8, 2.1, 1.5, 2.3, 0.7, 1.4, 0.6],
    "Y":  [0, 0, 1, 1, 1, 0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)
X = df[['X1', 'X2']].values
y = df['Y'].values

def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std

X_std = standardize(X)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-9 
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss
def gradient_descent(X, y, weights, learning_rate, iterations):
    m = X.shape[0]
    for i in range(iterations):
        z = np.dot(X, weights)
        y_pred = sigmoid(z)
        gradient = np.dot(X.T, (y_pred - y)) / m
        weights -= learning_rate * gradient
        
        if i % 100 == 0:
            loss = cross_entropy_loss(y, y_pred)
            print(f"Iteration {i}, Loss: {loss:.4f}")
    return weights

def predict(X, weights, threshold=0.5):
    probabilities = sigmoid(np.dot(X, weights))
    return (probabilities >= threshold).astype(int)

def logistic_regression(X, y, learning_rate=0.01, iterations=1000):
    X_with_bias = np.c_[np.ones((X.shape[0], 1)), X] 
    weights = np.zeros(X_with_bias.shape[1])
    weights = gradient_descent(X_with_bias, y, weights, learning_rate, iterations)
    return weights

def evaluate(y_true, y_pred):
    accuracy = np.mean(y_true == y_pred)
    return accuracy

def plot_decision_boundary(X, y, weights):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y==0, 0], X[y==0, 1], c='red', label='Class 0')
    plt.scatter(X[y==1, 0], X[y==1, 1], c='blue', label='Class 1')
    
    x1 = np.linspace(-2, 2, 100)
    x2 = -(weights[0] + weights[1] * x1) / weights[2]
    plt.plot(x1, x2, c='green', label='Decision Boundary')
    
    plt.xlabel("Feature 1 (Standardized)")
    plt.ylabel("Feature 2 (Standardized)")
    plt.title("Logistic Regression Decision Boundary")
    plt.legend()
    plt.grid()
    plt.show()
learning_rate = 0.1
iterations = 1000
weights = logistic_regression(X_std, y, learning_rate, iterations)
print("\nTrained Weights:", weights)

X_with_bias = np.c_[np.ones((X_std.shape[0], 1)), X_std]
y_pred = predict(X_with_bias, weights)
accuracy = evaluate(y, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
plot_decision_boundary(X_std, y, weights)
