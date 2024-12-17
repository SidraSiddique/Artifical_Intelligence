import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = - np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

def gradient_descent(X, y, weights, learning_rate, iterations):
    m = len(y)
    for i in range(iterations):
        z = np.dot(X, weights)
        y_pred = sigmoid(z)
        gradient = (1/m) * np.dot(X.T, (y_pred - y))
        weights -= learning_rate * gradient
        if i % 100 == 0:
            loss = cross_entropy_loss(y, y_pred)
            print(f"Iteration {i}, Loss: {loss}")
    return weights

def predict(X, weights):
    z = np.dot(X, weights)
    y_pred = sigmoid(z)
    return (y_pred >= 0.5).astype(int)

def logistic_regression(X, y, learning_rate=0.01, iterations=1000):
    m, n = X.shape
    weights = np.zeros(n)
    weights = gradient_descent(X, y, weights, learning_rate, iterations)
    return weights

def evaluate(y_true, y_pred):
    accuracy = np.mean(y_true == y_pred)
    loss = cross_entropy_loss(y_true, y_pred)
    return accuracy, loss

if __name__ == "__main__":
    X = np.array([[0.1, 1.1], [1.2, 0.9], [1.5, 1.6], [2.0, 1.8], [2.5, 2.1],
                  [0.5, 1.5], [1.8, 2.3], [0.2, 0.7], [1.9, 1.4], [0.8, 0.6]])
    y = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 0])

    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)

    plt.scatter(X_standardized[y == 0][:, 0], X_standardized[y == 0][:, 1], color='red', label='Class 0')
    plt.scatter(X_standardized[y == 1][:, 0], X_standardized[y == 1][:, 1], color='blue', label='Class 1')
    plt.xlabel('Feature 1 (X1)')
    plt.ylabel('Feature 2 (X2)')
    plt.legend()
    plt.title('Data Points Visualization')
    plt.show()

    weights = logistic_regression(X_standardized, y, learning_rate=0.01, iterations=1000)

    y_pred = predict(X_standardized, weights)

    accuracy, loss = evaluate(y, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Cross-Entropy Loss: {loss:.4f}")

    x_min, x_max = X_standardized[:, 0].min() - 1, X_standardized[:, 0].max() + 1
    y_min, y_max = X_standardized[:, 1].min() - 1, X_standardized[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_points_standardized = scaler.transform(grid_points)
    Z = predict(grid_points_standardized, weights)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.75, cmap=plt.cm.RdBu)
    plt.scatter(X_standardized[:, 0], X_standardized[:, 1], c=y, cmap=plt.cm.RdBu, edgecolors='k', marker='o')
    plt.xlabel('Feature 1 (X1)')
    plt.ylabel('Feature 2 (X2)')
    plt.title('Decision Boundary')
    plt.show()
