import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import make_classification 
from sklearn.linear_model import Perceptron 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score # Generate synthetic data with two features 
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2, 
                           n_clusters_per_class=1, random_state=42) # Split the data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Create and train the Perceptron model 
perceptron = Perceptron(max_iter=1000, random_state=42) 
perceptron.fit(X_train, y_train) # Make predictions on the test set 
y_pred = perceptron.predict(X_test) # Calculate accuracy 
accuracy = accuracy_score(y_test, y_pred) 
print("Accuracy:", accuracy) # Plot classification results and decision boundary 
x1_min, x1_max = X[:, 0].min()- 1, X[:, 0].max() + 1 
x2_min, x2_max = X[:, 1].min()- 1, X[:, 1].max() + 1 
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.01), np.arange(x2_min, x2_max, 0.01)) 
Z = perceptron.predict(np.c_[xx1.ravel(), xx2.ravel()]) 
Z = Z.reshape(xx1.shape) 
plt.contourf(xx1, xx2, Z, alpha=0.4) 
plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', s=25, edgecolor='k') 
plt.xlabel('Feature 1') 
plt.ylabel('Feature 2') 
plt.title('Perceptron Classification Results') 
plt.show()