from regression import KNNRegressor
from classification import KNNClassifier
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from utils import mean_squared_error, r2_error, calculate_accuracy

print("Regression:")
breast_cancer_data = load_breast_cancer()
X, y = breast_cancer_data.data, breast_cancer_data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

regressor = KNNRegressor(k=5)
regressor.fit(X_train, y_train)
y_predicted = regressor.predict(X_test)

mse = mean_squared_error(y_predicted, y_test)
r2 = r2_error(y_predicted, y_test)

print("Mean Squared Error:", mse)
print("R-Squared Error", r2)

print("Classification:")
iris_data = load_iris()
X, y = iris_data.data, iris_data.target

classifier = KNNClassifier(k=3)
classifier.fit(X_train, y_train)
y_predicted = classifier.predict(X_test)

accuracy = calculate_accuracy(y_predicted, y_test)

print("Accuracy:", accuracy)