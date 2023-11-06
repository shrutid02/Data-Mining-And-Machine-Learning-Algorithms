import pandas as pd, sklearn, sklearn.model_selection as ms, matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# load data and report Class 1 and Class 0 size
svm_data = pd.read_csv("svm_data_2023.csv")

Y = svm_data.iloc[:,60]
X = svm_data.iloc[:,0:59]
print("Initial data:", Y.value_counts())

# Use stratified random sampling to divide the dataset
X_train, X_test, Y_train, Y_test = ms.train_test_split(X, Y, random_state = 2023)
print("Training data (75%):",Y_train.value_counts())
print("Testing data (25%):",Y_test.value_counts())

# Declare array to store support vectors
train_data = pd.read_csv("train_data_2023.csv")

Y_train = train_data.iloc[:,10]
X_train = train_data.iloc[:,0:9]

support_vectors = []

# Train the model for different values of C
reg_param = [0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10]

for c in reg_param:
    svc_model = SVC(kernel = 'linear', C=c)
    svc_model.fit(X_train, Y_train)
    
    total_vectors = sum(svc_model.n_support_)
    support_vectors.append(total_vectors)
    
# Plot C vs Number of Support vectors
plt.plot(reg_param, support_vectors)
plt.xlabel("Regularization parameter (C)")
plt.ylabel("Number of Support Vectors")
print(support_vectors)


# setup test data
test_data = pd.read_csv("test_data_2023.csv")

Y_test = test_data.iloc[:,10]
X_test = test_data.iloc[:,0:9]


C = [0.1, 0.2, 0.3, 1, 5, 10, 20, 100, 200, 1000]
degree = [1, 2, 3, 4, 5]
coef0 = [0.0001, 0.001, 0.002, 0.01, 0.02, 0.1, 0.2, 0.3, 1, 2, 5, 10]
gamma = [0.0001,0.001, 0.002, 0.01, 0.02, 0.03, 0.1, 0.2, 1, 2, 3]

# for linear kernel
param_grid = {'C':C}
grid = GridSearchCV(SVC(kernel = 'linear'), param_grid)
grid.fit(X_train, Y_train)

print("Best params for linear kernel: ", grid.best_params_) 
grid_predictions = grid.predict(X_test) 
print(classification_report(Y_test, grid_predictions)) 

# for polynomial kernel
param_grid = {'C':C, 'degree':degree, 'coef0':coef0}
grid = GridSearchCV(SVC(kernel = 'poly'), param_grid)
grid.fit(X_train, Y_train)

print("Best params for polynomial kernel: ", grid.best_params_) 
grid_predictions = grid.predict(X_test) 
print(classification_report(Y_test, grid_predictions)) 

# for RBF kernel
param_grid = {'C':C, 'gamma':gamma}
grid = GridSearchCV(SVC(kernel = 'rbf'), param_grid)
grid.fit(X_train, Y_train)

print("Best params for rbf kernel: ", grid.best_params_) 
grid_predictions = grid.predict(X_test) 
print(classification_report(Y_test, grid_predictions)) 

# for Sigmoid kernel
param_grid = {'C':C, 'coef0':coef0,'gamma':gamma}
grid = GridSearchCV(SVC(kernel = 'sigmoid'), param_grid)
grid.fit(X_train, Y_train)

print("Best params for polynomial kernel: ", grid.best_params_) 
grid_predictions = grid.predict(X_test) 
print(classification_report(Y_test, grid_predictions)) 