import hw0_solution

from sklearn import datasets, linear_model

diabetes = datasets.load_diabetes()


diabetes_X = diabetes.data
 

diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]
 

diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]
 
model = hw0_solution.LinearRegressor()
 
model.fit(diabetes_X_train, diabetes_y_train, learning_rate=1e-4, n_steps=10000)
diabetes_y_pred = model.predict(diabetes_X_test)

print("Mean squared error: %.2f"
      % hw0_solution.mean_squared_error(diabetes_y_test, diabetes_y_pred))


regr = linear_model.LinearRegression()
 
regr.fit(diabetes_X_train, diabetes_y_train)
 
diabetes_y_pred = regr.predict(diabetes_X_test)

print("Mean squared error: %.2f"
      % hw0_solution.mean_squared_error(diabetes_y_test, diabetes_y_pred))