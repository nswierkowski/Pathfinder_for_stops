import pandas as pd
import matplotlib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib


def create_regression_model_for_time():
    file_path = '../results/dijkstra_results.csv'
    df = pd.read_csv(file_path)

    column1 = 'Distance'
    column2 = 'Change number'

    X = df[['Distance']]
    y = df['Change number']
    model = LinearRegression()
    model.fit(X.values, y)

    residuals = y - model.predict(X)

    measurement_error = np.std(residuals)
    print("Measurement error of linear regression:", measurement_error)

    plt.scatter(X, y, label='Data')
    plt.plot(X, model.predict(X), color='red', label='Linear Regression')
    plt.title('Scatter Plot with Linear Regression')
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.legend()
    plt.grid(True)

    diagram_file = 'scatter_plot_with_linear_regression.png'
    plt.savefig(diagram_file)
    print(f"Scatter plot diagram with linear regression saved to '{diagram_file}'")

    model_file = '../linear_regression_model.pkl'
    joblib.dump(model, model_file)


def create_regression_model():
    file_path = '../results/dijkstra_results.csv'
    df = pd.read_csv(file_path)

    column1 = 'Distance'
    column2 = 'Change number'

    X = df[['Distance']]
    y = df['Change number']
    model = LinearRegression()
    model.fit(X.values, y)

    residuals = y - model.predict(X)

    measurement_error = np.std(residuals)
    print("Measurement error of linear regression:", measurement_error)

    plt.scatter(X, y, label='Data')
    plt.plot(X, model.predict(X), color='red', label='Linear Regression')
    plt.title('Scatter Plot with Linear Regression')
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.legend()
    plt.grid(True)

    diagram_file = 'scatter_plot_with_linear_regression.png'
    plt.savefig(diagram_file)
    print(f"Scatter plot diagram with linear regression saved to '{diagram_file}'")

    model_file = '../linear_regression_model.pkl'
    joblib.dump(model, model_file)


def model_mean_error():
    test_file_path = '../test_data.csv'
    test_df = pd.read_csv(test_file_path)

    model_file_path = '../linear_regression_model.pkl'
    model = joblib.load(model_file_path)

    X_test = test_df[['Distance']]
    y_test = test_df['Change number']

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    absolute_errors = np.abs(y_test - y_pred)

    max_absolute_error = np.max(absolute_errors)
    r2 = r2_score(y_test, y_pred)

    print("Evaluation metrics:")
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"R^2 Score: {r2}")
    print(f"Max absolute error: {max_absolute_error}")

# if __name__ == "__main__":
#     # create_regression_model()
#     # model_mean_error()
