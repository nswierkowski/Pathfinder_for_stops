import joblib
import numpy as np
import pandas as pd

import matplotlib
from scipy.stats import stats
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import SVR
import statsmodels.api as sm
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


def create_time_regression_model() -> None:
    file_path = '../dijkstra_results.csv'
    df = pd.read_csv(file_path)

    df = df.dropna(subset=['Travel time'])
    data = df.values
    X, y = data[:, 0:1], data[:, 2:3]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    mask = LocalOutlierFactor().fit_predict(X_train) != -1
    X_train, y_train = X_train[mask, :], y_train[mask]

    StandardScaler().fit(X_train)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_model = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_model)
    mse = mean_squared_error(y_test, y_model)
    r2 = r2_score(y_test, y_model)
    print(F"MAE: {mae}")
    print(F"MSE: {mse}")
    print(F"R2: {r2}")

    max_absolute_error = np.max(np.abs(y_test - y_model))
    print(F"Max absolute error: {max_absolute_error}")

    plt.scatter(X, y, label='Data')
    plt.plot(X, model.predict(X), color='red', label='Linear Regression')
    plt.title('Time Linear Regression')
    plt.xlabel("Manhattan distance")
    plt.ylabel("Travel time (in seconds)")
    plt.legend()
    plt.grid(True)

    diagram_file = 'time_linear_regression.png'
    plt.savefig(diagram_file)
    print(f"Scatter plot diagram with linear regression saved to '{diagram_file}'")

    model_file = '../linear_regression_time_model.pkl'
    joblib.dump(model, model_file)


def create_time_regression_model_2() -> None:
    file_path = '../dijkstra_results.csv'
    df = pd.read_csv(file_path)

    df = df.dropna(subset=['Travel time'])
    X, y = df[['Distance', 'Starting hour']], df['Travel time']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    StandardScaler().fit(X_train)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_model = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_model)
    r2 = r2_score(y_test, y_model)
    print(F"MAE: {mae}")
    print(F"R2: {r2}")

    plt.scatter(X['Distance'], y, label='Data')
    plt.plot(X['Distance'], model.predict(X), color='red', label='Linear Regression')
    plt.title('Time Linear Regression')
    plt.xlabel("Manhattan distance")
    plt.ylabel("Travel time (in seconds)")
    plt.legend()
    plt.grid(True)

    diagram_file = 'time_linear_regression_based.png'
    plt.savefig(diagram_file)
    print(f"Scatter plot diagram with linear regression saved to '{diagram_file}'")

    model_file = '../linear_regression_time_model_based.pkl'
    joblib.dump(model, model_file)


def create_time_svr() -> None:
    file_path = '../dijkstra_results.csv'
    df = pd.read_csv(file_path)

    df = df.dropna(subset=['Travel time'])

    X, y = df[['Distance', 'Starting hour']], df['Travel time']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    StandardScaler().fit(X_train)

    model = SVR()
    model.fit(X_train, y_train)

    y_model = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_model)
    r2 = r2_score(y_test, y_model)
    print(F"MAE: {mae}")
    print(F"R2: {r2}")

    plt.scatter(X['Distance'], y, label='Data')
    plt.plot(X['Distance'], model.predict(X), color='red', label='Linear Regression')
    plt.title('Time Linear Regression')
    plt.xlabel("Manhattan distance")
    plt.ylabel("Travel time (in seconds)")
    plt.legend()
    plt.grid(True)

    diagram_file = 'time_svr.png'
    plt.savefig(diagram_file)
    model_file = '../time_svr.pkl'
    joblib.dump(model, model_file)


if __name__=="__main__":
    create_time_regression_model()
    #create_time_regression_model_2()
    #create_time_svr()