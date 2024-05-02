import joblib
import numpy as np
import pandas as pd
import matplotlib
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import SVR

matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler


def create_change_regression_model() -> None:
    file_path = '../results/dijkstra_results.csv'
    df = pd.read_csv(file_path)

    data = df.values
    X, y = data[:, 0:1], data[:, 1:2]

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
    print(F"R2: {r2}")
    print(F"MSE: {mse}")
    max_absolute_error = np.max(np.abs(y_test - y_model))
    print(f"Max_absolute_error {max_absolute_error}")

    plt.scatter(X, y, label='Data')
    plt.plot(X, model.predict(X), color='red', label='Linear Regression')
    plt.title('Change Linear Regression')
    plt.xlabel("Manhattan distance")
    plt.ylabel("Change number")
    plt.legend()
    plt.grid(True)

    diagram_file = 'change_linear_regression.png'
    plt.savefig(diagram_file)

    model_file = '../linear_regression_change_model.pkl'
    joblib.dump(model, model_file)


if __name__=="__main__":
    create_change_regression_model()
    #create_time_regression_model_2()
    #create_time_svr()