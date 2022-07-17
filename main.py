import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import neighbors
import math


# Linear regression
def create_matrix_linear(x, y):
    matrix = np.ones((x.shape[0], 1))
    matrix_salary = y

    for c in x.columns:
        mtrx = np.append(matrix, x[[c]].to_numpy(), axis=1)
        matrix = mtrx
    return matrix, matrix_salary


def linear_fit(x, y, l):
    # Create matrix with 1 on diagonal
    m = np.zeros((x.shape[1], x.shape[1]), int)
    np.fill_diagonal(m, 1)
    m[0][0] = 0

    XTX = np.dot(x.T, x)
    XTy = np.dot(x.T, y)
    first_param = np.linalg.pinv(XTX + l * m)
    theta = np.dot(first_param, XTy)
    return theta


def predict_linear(df, theta):
    rezz = theta[0]
    i = 1
    for c in df.columns:
        rezz += theta[i] * df[[c]].to_numpy()
        i += 1
    return rezz


# Preprocessing
def preprocess(df):
    df = df.drop("Tm", axis=1)
    df = df.drop("Player", axis=1)
    df = df.drop("Pos", axis=1)
    df = df.drop("Age", axis=1)
    df = df.drop("OWS", axis=1)
    df = df.drop("DWS", axis=1)
    df = df.drop("WS/48", axis=1)
    df = df.drop("OBPM", axis=1)
    df = df.drop("DBPM", axis=1)
    df = df.drop("VORP", axis=1)
    df = df.drop("FG%", axis=1)
    # df = df.drop(
    #     ["3P%", "2P%", "FT%", "ORB", "TRB", "DRB", "STL", "BLK", "TOV", "AST", "G", "GS", "PER", "TS%", "3PAr", "FTr",
    #      "ORB%", "TRB%", "DRB%", "STL%", "BLK%", "TOV%", "USG%", "AST%",
    #      "FG", "FGA", "3P", "3PA", "2P", "2PA", "eFG%", "FT", "FTA"], axis=1)

    # df["Pos"] = df["Pos"].map({'C': 0, 'PF': 1, 'PF-C': 2, 'PG': 3, 'SF': 4, 'SG': 5})
    df = df.iloc[:, 1:]
    df = df.fillna(0)
    return df


# Ridge
def ridge_fit(X, y, alpha):
    x = np.c_[np.ones((X.shape[0], 1)), X]
    A = np.identity(x.shape[1])
    A[0, 0] = 0
    inv = np.linalg.inv(x.T.dot(x) + alpha * A)
    XTy = np.dot(x.T, y)
    theta = np.dot(inv, XTy)
    return theta


def ridge_predict(X, theta):
    return np.c_[np.ones((X.shape[0], 1)), X].dot(theta)


def ridge_alpha(X_train, y_train, X_test, y_test):
    results = []
    for alpha in range(0, 5000):
        alpha = alpha / 100
        theta = ridge_fit(X_train.to_numpy(), y_train, alpha)
        results.append(calculate_rmse(y_test, ridge_predict(X_test.to_numpy(), theta)))
    results.sort()
    return round(results[0], 2)


# Metrics
def calculate_rmse(salary_true, salary_predict):
    return math.sqrt(np.sum((salary_true-salary_predict)*(salary_true-salary_predict), axis=0)/salary_true.shape[0])


if __name__ == '__main__':
    np.random.seed(4)

    df_salary = pd.read_csv("NBA_season1718_salary.csv")
    df_stats = pd.read_csv("Seasons_Stats.csv")

    df_stats = df_stats.dropna(subset=['Year', 'Player'])
    df_stats = df_stats.drop("Tm", axis=1)
    df_stats = df_stats.drop("blanl", axis=1)
    df_stats = df_stats.drop("blank2", axis=1)
    df_stats = df_stats.drop(df_stats[df_stats.Year < 2017].index)
    df_stats = df_stats.iloc[:, 1:]
    # print(df_stats)

    df_salary = df_salary.merge(df_stats, on='Player')
    df_salary = preprocess(df_salary)

    # plt.scatter(df_salary.PTS, df_salary.MP)
    # plt.show()

    X_train, X_test, y_train, y_test = train_test_split(df_salary.loc[:, df_salary.columns != 'season17_18'],
                                                        df_salary['season17_18'].values,
                                                        test_size=0.3, random_state=50)

    # Linear
    # X_train, Y_train = create_matrix_linear(X_train, y_train)
    # theta = linear_fit(X_train, Y_train, 100)
    # rez_validation = predict_linear(X_test, theta)
    # rez_validation = list(map(lambda i: i[0], rez_validation))

    # Ridge
    err = ridge_alpha(X_train, y_train, X_test, y_test)
    print("\nRMSE test: " + str(err))

    # ElasticNet
    # model = linear_model.ElasticNet(alpha=12.0, max_iter=5000)
    # model.fit(X_train, y_train)
    # rez_validation = model.predict(X_test)
    # print("\nRMSE test: " + str(round(calculate_rmse(y_test, rez_validation), 2)))

    # KNN
    # rmse_val = []
    # for K in range(20):
    #     K = K + 1
    #     model = neighbors.KNeighborsRegressor(n_neighbors=K)
    #     model.fit(X_train, y_train)
    #     pred = model.predict(X_test)
    #     error = round(calculate_rmse(y_test, pred), 2)
    #     rmse_val.append(error)
    #     print('RMSE value for k= ', K, 'is:', error)
