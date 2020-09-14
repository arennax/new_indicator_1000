from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn import preprocessing
import pandas as pd
import numpy as np


def df_split(df, month):
    trainData = df.iloc[:-month]
    testData = df.iloc[-1:]
    yield trainData, testData


def df_split_sd(df):

    last_col_original = df.iloc[:, -1]
    df_new = df.copy()
    for i in range(len(df)):
        temp = last_col_original.iloc[:i+1]
        df_new.iloc[i, -1] = np.std(temp)
    sd_column = df_new.iloc[:,-1]
    sd_gain_column = sd_column.copy()
    for i in range(len(sd_column)-1):
        if sd_column[i] == 0:
            sd_gain_column[i+1] = 0
        else:
            sd_gain_column[i + 1] = (sd_column[i+1] - sd_column[i]) / sd_column[i]
    month = abs(sd_gain_column[12:]).idxmax()
    # with open("./spike_6month.csv", "a+") as output:
    #     output.write(str(month+1) + ",")
    trainData = df_new.iloc[:month-6]
    testData = df_new.iloc[month:month+1]
    # pd.set_option("max_columns", 3)
    # print("trainData: ", trainData)
    # print("testData: ", testData)
    yield trainData, testData


def normalize(df):
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(df)
    df_normalized = pd.DataFrame(np_scaled, columns=df.columns, index=df.index)
    lst_col = df.columns[-1]
    df_normalized[lst_col] = df[lst_col]
    return df_normalized


def mre_calc(y_predict, y_actual):
    mre = []
    for predict, actual in zip(y_predict, y_actual):
        if actual == 0:
            if predict == 0:
                mre.append(0)
            elif abs(predict) <= 1:
                mre.append(1)
            else:
                mre.append(round(abs(predict - actual)+1 / (actual+1), 3))
        else:
            mre.append(round(abs(predict - actual) / (actual), 3))
    mMRE = np.median(mre)
    return mMRE


def sa_calc(Y_predict, Y_actual, X_actual):
    Absolute_Error = 0
    for predict, actual in zip(Y_predict, Y_actual):
        Absolute_Error += abs(predict - actual)
    Mean_Absolute_Error = Absolute_Error / (len(Y_predict))
    random_guess = np.mean(X_actual)
    AE_random_guess = 0
    for predict in Y_predict:
        AE_random_guess += abs(predict - random_guess)
    MAE_random_guess = AE_random_guess / (len(Y_predict))
    if MAE_random_guess == 0:
        sa_error = round((1 - (Mean_Absolute_Error+1) / (MAE_random_guess+1)), 3)
    else:
        sa_error = round((1 - Mean_Absolute_Error / MAE_random_guess), 3)
    return sa_error


def data_goal_arrange(repo_name, directory, goal):
    df_raw = pd.read_csv(directory + repo_name, sep=',')
    df_raw = df_raw.drop(columns=['dates'])
    last_col = ''
    if goal == 0:
        last_col = 'number_of_commits'
    elif goal == 1:
        last_col = 'number_of_contributors'
    elif goal == 2:
        last_col = 'number_of_new_contributors'
    elif goal == 3:
        last_col = 'number_of_contributor-domains'
    elif goal == 4:
        last_col = 'number_of_new_contributor-domains'
    elif goal == 5:
        last_col = 'number_of_open_PRs'
    elif goal == 6:
        last_col = 'number_of_closed_PRs'
    elif goal == 7:
        last_col = 'number_of_open_issues'
    elif goal == 8:
        last_col = 'number_of_closed_issues'
    elif goal == 9:
        last_col = 'number_of_stargazers'

    cols = list(df_raw.columns.values)
    cols.pop(cols.index(last_col))
    df_adjust = df_raw[cols+[last_col]]

    return df_adjust


if __name__ == '__main__':

    # path = r'../data/data_cleaned/'
    # repo = "abp_monthly.csv"
    #
    # dataset = data_goal_arrange(repo, path, 9)
    # print(len(dataset.columns))

    # for train, test in df_split(dataset, 1):
    #     print(train)
    #     print(test)
    #     for train, test in df_split(train, 1):
    #         print(train)
    #         print(test)

    temp = [[1.1], [2.3], [3.5]]
    print(np.around(temp))
    print(np.rint(temp).astype(np.int))
