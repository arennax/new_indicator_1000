import pandas as pd
import numpy as np
from statistics import stdev, median, mean
from experiment.utils import *
from scipy.io.arff import loadarff
import os


repo_pool = []
path = r'../release_based_test/release_based_data/'


def month_counter():
    i = 0
    for filename in os.listdir(path):
        repo_pool.append(os.path.join(filename))
    for repo in repo_pool:
        df_raw = pd.read_csv(path + repo, sep=',')
        i += len(df_raw.index)
    return i


def data_stats(path):
    for filename in os.listdir(path):
        if not filename.startswith('.'):
            repo_pool.append(os.path.join(filename))
    data_sample = pd.read_csv(path + repo_pool[0], sep=',').drop(columns=['release', 'tag_name', 'commit_id', 'date'])
    n_column = len(data_sample.columns)

    for feature in range(n_column):

        feature_name = data_sample.columns.values[feature]
        stats_list = []

        for repo in repo_pool:
            df_raw = pd.read_csv(path + repo, sep=',').drop(columns=['release', 'tag_name', 'commit_id', 'date'])
            for i in range(len(df_raw)):
                stats_list.append(df_raw.iloc[i, feature])

        mins = min(stats_list)
        maxs = max(stats_list)
        means = mean(stats_list)
        medians = median(stats_list)
        stds = round(stdev(stats_list), 2)
        iqrs = int(np.subtract(*np.percentile(stats_list, [75, 25])))

        temp_list = [mins, maxs, means, medians, stds, iqrs]

        with open("../release_based_test/stats_released_based_data.csv", "a+") as output:
            for i in range(len(temp_list)):
                if i == 0:
                    output.write("%s, " % feature_name)
                    output.write(str(temp_list[i]) + ",")
                elif i == len(temp_list) - 1:
                    output.write(str(temp_list[i]) + "\n")
                else:
                    output.write(str(temp_list[i]) + ",")


if __name__ == '__main__':
    data_stats(path)