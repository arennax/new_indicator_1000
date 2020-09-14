from experiment.predictor_baseline import *
from experiment.predictor_advance import *
from experiment.utils import *
import os, time


def prediction(Repo, Directory, metrics, repeats, goal, month, tocsv):
    data = data_goal_arrange(Repo, Directory, goal)

    for way in range(6):

        if way == 0:
            list_temp = []
            for _ in range(repeats):
                list_temp.append(KNN(data, month)[metrics])

            flat_list = np.array(list_temp).flatten()
            list_output = sorted(flat_list.tolist())

            print("median for KNN:", np.median(list_output))
            if tocsv:
                with open("../result_experiment/goal{}.csv".format(goal), "a+") as output:
                    output.write("%s, " % Repo)
                    output.write(str(np.median(list_output)) + ",")

        if way == 1:
            list_temp = []
            for _ in range(repeats):
                list_temp.append(SVM(data, month)[metrics])

            flat_list = np.array(list_temp).flatten()
            list_output = sorted(flat_list.tolist())

            print("median for SVR:", np.median(list_output))
            if tocsv:
                with open("../result_experiment/goal{}.csv".format(goal), "a+") as output:
                    output.write(str(np.median(list_output)) + ",")

        if way == 2:
            list_temp = []
            for _ in range(repeats):
                list_temp.append(RFT(data, month)[metrics])

            flat_list = np.array(list_temp).flatten()
            list_output = sorted(flat_list.tolist())

            print("median for RFT:", np.median(list_output))
            if tocsv:
                with open("../result_experiment/goal{}.csv".format(goal), "a+") as output:
                    output.write(str(np.median(list_output)) + ",")

        if way == 3:
            list_temp = []
            for _ in range(repeats):
                list_temp.append(LNR(data, month)[metrics])

            flat_list = np.array(list_temp).flatten()
            list_output = sorted(flat_list.tolist())

            print("median for LNR:", np.median(list_output))
            if tocsv:
                with open("../result_experiment/goal{}.csv".format(goal), "a+") as output:
                    output.write(str(np.median(list_output)) + ",")

        if way == 4:
            list_temp = []
            for _ in range(repeats):
                list_temp.append(CART(data, month)[metrics])

            flat_list = np.array(list_temp).flatten()
            list_output = sorted(flat_list.tolist())

            print("median for CART:", np.median(list_output))
            if tocsv:
                with open("../result_experiment/goal{}.csv".format(goal), "a+") as output:
                    output.write(str(np.median(list_output)) + ",")
                    # output.write(str(np.median(list_output)) + "\n")

        if way == 5:
            list_temp = []
            for _ in range(repeats):
                list_temp.append(DECART(data, metrics, month))

            flat_list = np.array(list_temp).flatten()
            list_output = sorted(flat_list.tolist())

            # print("--------------------------------")
            print("median for DECART:", np.median(list_output))
            if tocsv:
                with open("../result_experiment/goal{}.csv".format(goal), "a+") as output:
                    output.write(str(np.median(list_output)) + "\n")


if __name__ == '__main__':

    # path = r'../data/data_cleaned/'
    # repo = "abp_monthly.csv"

    repo_pool = []
    path = r'../data/data_cleaned/'
    for filename in os.listdir(path):
        if not filename.startswith('.'):
            repo_pool.append(os.path.join(filename))

    Metrics = 0  # "0" for MRE, "1" for SA
    Repeats = 3
    Goal = 0
    month = 1

    # goal == 0: 'number_of_commits'
    # goal == 1: 'number_of_contributors'
    # goal == 2: 'number_of_new_contributors'
    # goal == 3: 'number_of_contributor-domains'
    # goal == 4: 'number_of_new_contributor-domains'
    # goal == 5: 'number_of_open_PRs'
    # goal == 6: 'number_of_closed_PRs'
    # goal == 7: 'number_of_open_issues'
    # goal == 8: 'number_of_closed_issues'
    # goal == 9: 'number_of_stargazers'

    time_begin = time.time()

    number = 0
    for i in range(10):
        for repo in sorted(repo_pool):
            print("-----------------------------------------")
            print(number, repo, "Goal:", i)
            prediction(repo, path, Metrics, Repeats, goal=i, month=1, tocsv=False)
            number += 1

    run_time = str(time.time() - time_begin)
    print(run_time)

