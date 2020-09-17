from experiment.predictor_baseline import *
from experiment.predictor_advance import *
from experiment.utils import *
from anomaly_test.detector_baseline import *
import os, time


def single_test(Repo, Directory, repeats, goal, tocsv):
    data = data_goal_arrange(Repo, Directory, goal)

    a, p, r, f = [], [], [], []
    for _ in range(repeats):
        temp = CART_anomaly(data)
        a.append(temp[0])
        p.append(temp[1])
        r.append(temp[2])
        f.append(temp[3])

    values = [np.median(a), np.median(p), np.median(r), np.median(f)]
    ans = []
    for value in values:
        ans.append(round(value,2))
    print(ans)
    if tocsv:
        with open("../anomaly_test/results/anomaly_goal{}.csv".format(goal), "a+") as output:
            output.write("%s, " % Repo)
            for i in range(len(ans)):
                if i < len(ans) - 1:
                    output.write(str(ans[i]) + ",")
                else:
                    output.write(str(ans[i]) + "\n")


if __name__ == '__main__':

    path = r'../data/data_cleaned/'
    # repo = "ACE3_monthly.csv"
    repo = "abp_monthly.csv"

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

    Repeats = 1
    Goal = 0

    single_test(repo, path, Repeats, Goal, tocsv=False)

