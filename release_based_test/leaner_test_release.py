# from experiment.learner_untuned import *
# from experiment.learner_tuned import *
# from data.data_ready import *
# import os, time
#
#
# def data_github_release(repo_name, directory, goal):
#     df_raw = pd.read_csv(directory + repo_name, sep=',')
#     df_raw = df_raw.drop(columns=['date', 'release', 'tag_name', 'commit_id'])
#     last_col = ''
#     if goal == 0:
#         last_col = 'number_of_contributors'
#     elif goal == 1:
#         last_col = 'number_of_new_contributors'
#     elif goal == 2:
#         last_col = 'number_of_contributor-domains'
#     elif goal == 3:
#         last_col = 'number_of_new_contributor-domains'
#
#     cols = list(df_raw.columns.values)
#     cols.pop(cols.index(last_col))
#     df_adjust = df_raw[cols+[last_col]]
#
#     return df_adjust
#
#
# def release_learner_check(Repo, Directory, metrics, repeats, goal, month, tocsv):
#     data = data_github_release(Repo, Directory, goal)
#
#     for way in range(6):
#
#         if way == 0:
#             list_temp = []
#             for way in range(repeats):
#                 list_temp.append(KNN(data, month)[metrics])
#
#             flat_list = np.array(list_temp).flatten()
#             list_output = sorted(flat_list.tolist())
#
#             print("median for KNN:", np.median(list_output))
#
#             if tocsv:
#                 with open("./release_goal{}.csv".format(goal), "a+") as output:
#                     output.write("%s, " % Repo)
#                     output.write(str(np.median(list_output)) + ",")
#
#         if way == 1:
#             list_temp = []
#             for way in range(repeats):
#                 list_temp.append(SVM(data, month)[metrics])
#
#             flat_list = np.array(list_temp).flatten()
#             list_output = sorted(flat_list.tolist())
#
#             print("median for SVR:", np.median(list_output))
#
#             if tocsv:
#                 with open("./release_goal{}.csv".format(goal), "a+") as output:
#                     output.write(str(np.median(list_output)) + ",")
#
#         if way == 2:
#             list_temp = []
#             for way in range(repeats):
#                 list_temp.append(RFT(data, month)[metrics])
#
#             flat_list = np.array(list_temp).flatten()
#             list_output = sorted(flat_list.tolist())
#
#             print("median for RFT:", np.median(list_output))
#
#             if tocsv:
#                 with open("./release_goal{}.csv".format(goal), "a+") as output:
#                     output.write(str(np.median(list_output)) + ",")
#
#         if way == 3:
#             list_temp = []
#             for way in range(repeats):
#                 list_temp.append(LNR(data, month)[metrics])
#
#             flat_list = np.array(list_temp).flatten()
#             list_output = sorted(flat_list.tolist())
#
#             print("median for LNR:", np.median(list_output))
#
#             if tocsv:
#                 with open("./release_goal{}.csv".format(goal), "a+") as output:
#                     output.write(str(np.median(list_output)) + ",")
#
#         if way == 4:
#             list_temp = []
#             for way in range(repeats):
#                 list_temp.append(CART(data, month)[metrics])
#
#             flat_list = np.array(list_temp).flatten()
#             list_output = sorted(flat_list.tolist())
#
#             print("median for CART:", np.median(list_output))
#
#             if tocsv:
#                 with open("./release_goal{}.csv".format(goal), "a+") as output:
#                     output.write(str(np.median(list_output)) + ",")
#
#         if way == 5:
#             list_temp = []
#             for way in range(repeats):
#                 list_temp.append(CART_DE(data, metrics, month))
#
#             flat_list = np.array(list_temp).flatten()
#             list_output = sorted(flat_list.tolist())
#
#             # print("--------------------------------")
#             print("median for DECART:", np.median(list_output))
#
#             if tocsv:
#                 with open("./release_goal{}.csv".format(goal), "a+") as output:
#                     output.write(str(np.median(list_output)) + "\n")
#
#         if way == 6:
#             list_temp = []
#             for way in range(repeats):
#                 list_temp.append(CART_FLASH(data, metrics, month))
#
#             flat_list = np.array(list_temp).flatten()
#             list_output = sorted(flat_list.tolist())
#
#             print("median for ROME:", np.median(list_output))
#
#
# # goal: "0" for commits, "1" for contributors, "2" for stargazer, "3" for open_PRs,
# # "4" for closed_PRs, "5" for open_issues, "6" for closed_issues,
#
#
#
# if __name__ == '__main__':
#
#     time_begin = time.time()
#
#     repo_pool = []
#     path = r'../release_based_tasks/data_release/'
#     for filename in sorted(os.listdir(path)):
#         repo_pool.append(os.path.join(filename))
#
#     for i in range(4):
#         for repo in repo_pool:
#             if repo == ".DS_Store":
#                 continue
#             print("==============================")
#             print(repo, i)
#             release_learner_check(repo, path, metrics=0, repeats=1, goal=i, month=1, tocsv=False)
#
#     run_time = str(time.time() - time_begin)
#     print(run_time)
#
#     # goal::  0:c 1:nc 2: cd 3:ncd
#
