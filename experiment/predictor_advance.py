from experiment.utils import *
from sklearn.tree import DecisionTreeRegressor
from experiment.optimizer import *
from experiment.predictor_baseline import *


def DECART(dataset, metrics, month):

    dataset = normalize(dataset)

    for trainset, testset in df_split(dataset, month):
        train_input = trainset.iloc[:, :-1]
        train_output = trainset.iloc[:, -1]
        test_input = testset.iloc[:, :-1]
        test_output = testset.iloc[:, -1]

    for validate_trainset, validate_testset in df_split(trainset, 1):
        validate_train_input = validate_trainset.iloc[:, :-1]
        validate_train_output = validate_trainset.iloc[:, -1]
        validate_test_input = validate_testset.iloc[:, :-1]
        validate_test_output = validate_testset.iloc[:, -1]

    def cart_builder(a, b, c):
        model = DecisionTreeRegressor(max_depth=a, min_samples_leaf=b, min_samples_split=c)
        model.fit(train_input, train_output)
        test_predict = model.predict(test_input)
        test_actual = test_output.values
        if metrics == 0:
            return mre_calc(test_predict, test_actual)
        if metrics == 1:
            return sa_calc(test_predict, test_actual, train_output)

    def cart_builder_future(a, b, c):
        model = DecisionTreeRegressor(max_depth=a, min_samples_leaf=b, min_samples_split=c)
        model.fit(validate_train_input, validate_train_output)
        validate_test_predict = model.predict(validate_test_input)
        validate_test_actual = validate_test_output.values
        if metrics == 0:
            return mre_calc(validate_test_predict, validate_test_actual)
        if metrics == 1:
            return sa_calc(validate_test_predict, validate_test_actual, validate_train_output)

    config_optimized = de(cart_builder, metrics, bounds=[(1, 12), (1, 12), (2, 21)])[1]
    # print(config_optimized[0], config_optimized[1], config_optimized[2])
    model_touse = DecisionTreeRegressor(max_depth=config_optimized[0],
                                        min_samples_leaf=config_optimized[1],
                                        min_samples_split=config_optimized[2])

    model_touse.fit(train_input, train_output)
    test_predict = np.rint(model_touse.predict(test_input))
    test_actual = test_output.values

    result_list_mre = []
    result_list_sa = []
    # print("DECART", "predict", test_predict, "actual", test_actual)
    result_list_mre.append(mre_calc(test_predict, test_actual))
    result_list_sa.append(sa_calc(test_predict, test_actual, train_output))

    if metrics == 0:
        return result_list_mre
    if metrics == 1:
        return result_list_sa


if __name__ == '__main__':

    path = r'../data/data_cleaned/'
    repo = "joker_monthly.csv"

    metrics = 0  # "0" for MRE, "1" for SA
    repeats = 1
    goal = 1
    month = 1

    data = data_goal_arrange(repo, path, goal)

    list_temp = []
    for way in range(repeats):
        list_temp.append(DECART(data, metrics, month))

    flat_list = np.array(list_temp).flatten()
    list_output = sorted(flat_list.tolist())

    print(list_output)
    print("median DECART:", np.median(list_output))

    list_temp = []
    for way in range(repeats):
        list_temp.append(CART(data, month)[metrics])

    flat_list = np.array(list_temp).flatten()
    list_output = sorted(flat_list.tolist())

    print(list_output)
    print("median CART:", np.median(list_output))

