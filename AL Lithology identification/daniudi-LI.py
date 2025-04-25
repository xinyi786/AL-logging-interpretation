import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from query_strategies.uncertainty_margin import UncertaintyMargin
from query_strategies.uncertainty_entropy import UncertaintyEntropy
from query_strategies.active_learning_with_cost_embedding import ALCE
from query_strategies.maximum_expected_cost import MEC
from query_strategies.cost_weighted_minimum_margin import CWMM
from query_strategies.random import Random
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import sys
from collections import Counter
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()


def read_data(file_path):
    x_data1 = np.array(pd.read_excel(file_path).astype(float))[:, 1:8]
    x_data = scaler.fit_transform(x_data1)
    y_data = np.array(pd.read_excel(file_path).astype(int))[:, -1] + 1

    return x_data, y_data


x_data, y_data = read_data("./daniudi-data.xlsx")
print(x_data, y_data)
print("count：", Counter(y_data))

C = 8  # number of class
N = x_data.shape[0]  # number of instances
T = 400  # number of testing instances
Q = 200  # number of queries


def shuffle_data(x_data, y_data, N, T, C):
    idx = np.arange(N)
    np.random.shuffle(idx)
    x_pool = x_data[idx[:-T]]
    print(x_pool.shape)
    y_pool = y_data[idx[:-T]]
    x_test = x_data[idx[-T:]]
    print(x_test.shape)
    y_test = y_data[idx[-T:]]
    N_train = x_pool.shape[0]

    # sample initial labeled instances
    idx_lbl = np.zeros((N_train,), dtype=bool)
    for i in range(C):
        idx_lbl[np.random.choice(np.where(y_pool == (i + 1))[0])] = True

    # # you can generate random cost matrix in this
    # unique, counts = np.unique(y_data, return_counts=True)
    # class_counts = dict(zip(unique, counts))
    # cost_mat = np.zeros((C, C))
    # for i in range(C):
    #     for j in range(C):
    #         if i == j:
    #             continue
    #         cost_mat[i, j] = np.random.random() * 100 * class_counts[j + 1] / class_counts[i + 1]

    # daniudi data cost-mat
    cost_mat = np.array([[0, 10, 6, 6, 6, 4, 4, 4],
                         [10, 0, 2, 1, 1, 1, 2, 3],
                         [6, 2, 0, 2, 1, 1, 2, 2],
                         [6, 1, 2, 0, 1, 1, 2, 2],
                         [6, 1, 1, 1, 0, 1, 1, 2],
                         [4, 1, 1, 1, 1, 0, 1, 1],
                         [4, 2, 2, 2, 1, 1, 0, 1],
                         [4, 3, 2, 2, 2, 1, 1, 0]])

    return x_pool, y_pool, x_test, y_test, idx_lbl, cost_mat


total_results = np.zeros((Q, 6))
total_accuracy = np.zeros((Q, 6))

# run several experiments
sys.stderr.write('total  #####\n')
sys.stderr.write('runing ')

for i in [10, 20, 30]:
    np.random.seed(i)
    # shuffle dataset
    x_pool, y_pool, x_test, y_test, idx_lbl, cost_mat = shuffle_data(x_data, y_data, N, T, C)
    # different AL models
    models = \
        [UncertaintyMargin(x_pool, y_pool * idx_lbl),
         UncertaintyEntropy(x_pool, y_pool * idx_lbl),
         Random(x_pool, y_pool * idx_lbl),
         ALCE(x_pool, y_pool * idx_lbl, cost_mat),
         CWMM(x_pool, y_pool * idx_lbl, cost_mat),
         MEC(x_pool, y_pool * idx_lbl, cost_mat)]

    # for recording rewards and actions
    results = np.zeros((Q, len(models)))
    accuracy = np.zeros((Q, len(models)))
    idx_lbls = np.repeat(idx_lbl[:, None], len(models), axis=1)

    for nq in range(Q):
        for j, model in enumerate(models):
            q = model.query()
            model.update(q, y_pool[q])
            idx_lbls[q, j] = True

            # testing performance
            clf = OneVsRestClassifier(SVC())
            clf.fit(x_pool[idx_lbls[:, j]], y_pool[idx_lbls[:, j]])
            p_test = clf.predict(x_test)
            # results[nq, j] = np.mean([cost_mat[y-1, p-1] for y, p in zip(y_test, p_test)])
            results[nq, j] = np.sum(cost_mat * confusion_matrix(y_test, p_test))
            accuracy[nq, j] = np.mean(accuracy_score(y_test, p_test))
    # ipdb.set_trace()
    # print 1

    total_results += results
    total_accuracy += accuracy
    sys.stderr.write('#')

sys.stderr.write('\nPlease see result.png\n')
avg_results = total_results / 3
avg_accuracy = total_accuracy / 3
print(avg_results.shape)
print(avg_accuracy.shape)

# plot cost result
plt.figure(figsize=(7, 6))
show_x = np.arange(0, Q, 5)
plt.plot(show_x, avg_results[::5, 0], '-o',  linewidth=2, ms=4, label='Random')
plt.plot(show_x, avg_results[::5, 1], '-o',  linewidth=2, ms=4, label='UCE')
plt.plot(show_x, avg_results[::5, 2], '-o',  linewidth=2, ms=4, label='UCM')
plt.plot(show_x, avg_results[::5, 3], '-o',  linewidth=2, ms=4, label='ALCE')
plt.plot(show_x, avg_results[::5, 4], '-o',  linewidth=2, ms=4, label='CWMM')
plt.plot(show_x, avg_results[::5, 5], '-o',  linewidth=2, ms=4, label='MEC')
plt.xlabel('number of queries')
plt.ylabel('average costs')
plt.legend(loc='upper right')
plt.savefig('./cost.png')

# plot accuracy result
plt.figure(figsize=(7, 6))
show_x = np.arange(0, Q, 5)
plt.plot(show_x, avg_accuracy[::5, 0], '-o',  linewidth=2, ms=4, label='Random')
plt.plot(show_x, avg_accuracy[::5, 1], '-o',  linewidth=2, ms=4, label='UCE')
plt.plot(show_x, avg_accuracy[::5, 2], '-o',  linewidth=2, ms=4, label='UCM')
plt.plot(show_x, avg_accuracy[::5, 3], '-o',  linewidth=2, ms=4, label='ALCE')
plt.plot(show_x, avg_accuracy[::5, 4], '-o',  linewidth=2, ms=4, label='CWMM')
plt.plot(show_x, avg_accuracy[::5, 5], '-o',  linewidth=2, ms=4, label='MEC')
plt.xlabel('number of queries')
plt.ylabel('average accuracy')
plt.legend(loc='lower right')
plt.savefig('./acc.png')
plt.show()
