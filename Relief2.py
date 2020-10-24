import numpy as np
from random import randrange


def relief(features, labels, times):  # 传入特征矩阵，标签矩阵和随机选择的次数，因为有可能样本有很多，所以我们随机选择若干个样本来计算;这里的矩阵都是np.array
    (n_samples, n_features) = np.shape(features)
    delta = []
    delta_features = []
    delta_index = []
    sample_distance = sap_distance(features)  # 计算每两个样本之间的距离
    new_features = normalize(features)  # 对特征值归一化

    """
    # 下面开始计算相关统计量，并对各个特征的相关统计量进行比较，最后返回各个特征值相关统计量从高到低的排名
    # 这是将随机选取的样本代入计算出来的delta
    for i in range(0, times):
        randnum = randrange(0, n_samples, 1)        # 生成一个随机数
        one_sample = features[randnum]        # 随机选择一个样本
        (nearhit, nearmiss, nearhit_index, nearmiss_index) = search_near(sample_distance, labels, randnum, features)  # 找出猜中近邻和猜错近邻,nearhit为猜中近邻样本的行向量
        delta.append(relevant_feature(nearhit_index, nearmiss_index, new_features, randnum))  # 计算相关统计量矩阵
    delta = np.asarray(delta)
    for j in range(0, n_features):
        delta_features.append(np.sum(delta[:, j]))
    midd = list(set(delta_features))
    midd.sort(reverse=True)
    for p in midd:
        for q in range(0, len(delta_features)):
            if delta_features[q] == p:
                delta_index.append(q)
    return delta_index
    """
    # 这是将所有样本都带入计算的delta
    for i in range(0, n_samples):
        (nearhit, nearmiss, nearhit_index, nearmiss_index) = search_near(sample_distance, labels, i,
                                                                         features)  # 找出猜中近邻和猜错近邻,nearhit为猜中近邻样本的行向量
        delta.append(relevant_feature(nearhit_index, nearmiss_index, new_features, i))  # 计算相关统计量矩阵
    delta = np.asarray(delta)
    for j in range(0, n_features):
        delta_features.append(np.sum(delta[:, j]))
    midd = list(set(delta_features))
    midd.sort(reverse=True)
    for p in midd:
        for q in range(0, len(delta_features)):
            if delta_features[q] == p:
                delta_index.append(q)
    return delta_index


def normalize(features):
    (n_samples, n_features) = np.shape(features)
    print("shape=", n_samples, n_features)
    fe_max = []
    fe_min = []
    n_deno = []
    new_features = np.zeros((n_samples, n_features))
    print("new_features=", new_features)
    for i in range(0, n_features):
        max_index = np.argmax(features[:, i])
        min_index = np.argmin(features[:, i])
        fe_max.append(features[max_index, i])  # 计算每一个特征的最大值
        fe_min.append(features[min_index, i])  # 计算每一个特征的最小值
    n_deno = np.asarray(fe_max) - np.asarray(fe_min)  # 求出归一化的分母
    for j in range(0, n_features):
        for k in range(0, n_samples):
            new_features[k, j] = (features[k, j] - fe_min[j]) / n_deno[j]  # 归一化
    return new_features


def sap_distance(features):
    (n_samples, n_features) = np.shape(features)
    distance = np.zeros((n_samples, n_samples))
    for i in range(0, n_samples):
        for j in range(0, n_samples):
            diff_distance = features[i] - features[j]
            if i == j:
                distance[i, j] = 9999
            else:
                distance[i, j] = euclid_distance(diff_distance)  # 使用欧几里德距离定义样本之间的距离
    print("距离：", distance)
    return distance


def euclid_distance(diff_distance):
    counter = np.power(diff_distance, 2)
    counter = np.sum(counter)
    counter = np.sqrt(counter)
    return counter


def search_near(sample_distance, labels, randnum, feartures):
    (n_samples, n_features) = np.shape(feartures)
    nearhit_list = []
    nearmiss_list = []
    hit_index = []
    miss_index = []
    for i in range(0, n_samples):
        if labels[i] == labels[randnum]:
            nearhit_list.append(sample_distance[i, randnum])  # 将距离放在一个列表里面
            hit_index.append(i)  # 将样本标号放在另一个列表里面
        else:
            nearmiss_list.append(sample_distance[i, randnum])
            miss_index.append(i)
    nearhit_dis_index = nearhit_list.index(min(nearhit_list))  # 算出猜中近邻
    nearhit_index = hit_index[nearhit_dis_index]  # 将猜中近邻的样本标号赋给nearhit_index

    nearmiss_dis_index = nearmiss_list.index(min(nearmiss_list))
    nearmiss_index = miss_index[nearmiss_dis_index]

    nearhit = feartures[nearhit_index]
    nearmiss = feartures[nearmiss_index]

    return nearhit, nearmiss, nearhit_index, nearmiss_index


def relevant_feature(nearhit_index, nearmiss_index, new_features, randnum):
    diff_hit = abs(new_features[nearhit_index] - new_features[randnum])
    diff_miss = abs(new_features[nearmiss_index] - new_features[randnum])
    delta = -np.power(diff_hit, 2) + np.power(diff_miss, 2)
    return delta


input_vecs = [[697, 460], [774, 376], [634, 264], [608, 318],
              [556, 215], [403, 237], [481, 149], [437, 211],
              [666, 91], [243, 267], [245, 57], [343, 99],
              [639, 161], [657, 198], [360, 370], [593, 42],
              [719, 103]]

labels = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
out_features = np.array(input_vecs)

times = 22
features_importance = relief(out_features, labels, times)
print("排序：", features_importance)
