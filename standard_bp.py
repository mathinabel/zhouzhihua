import numpy as np

X = [[1, 1, 1, 1, 1, 1, 0.697, 0.460],
     [2, 1, 2, 1, 1, 1, 0.774, 0.376],
     [2, 1, 1, 1, 1, 1, 0.634, 0.264],
     [1, 1, 2, 1, 1, 1, 0.608, 0.318],
     [2, 1, 1, 1, 1, 1, 0.556, 0.215],
     [1, 2, 1, 1, 2, 2, 0.403, 0.237],
     [2, 2, 1, 2, 2, 2, 0.481, 0.149],
     [2, 2, 1, 1, 2, 1, 0.437, 0.211],
     [2, 2, 2, 2, 2, 1, 0.666, 0.091],
     [1, 3, 3, 1, 3, 2, 0.243, 0.267],
     [2, 3, 3, 3, 3, 1, 0.245, 0.057],
     [2, 1, 1, 3, 3, 2, 0.343, 0.099],
     [1, 2, 1, 2, 1, 1, 0.639, 0.161],
     [2, 2, 2, 2, 1, 1, 0.657, 0.198],
     [2, 2, 1, 1, 2, 2, 0.360, 0.370],
     [2, 1, 1, 3, 3, 1, 0.593, 0.042],
     [1, 1, 2, 2, 2, 1, 0.719, 0.103]]
Y = [1, 1, 1, 1, 1, 1, 1, 1,
     0, 0, 0, 0, 0, 0, 0, 0, 0]
X1 = np.asarray(X)
Y1 = np.asarray(Y)


def forward(weighted_input):
    return 1.0 / (1.0 + np.exp(-weighted_input))


def bp(batch):
    weights0 = np.random.uniform(0, 1, (8, 8))
    # 行列
    bias0 = np.zeros(8)

    weights1 = np.random.uniform(0, 1, (8, 8))
    bias1 = np.zeros(8)
    for i in range(batch):
        for i in range(17):
            # 前向计算隐藏层
            x = np.dot(X1[i], weights0)
            a1 = forward(x)
            b4 = a1 + bias0
            # print(b4)
            y = np.dot(b4, weights1)
            y1 = forward(y)
            y4 = y1 + bias1
            # print(y4)
            # 取平均的y4为输出
            y5 = np.sum(y4) / 8

            # 输出层梯度，用于更新w1
            e_y = y4 * (1 - y4) * (Y[i] - y4)
            gj = np.array(e_y)
            gj = np.array([gj])
            # print(gj.T)
            # 更新W1

            aa = np.array(b4)
            aa = np.array([aa])
            weights1 = weights1 + 0.1 * np.dot(gj.T, aa)
            # print(weights0)

            # 更新V矩阵
            weights11 = np.array([weights1.sum(axis=0)])
            e_b4 = b4 * (1 - b4) * (weights11 * gj)

            # print(e_b4)

            bb1 = np.array(X1[i])
            bb1 = np.array([bb1])

            weights0 = weights0 + 0.1 * np.dot(e_b4.T, bb1)
            # print( e_b4)
    # 测试
    for k in range(17):
        x = np.dot(X1[k], weights0)
        a1 = forward(x)
        b4 = a1 + bias0
        # print(b4)
        y = np.dot(b4, weights1)
        y1 = forward(y)
        y4 = y1 + bias1
        y5 = np.sum(y4) / 8
    # print(y4)


bp(10)
