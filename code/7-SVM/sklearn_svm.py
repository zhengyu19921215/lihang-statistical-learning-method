from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import  train_test_split
import matplotlib.pyplot as plt
'''
### sklearn.svm.SVC


C = 1.0, kernel = 'rbf', degree = 3, gamma = 'auto', coef0 = 0.0, shrinking = True, probability = False, tol = 0.001, cache_size = 200, class_weight = None, verbose = False, max_iter = -1, decision_function_shape = None, random_state = None) *

参数：

- C：C - SVC的惩罚参数C?默认值是1
.0

C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，趋向于对训练集全分对的情况，这样对训练集测试时准确率很高，但泛化能力弱。C值小，对误分类的惩罚减小，允许容错，将他们当成噪声点，泛化能力较强。

- kernel ：核函数，默认是rbf，可以是‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’

– 线性：u'v

– 多项式：(gamma * u'*v + coef0)^degree

– RBF函数：exp(-gamma | u - v | ^ 2)

– sigmoid：tanh(gamma * u'*v + coef0)

               - degree ：多项式poly函数的维度，默认是3，选择其他核函数时会被忽略。


- gamma ： ‘rbf’, ‘poly’ 和‘sigmoid’的核函数参数。默认是’auto’，则会选择1 / n_features

                                                   - coef0 ：核函数的常数项。对于‘poly’和 ‘sigmoid’有用。


- probability ：是否采用概率估计？.默认为False

                         - shrinking ：是否采用shrinking
heuristic方法，默认为true

            - tol ：停止训练的误差值大小，默认为1e - 3

                              - cache_size ：核函数cache缓存大小，默认为200

                                                         - class_weight ：类别的权重，字典形式传递。设置第几类的参数C为weight * C(C - SVC中的C)

                                                                                      - verbose ：允许冗余输出？


- max_iter ：最大迭代次数。-1
为无限制。


- decision_function_shape ：‘ovo’, ‘ovr’ or None, default = None3

                                                           - random_state ：数据洗牌时的种子值，int值

主要调节的参数有：C、kernel、degree、gamma、coef0。
'''
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    for i in range(len(data)):
        if data[i,-1] == 0:
            data[i,-1] = -1
    # print(data)
    return data[:,:2], data[:,-1]

X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
clf = SVC()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
