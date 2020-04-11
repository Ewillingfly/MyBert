'''k-means聚类结果'''
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.cluster import KMeans
# # #from sklearn import datasets
# # from sklearn.datasets import load_iris
# f = open('vec1.txt')
# result = []
# for i in f.readlines():
#     i=eval(i)
#     result.append(i[0])
# estimator = KMeans(n_clusters=7)#构造聚类器
# estimator.fit(result)#聚类
# for i in estimator.labels_:#获取聚类标签
#     print(i)


from sklearn import svm
import numpy as np
from sklearn import model_selection

f = open('vec1.txt')
result = []
for i in f.readlines():
    i=eval(i)
    result.append(i[0])

label = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]


x_train,x_test,y_train,y_test=model_selection.train_test_split(result,label,random_state=42,test_size=0.1)


#（3）搭建模型，训练SVM分类器
# classifier=svm.SVC(kernel='linear',gamma=0.1,decision_function_shape='ovo',C=0.1)
# kernel='linear'时，为线性核函数，C越大分类效果越好，但有可能会过拟合（defaul C=1）。
classifier=svm.SVC(kernel='rbf',gamma=0.01,decision_function_shape='ovo',C=1)
# kernel='rbf'（default）时，为高斯核函数，gamma值越小，分类界面越连续；gamma值越大，分类界面越“散”，分类效果越好，但有可能会过拟合。
# decision_function_shape='ovo'时，为one v one分类问题，即将类别两两之间进行划分，用二分类的方法模拟多分类的结果。
# decision_function_shape='ovr'时，为one v rest分类问题，即一个类别与其他类别进行划分。

classifier.fit(x_train,y_train)
#（4）计算svm分类器的准确率
print("SVM-输出训练集的准确率为：",classifier.score(x_train,y_train))
print("SVM-输出测试集的准确率为：",classifier.score(x_test,y_test))

from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection

x_train,x_test,y_train,y_test=model_selection.train_test_split(result,label,random_state=1,test_size=0.3)
# 用train_test_split将数据随机分为训练集和测试集，测试集占总数据的30%（test_size=0.3),random_state是随机数种子
# 参数解释：
# x：train_data：所要划分的样本特征集。
# y：train_target：所要划分的样本结果。
# test_size：样本占比，如果是整数的话就是样本的数量。
# random_state：是随机数的种子。
# （随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。
# 随机数的产生取决于种子，随机数和种子之间的关系遵从以下两个规则：种子不同，产生不同的随机数；种子相同，即使实例不同也产生相同的随机数。）
#（3）搭建模型，训练GaussianNB分类器
classifier=GaussianNB()
#classifier=BernoulliNB()
#开始训练
classifier.fit(x_train,y_train)
#（4）计算GaussianNB分类器的准确率
print("GaussianNB-输出训练集的准确率为：",classifier.score(x_train,y_train))
print("GaussianNB-输出测试集的准确率为：",classifier.score(x_test,y_test))