import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from seqlearn.hmm import MultinomialHMM
import train_set
import predict_set


def knn_pred(X, y):
    # k ближайших соседей
    knn = KNeighborsClassifier(n_neighbors=11, metric='euclidean')
    knn.fit(X, y)
    return knn


def lr_pred(X, y):
    # логистическая регрессия
    lr = LogisticRegression(penalty="l2", fit_intercept=True, max_iter=100, C=27, solver="lbfgs")
    lr.fit(X, y)
    return lr


def svm_pred(X, y):
    # метод опорных векторов
    svm = SVC(C=500)
    svm.fit(X, y)
    return svm


def dtree_pred(X, y):
    # решающие деревья
    dtree = DecisionTreeClassifier(max_depth=21)
    dtree.fit(X, y)
    return dtree


def rforest_pred(X, y):
    # случайный лес
    rforest = RandomForestClassifier(n_estimators=31)
    rforest.fit(X, y)
    return rforest


def sgd_pred(X, y):
    # метод стохастического градиента
    sgd = SGDClassifier(alpha=0.005)
    sgd.fit(X, y)
    return sgd


def hmm_pred(X, y):
    # скрытая марковская модель
    hmm = MultinomialHMM(alpha=0.1)
    hmm.fit(X, y, lengths=np.array([1 for i in y]))

    return hmm


def pos_tagger():

    X, y = train_set.make_train_set() # делаем обучающую матрицу

    knn = knn_pred(X, y)
    print("got knn")

    lr = lr_pred(X, y)
    print("got logreg")

    svm = svm_pred(X, y)
    print("got svm")

    dtree = dtree_pred(X, y)
    print("got decision tree")

    rforest = rforest_pred(X, y)
    print("got random forest")

    sgd = sgd_pred(X, y)
    print("got sgd")

    hmm= hmm_pred(X, y)
    print("got hmm")

    return knn, lr, svm, dtree, rforest, sgd, hmm


def main():
    pos_tagger()


if __name__ == '__main__':
    main()