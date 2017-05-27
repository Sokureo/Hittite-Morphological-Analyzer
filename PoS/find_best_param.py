import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from seqlearn.hmm import MultinomialHMM
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import train_set


def get_estimation(algorithm, X_train, X_test, y_train, y_test):
    # считаем всякие оценки
    algorithm.fit(X_train, y_train)
    y_pred = algorithm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    F1 = f1_score(y_test, y_pred, average='weighted')

    return [accuracy, precision, recall, F1]


def knn_pred(n, X_train, X_test, y_train, y_test):
    # k ближайших соседей
    knn = KNeighborsClassifier(n_neighbors=n, metric='euclidean')
    knn_estimation = get_estimation(knn, X_train, X_test, y_train, y_test)
    return knn_estimation


def lr_pred(с, X_train, X_test, y_train, y_test):
    # логистическая регрессия
    lr = LogisticRegression(penalty="l2", fit_intercept=True, max_iter=100, C=с, solver="lbfgs")
    lr_estimation = get_estimation(lr, X_train, X_test, y_train, y_test)
    return lr_estimation


def svm_pred(c, X_train, X_test, y_train, y_test):
    # метод опорных векторов
    svm = SVC(C=c)
    svm_estimation = get_estimation(svm, X_train, X_test, y_train, y_test)
    return svm_estimation


def dtree_pred(m, X_train, X_test, y_train, y_test):
    # решающие деревья
    dtree = DecisionTreeClassifier(max_depth=m)
    dtree_estimation = get_estimation(dtree, X_train, X_test, y_train, y_test)
    return dtree_estimation


def rforest_pred(n, X_train, X_test, y_train, y_test):
    # случайный лес
    rforest = RandomForestClassifier(n_estimators=n)
    rforest_estimation = get_estimation(rforest, X_train, X_test, y_train, y_test)
    return rforest_estimation


def sgd_pred(a, X_train, X_test, y_train, y_test):
    # метод стохастического градиента
    sgd = SGDClassifier(alpha=a)
    sgd_estimation = get_estimation(sgd, X_train, X_test, y_train, y_test)
    return sgd_estimation


def hmm_pred(a, X_train, X_test, y_train, y_test):
    # скрытая марковская модель
    hmm = MultinomialHMM(alpha=a)
    hmm.fit(X_train, y_train, lengths=np.array([1 for i in y_train]))
    y_pred = hmm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    F1 = f1_score(y_test, y_pred, average='weighted')
    return [accuracy, precision, recall, F1]


def kfolds_validation(model, param, X, y):
    # кросс-валидация
    kf = KFold(n_splits=4, shuffle=True, random_state=12345)
    estimations = []

    for train, test in kf.split(X):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        estimations.append(model(param, X_train, X_test, y_train, y_test)[3])

    estimation = np.array(estimations).mean() # средняя оценка по фолдам

    return estimation


def get_params_dic(model, params, X, y):
    # создаём словарь, где ключи -- параметр, а значения -- оценка при этом параметре
    estimations = {}

    for param in params:
        estimations[param] = kfolds_validation(model, param, X, y)

    return estimations


def create_plot(model, estimation):
    # строим график зависимости параметра и оценки
    plt.plot([key for key in estimation.keys()], [value for value in estimation.values()])
    plt.title('F1-score of ' + model)
    plt.xlabel('Hyperparameters')
    plt.ylabel('F1-score')
    plt.tight_layout()
    plt.savefig('./graphs/' + model + ' F1-score')
    plt.close()
    print(model + ' plot created\n')


def best_param():

    X, y = train_set.make_train_set()  # делаем обучающую матрицу

    ns_knn = np.arange(1, 150, 10)  # количество соседей
    cs_lr = np.logspace(-2, 10, 8, base=10)  # параметр регуляризации для логрега
    cs_svm = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500]  # параметр регуляризации для опорных векторов
    ms = np.arange(1, 150, 10)  # максимальная глубина деревьев
    ns_rf = np.arange(1, 150, 10)  # количество деревьев в лесу
    als = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]
    # параметр регуляризации для стохастического градиента и скрытой марковской модели

    knn_estimation = get_params_dic(knn_pred, ns_knn, X, y)
    print('knn F1-score: ' + '\n', '\n'.join(['{}: {}'.format(key, knn_estimation[key]) for key in knn_estimation]), '\n')
    create_plot('kNN Classifier', knn_estimation)

    lr_estimation = get_params_dic(lr_pred, cs_lr, X, y)
    print('log reg F1-score: ' + '\n', '\n'.join(['{}: {}'.format(key, lr_estimation[key]) for key in lr_estimation]), '\n')
    create_plot('Logistic Regression', lr_estimation)

    svm_estimation = get_params_dic(svm_pred, cs_svm, X, y)
    print('svm F1-score: ' + '\n', '\n'.join(['{}: {}'.format(key, svm_estimation[key]) for key in svm_estimation]), '\n')
    create_plot('SVM', svm_estimation)

    dtree_estimation = get_params_dic(dtree_pred, ms, X, y)
    print('decision tree F1-score: ' + '\n', '\n'.join(['{}: {}'.format(key, dtree_estimation[key]) for key in dtree_estimation]), '\n')
    create_plot('Decision Tree', dtree_estimation)

    rforest_estimation = get_params_dic(rforest_pred, ns_rf, X, y)
    print('random forest F1-score: ' + '\n', '\n'.join(['{}: {}'.format(key, rforest_estimation[key]) for key in rforest_estimation]), '\n')
    create_plot('Random Forest', rforest_estimation)

    sgd_estimation = get_params_dic(sgd_pred, als, X, y)
    print('Stochastic gradient descent F1-score: ' + '\n', '\n'.join(['{}: {}'.format(key, sgd_estimation[key]) for key in sgd_estimation]), '\n')
    create_plot('Stochastic Gradient Descent', sgd_estimation)

    hmm_estimation = get_params_dic(hmm_pred, als, X, y)
    print('Hidden Markov model F1-score: ' + '\n', '\n'.join(['{}: {}'.format(key, hmm_estimation[key]) for key in hmm_estimation]), '\n')
    create_plot('Hidden Markov Model', hmm_estimation)


def main():
    best_param()


if __name__ == '__main__':
    main()