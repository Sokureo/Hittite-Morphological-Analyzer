import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import train_set
import word_order


def get_estimation(algorithm, X_train, X_test, y_train, y_test):
    # считаем всякие оценки
    algorithm.fit(X_train, y_train)
    y_pred = algorithm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


def knn_pred(n, X_train, X_test, y_train, y_test):
    # k ближайших соседей
    knn = KNeighborsClassifier(n_neighbors=n, metric='euclidean')
    knn_estimation = get_estimation(knn, X_train, X_test, y_train, y_test)
    return knn_estimation


def lr_pred(с, X_train, X_test, y_train, y_test):
    # логистическая регрессия
    lr = LogisticRegression(penalty="l2", fit_intercept=True, max_iter=100, C=с, solver="lbfgs", random_state=12345)
    lr_estimation = get_estimation(lr, X_train, X_test, y_train, y_test)
    return lr_estimation


def kfolds_validation(model, param, X, y):
    # кросс-валидация
    kf = KFold(n_splits=4, shuffle = True, random_state=12345)
    estimations = []

    for train, test in kf.split(X):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        estimations.append(model(param, X_train, X_test, y_train, y_test))

    estimation = np.array(estimations).mean() # средняя оценка по фолдам

    return estimation


def get_params_dic(model, params, X, y):
    # создаём словарь, где ключи -- параметр, а значения -- оценка при этом параметре
    estimations = {}

    for param in params:
        estimations[param] = kfolds_validation(model, param, X, y)

    return estimations


def create_plot(model, estimatiom):
    # строим график зависимости параметра и оценки
    plt.bar(range(len(estimatiom)), [value for value in estimatiom.values()])
    plt.xticks(range(len(estimatiom)), [key for key in estimatiom.keys()], rotation='vertical')
    plt.title(model + ' accuracy')
    plt.xlabel('Hyperparameters')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig('./graphs/' + model + ' accuracy')
    plt.close()
    print(model + ' plot created\n')


def pos_tagger():

    #word_order.word_order() # проставляем порядковый номер слова в обучающем материале
    X, y = train_set.make_train_set() # делаем обучающую матрицу

    ns = np.arange(1, 150, 10)  # количество соседей
    cs = np.logspace(-2, 10, 8, base=10)  # параметр регуляризации

    knn_estimation = get_params_dic(knn_pred, ns, X, y)
    print('knn accuracy: ' + '\n', '\n'.join(['{}: {}'.format(key, knn_estimation[key]) for key in knn_estimation]), '\n')
    create_plot('knn algorithm', knn_estimation)


    lr_estimation = get_params_dic(lr_pred, cs, X, y)
    print('log reg accuracy: ' + '\n', '\n'.join(['{}: {}'.format(key, lr_estimation[key]) for key in lr_estimation]), '\n')
    create_plot('log regression algorithm', lr_estimation)


def main():
    pos_tagger()


if __name__ == '__main__':
    main()