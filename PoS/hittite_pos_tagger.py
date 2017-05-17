from bs4 import BeautifulSoup
import os
import re
import numpy as np
import closed_class
import train_set
import word_order


def pos():
    closed = closed_class.Closed()  # гениально

    return closed


def main():
    word_order.word_order() # проставляем порядковый номер слова в обучающем материале
    train_matrix = train_set.make_train_set() # делаем обучающую матрицу

    pos()


if __name__ == '__main__':
    main()