from bs4 import BeautifulSoup
import os
import re
import numpy as np
import closed_class
import train_set


def pos():
    closed = closed_class.Closed()  # гениально
    print(closed.connector)

    return closed

def main():
    #train_set.make_train_set()

    pos()


if __name__ == '__main__':
    main()