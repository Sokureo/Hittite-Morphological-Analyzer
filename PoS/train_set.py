from bs4 import BeautifulSoup
import os
import re
import numpy as np
import closed_class


def make_train_set(): # делаем обучающий сет

    closed = closed_class.Closed()
    # считываем класс закрытых классов
    speach_class = {i.split()[1]: i.split()[0] for i in open('../class.txt').read().split('\n') if i != ''}
    # считываем части речи с их порядковым номером в словарь
    features = open('../features.txt').read().split()
    # считываем признаки

    train_word_list = []  # массив для обучающих слов
    train_features_set = []  # массив для обучающих признаков
    train_tag_list = []  # массив с правильными частями речи

    fname_list = os.listdir('../materials_xlsx/') # массив имён файлов с обучающими материалами в .xlsx

    for fname in fname_list: # для каждого файла

        print('processing ' + fname + '\n')
        #os.system('libreoffice --convert-to xml --outdir ./materials_xml/ ../materials_xlsx/' + fname) # конвертируем в .xml
        # результат положится в папку materials_xml в данной дериктории

        soup = BeautifulSoup(open('./materials_xml/' + fname.replace('xlsx', 'xml')), 'xml') # читаем дерево

        row_list = []
        for l in soup.find_all('table', attrs={'table:name': 'Word Forms'}): # идём во второй лист "Word Forms"
            for i, line in enumerate(l.find_all('table-row')):
                row_list.append(re.split('\n{1,3}', line.get_text().replace('\xa0', '').strip('\n'))) #считываем каждый ряд
                # и получаем массив, где каждый элемент -- массив с содержимым каждой ячейки в этом ряду

        row_list.pop(0) # удаляем первые два ряда,
        row_list.pop(0) # потому что в них названия столбцов

        for row in row_list:
            if row[0] != '':
                if len(row) > 8 and row[8].strip() in speach_class.keys():
                    example = []
                    train_word_list.append(row[5].strip('!: ')) # добавляем слово
                    for feat in features: # добавляем его признаки
                        if row[5].strip('!: ').endswith(feat):
                            example.append(1) # есть признак
                        else:
                            example.append(0) # нет признака
                    example.append(row[10]) # добавляем номер в предложении
                    train_tag_list.append(speach_class[row[8].strip()]) # порядковый номер правильной части речи
                    train_features_set.append(example) # получаем матрицу из примеров

    train_features_set = np.array(train_features_set)
    train_tag_list = np.array(train_tag_list)

    print('train_matrix made')

    return train_features_set, train_tag_list


