from bs4 import BeautifulSoup
import os
import re

def make_train_set(): # делаем обучающий сет
    fname_list = os.listdir('../materials/') # массив имён файлов с обучающими материалами в .xlsx

    fw = open('train_set.csv', 'w', encoding='utf-8') # тут будет обучающая таблица

    for fname in fname_list: # для каждого файла

        print('processing ' + fname + '\n')
        os.system('/usr/bin/libreoffice --convert-to xml ../materials/' + fname) # конвертируем в .xml
        # результат положится в папку с этой программой


        soup = BeautifulSoup(open('./' + fname.replace('xlsx', 'xml')), 'xml') # читаем дерево

        row_list = []
        for l in soup.find_all('table', attrs={'table:name': 'Word Forms'}): # идём во второй лист "Word Forms"
            for i, line in enumerate(l.find_all('table-row')):
                row_list.append(re.split('\n{1,3}', line.get_text().replace('\xa0', '').strip('\n'))) #считываем каждый ряд
                # и получаем массив, где каждый элемент -- массив с содержимым каждой ячейки в этом ряду

        row_list.pop(0) # удаляем первые два ряда,
        row_list.pop(0) # потому что в них названия столбцов

        for row in row_list:
            if row[0] != '':
                if row[7] == 'fragment':
                    print(row)
                    fw.write(row[5].strip('! ') + ',' + row[7] + '\n') # записываем по слову в строку
                else:
                    print(row)
                    fw.write(row[5].strip('! ') + ',' + row[8] + '\n') # записываем по слову в строку

    fw.close()