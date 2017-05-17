from bs4 import BeautifulSoup
import os
import re
import xlrd


def make_predict_set():
    fname_list = os.listdir('../predict/') # массив имён файлов с обучающими материалами в .xlsx

    fw = open('predict_set.csv', 'w', encoding='utf-8') # тут будет таблица для определения части речи

    for fname in fname_list: # для каждого файла

        print('processing ' + fname + '\n')
        #os.system('libreoffice --convert-to xml --outdir ./predict/ ../predict/' + fname) # конвертируем в .xml
        # результат положится в папку predict в данной дериктории

        #soup = BeautifulSoup(open('./predict/' + fname.replace('xlsx', 'xml')), 'xml')  # читаем дерево
        rb = xlrd.open_workbook('../predict/' + fname) # читаем дерево
        sheet = rb.sheet_by_index(0)
        norm_spell = [] # массив с предложениями из колонки 'normalized spelling'
        for rownum in range(2, sheet.nrows): # обходим значения каждого ряда с двух, потому что первые два -- названия колонок
            row = sheet.row_values(rownum)
            norm_spell.append(row[12])

        print(norm_spell)


def main():
    make_predict_set()


if __name__ == '__main__':
    main()