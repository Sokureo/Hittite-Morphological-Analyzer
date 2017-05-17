import os
import openpyxl


def word_order():

    fname_list = os.listdir('../materials_xlsx/') # массив имён файлов с обучающими материалами в .xlsx

    for fname in fname_list: # для каждого файла

        wb = openpyxl.load_workbook(filename='../materials_xlsx/' + fname)  # читаем дерево
        sheet = wb['Word Forms'] # идём в лист 'Word Forms'

        clause_index = 0
        rownum = 3 # начинаем с третьего ряда, ибо первые два -- названия колонок
        clause = ' '

        while clause:
            clause = sheet['C' + str(rownum)].value
            if sheet['C' + str(rownum)].value == clause_index:
                wo += 1 # считаем порядковый номер слова в предложении
            elif not clause:
                break
            else:
                wo = 1
                clause_index = sheet['C' + str(rownum)].value
            sheet.cell(column=11, row=rownum, value=wo)
            rownum += 1

        wb.save(filename='../materials_xlsx/' + fname)

