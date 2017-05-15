import subprocess
import os

#cmd = ['"C:\Program Files (x86)\LibreOffice 5\program\soffice.exe" --convert-to xml "C:\materials\Letters-NH-3_KUB-19-5-KBo-19-79.xlsx"']
#for c in cmd:
    #subprocess.call(c)

#subprocess.call('/usr/bin/libreoffice --convert-to xml "/home/sokureo/Documents/Letters-NH-3_KUB-19-5-KBo-19-79.xlsx"')

os.system('/usr/bin/libreoffice --convert-to xml /home/sokureo/Documents/Hittite/materials/Letters-NH-3_KUB-19-5-KBo-19-79.xlsx --outdir /home/sokureo/Documents')
#/usr/lib/libreoffice/program/soffice --convert-to xml /home/sokureo/Documents/Hittite/materials/Letters-NH-3_KUB-19-5-KBo-19-79.xlsx --outdir /home/sokureo/Documents
