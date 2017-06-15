# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 17:34:00 2017

@author: Alexsandr
"""

import sqlite3
import datetime
import numpy as np

conn = sqlite3.connect('checkbase.db')
mass = []
massgroup = []
with conn:
    cur = conn.cursor()
#Запись в массив по институтам
    cur.execute('SELECT inst№ FROM inst')
rows = cur.fetchall()
for row in rows:
     mass.append(int("%2s" % row))
#Запись в массив по Группам
cur.execute('SELECT numGroup FROM CheckGroup')
rows = cur.fetchall()
for row in rows:
     massgroup.append(int("%2s" % row))

massnumb = np.arange(1,100)
# Функции проверки:

# Год
def check_Year(a):
 now = datetime.datetime.now()
 if ((now.year)-a)<=((now.year)-((now.year)-5)):
  return 1
 else:
  return 0

# Направление
def check_inst(b):
 if b in mass:
  return 1
 else:
  return 0

# Направление
def check_group(b):
 if b in massgroup:
  return 1
 else:
  return 0

#номер студента
def check_number(c):
 if c in massnumb:
  return 1
 else:
  return 0

from keras.models import load_model
from keras.preprocessing import image
model = load_model('my_model.h5')
img_path = '1.png'
img = image.load_img(img_path)
x = image.img_to_array(img)
x = x.reshape(28,28)
ex = model.predict(x)

# дробление ID на части
year = int(ex[0:4])
inst = int(ex[4:6])
group = int(ex[7])
number = int(ex[8:9])
yrcheck = check_Year(year)

# Проверка года
if yrcheck == 1:
 instcheck=check_inst(inst)
else:
 print('Проверка года не пройдена')

#Проверка направления
if instcheck == 1:
 groupcheck = check_group(group)
 cur.execute("SELECT name FROM inst WHERE inst№=:inststr",{"inststr": inst})
 print ("%2s" % cur.fetchone())
else:
 print('Проверка института не пройдена')

 #Проверка группы
if groupcheck == 1:
 studnomber = check_number(number)
 cur.execute("SELECT name FROM CheckGroup WHERE NumGroup=:groupstr",{"groupstr": group})
 print ("%2s" % cur.fetchone())
else:
 print('Проверка группы не пройдена')

#Проверка номера
if studnomber == 1:
 print('Проверка пройдена')
else:
 print('Проверка номера не пройдена')