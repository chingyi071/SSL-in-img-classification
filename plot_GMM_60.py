#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 13:30:57 2019

@author: shadow_iris
"""

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import csv
from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 6), dpi=60, facecolor='w', edgecolor='k')


with open('/Users/shadow_iris/Desktop/SSL4/report/SSL-clf-DNN-slt-GMM6050-match-label600.txt') as csv_file1:
#csv_file=f.read()

  csv_reader1= csv.reader(csv_file1, delimiter=' ')
    
  Test_accuracy1=[]
  New_label_add1=[]
  New_label_correct1=[]
  New_label_acc1=[]
  x1=[]
  x1.append(600)
  x1_value=600
  for item in csv_reader1:
      if csv_reader1.line_num == 1:
        continue
      if csv_reader1.line_num == 21:
        Test_accuracy1.append(float(item[5]))
      else:
      #Train_labeled_loss.append(item[0])
       Test_accuracy1.append(item[5])
       New_label_add1.append(item[7])
       New_label_correct1.append(item[6])
#        print('i',int(item[7]))
       x1_value +=float(item[7])
       x1.append(x1_value)
       corr=float(item[6])
       add=float(item[7])
       if(add==0):
          New_label_acc1.append(0)
       else:
          New_label_acc1.append(corr / add)
      
       #result[item[0]] = item[1]
  print('accn',New_label_acc1)
  print('x1',x1)
  csv_file1.close()


y1=Test_accuracy1

color1 = []
for i in range(len(New_label_acc1)):
    if float(New_label_acc1[i]) <=1 and float(New_label_acc1[i])>0.9:
        color1.append('#000080') #navy
    if float(New_label_acc1[i])<=0.9 and float(New_label_acc1[i])>0.8:
        color1.append('#006400') #darkgreen
    if float(New_label_acc1[i])<=0.8 and float(New_label_acc1[i])>0.7:
        color1.append('#FF7F50') #coral
#    if float(New_label_acc1[i])<=0.7 and float(New_label_acc1[i])>0.6:
#        color1.append('#A52A2A')
    else:
        color1.append('#DC143C') #crimson

#print(x)
#print(y)
#print('--------------------------------------')
points1 = np.array([x1, y1]).T.reshape(-1, 1, 2)
#print(points)
#print('--------------------------------------')
segments1 = np.concatenate([points1[:-1], points1[1:]], axis=1)
#print(segments)
lc1 = LineCollection(segments1, linewidths=2.0, color=color1,linestyle='-')

ax = plt.axes()
ax.set_xlim(600, 2500)
ax.set_ylim(0.85, 0.88)
ax.add_collection(lc1)
#plt.plot(x1,y1)

#######################################6010#########################
with open('/Users/shadow_iris/Desktop/SSL4/report/SSL-clf-DNN-slt-GMM6093-match-label600.txt') as csv_file7:
#csv_file=f.read()

  csv_reader7= csv.reader(csv_file7, delimiter=' ')
    

  #Train_labeled_loss=[]
  Test_accuracy7=[]
  #Test_accuracy7.append(0.85)
  New_label_add7=[]
  New_label_correct7=[]
  New_label_acc7=[]
  x7=[]
  x7.append(600)
  x7_value=600
  for item in csv_reader7:
      if csv_reader7.line_num == 1:
        continue
      if csv_reader7.line_num == 21:
        Test_accuracy7.append(float(item[5]))
      else:
      #Train_labeled_loss.append(item[0])
       Test_accuracy7.append(float(item[5]))
       New_label_add7.append(item[7])
       New_label_correct7.append(item[6])
       x7_value +=float(item[7])
       x7.append(x7_value)
       corr=float(item[6])
       add=float(item[7])
       if(add==0):
          New_label_acc7.append(0)
       else:
          New_label_acc7.append(corr / add)
      
       #result[item[0]] = item[1]
  #print(New_label_acc7)
  print('x7',x7)
  #print('t7',Test_accuracy7)
  csv_file7.close()


y7=Test_accuracy7

color7 = []
for i in range(len(New_label_acc7)):
    if float(New_label_acc7[i]) <=1 and float(New_label_acc7[i])>0.9:
        color7.append('#000080') #navy
    if float(New_label_acc7[i])<=0.9 and float(New_label_acc7[i])>0.8:
        color7.append('#006400') #darkgreen
    if float(New_label_acc7[i])<=0.8 and float(New_label_acc7[i])>0.7:
        color7.append('#FF7F50') #coral
#    if float(New_label_acc1[i])<=0.7 and float(New_label_acc1[i])>0.6:
#        color1.append('#A52A2A')
    else:
        color1.append('#DC143C') #crimson

#print(x)
#print(y)
#print('--------------------------------------')
points7 = np.array([x7, y7]).T.reshape(-1, 1, 2)
#print(points)
#print('--------------------------------------')
segments7 = np.concatenate([points7[:-1], points7[1:]], axis=1)
#print(segments)
lc7 = LineCollection(segments7, linewidths=2.0, color=color7,linestyle=':')


ax.add_collection(lc7)

##############################################4015###########################
with open('/Users/shadow_iris/Desktop/SSL4/report/SSL-clf-DNN-slt-GMM6075-match-label600.txt') as csv_file2:
#csv_file=f.read()

  csv_reader2= csv.reader(csv_file2, delimiter=' ')
    

  #Train_labeled_loss=[]
  Test_accuracy2=[]
  #Test_accuracy2.append(0.85)
  New_label_add2=[]
  New_label_correct2=[]
  New_label_acc2=[]
  x2=[]
  x2.append(600)
  x2_value=600
  for item in csv_reader2:
      if csv_reader2.line_num == 1:
        continue
      if csv_reader2.line_num == 21:
        Test_accuracy2.append(float(item[5]))
      else:
      #Train_labeled_loss.append(item[0])
       Test_accuracy2.append(float(item[5]))
       New_label_add2.append(item[7])
       New_label_correct2.append(item[6])
       x2_value +=float(item[7])
       x2.append(x2_value)
       corr=float(item[6])
       add=float(item[7])
       if(add==0):
          New_label_acc2.append(0)
       else:
          New_label_acc2.append(corr / add)
      
       #result[item[0]] = item[1]
  #print(New_label_acc2)
  print('x2',x2)
  #print('t2',Test_accuracy2)
  csv_file2.close()


y2=Test_accuracy2

color2 = []
for i in range(len(New_label_acc2)):
    if float(New_label_acc2[i]) <=1 and float(New_label_acc2[i])>0.9:
        color2.append('#000080') #navy
    if float(New_label_acc2[i])<=0.9 and float(New_label_acc2[i])>0.8:
        color2.append('#006400') #darkgreen
    if float(New_label_acc2[i])<=0.8 and float(New_label_acc2[i])>0.7:
        color2.append('#FF7F50') #coral
#    if float(New_label_acc1[i])<=0.7 and float(New_label_acc1[i])>0.6:
#        color1.append('#A52A2A')
    else:
        color2.append('#DC143C') #crimson

#print(x)
#print(y)
#print('--------------------------------------')
points2 = np.array([x2, y2]).T.reshape(-1, 1, 2)
#print(points)
#print('--------------------------------------')
segments2 = np.concatenate([points2[:-1], points2[1:]], axis=1)
#print(segments)
lc2 = LineCollection(segments2, linewidths=2.0, color=color2,linestyle='-.')


#ax.set_xlim(600, 1500)
#ax.set_ylim(0.8, 1)
ax.add_collection(lc2)
#plt.plot(x2,y2)

#########################4020#################################################
with open('/Users/shadow_iris/Desktop/SSL4/report/SSL-clf-DNN-slt-GMM6087-match-label600.txt') as csv_file3:
#csv_file=f.read()

  csv_reader3= csv.reader(csv_file3, delimiter=' ')
    

  #Train_labeled_loss=[]
  Test_accuracy3=[]
  #Test_accuracy3.append(0.85)
  New_label_add3=[]
  New_label_correct3=[]
  New_label_acc3=[]
  x3=[]
  x3.append(600)
  x3_value=600
  for item in csv_reader3:
      if csv_reader3.line_num == 1:
        continue
      if csv_reader3.line_num == 21:
        Test_accuracy3.append(float(item[5]))
      else:
      #Train_labeled_loss.append(item[0])
       Test_accuracy3.append(float(item[5]))
       New_label_add3.append(item[7])
       New_label_correct3.append(item[6])
       x3_value +=float(item[7])
       x3.append(x3_value)
       corr=float(item[6])
       add=float(item[7])
       if(add==0):
          New_label_acc3.append(0)
       else:
          New_label_acc3.append(corr / add)
      
       #result[item[0]] = item[1]
  #print(New_label_acc3)
  print('X3',x3)
  print(Test_accuracy3)
  csv_file3.close()


y3=Test_accuracy3

color3 = []
for i in range(len(New_label_acc3)):
    if float(New_label_acc3[i]) <=1 and float(New_label_acc3[i])>0.9:
        color3.append('#000080') #navy
    if float(New_label_acc3[i])<=0.9 and float(New_label_acc3[i])>0.8:
        color3.append('#006400') #darkgreen
    if float(New_label_acc3[i])<=0.8 and float(New_label_acc3[i])>0.7:
        color3.append('#FF7F50') #coral
#    if float(New_label_acc1[i])<=0.7 and float(New_label_acc1[i])>0.6:
#        color1.append('#A52A2A')
    else:
        color3.append('#DC143C') #crimson

#print(x)
#print(y)
#print('--------------------------------------')
points3 = np.array([x3, y3]).T.reshape(-1, 1, 2)
#print(points)
#print('--------------------------------------')
segments3 = np.concatenate([points3[:-1], points3[1:]], axis=1)
#print(segments)
lc3 = LineCollection(segments3, linewidths=2.0, color=color3,linestyle='--')


ax.add_collection(lc3)
#plt.plot(x2,y2)
plt.xlabel('Sample number')
plt.ylabel('Accuracy')
plt.title('Percentile threshold (prob threshold 60%)',fontsize='xx-large',fontweight='bold')

plt.show()