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
figure(num=None, figsize=(6,4), dpi=120, facecolor='w', edgecolor='k')

ax = plt.axes()
ax.set_xlim(600, 2500)
ax.set_ylim(85, 88)

final_points = []
final_dotx   = []
final_doty   = []

base_filename = 'report/SSL-clf-DNN-slt-GMM60%d-match-label600.txt'
for param, linestyle in zip([50, 75, 87, 93], [':','-.','--', '-']):

  filename = base_filename % param
  with open(filename) as csv_file1:
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
    print('accn(',len(New_label_acc1),')',New_label_acc1)
    print('x1(',len(x1),')',x1)
    csv_file1.close()


  y1=[float(y)*100 for y in Test_accuracy1]

  color1 = []
  for i in range(len(New_label_acc1)):
      if float(New_label_acc1[i]) <=1 and float(New_label_acc1[i])>0.9:
          color1.append('#000080') #navy
      elif float(New_label_acc1[i])<=0.9 and float(New_label_acc1[i])>0.8:
          color1.append('#006400') #darkgreen
      elif float(New_label_acc1[i])<=0.8 and float(New_label_acc1[i])>0.7:
          color1.append('#FF7F50') #coral
  #    if float(New_label_acc1[i])<=0.7 and float(New_label_acc1[i])>0.6:
  #        color1.append('#A52A2A')
      else:
          color1.append('#DC143C') #crimson 
      print("New_label_acc1(%d) = " % i, New_label_acc1[i], color1[-1])
  assert( len(color1) == len(New_label_acc1))

  #print(x)
  #print(y)
  #print('--------------------------------------')
  points1 = np.array([x1, y1]).T.reshape(-1, 1, 2)
  # print("points1(", points1.shape, "):", points1)
  
  # print("test acc = ", points1[:,:,1].flatten())
  # print('accn(',len(New_label_acc1),')',New_label_acc1)

  #print('--------------------------------------')
  segments1 = np.concatenate([points1[:-1], points1[1:]], axis=1)
  #print(segments)

  lc1 = LineCollection(segments1, linewidths=2.0, color=color1,linestyle=linestyle, label='perc th = '+str(param))

  ax.add_collection(lc1)
  final_points.append((float(points1[-1][0][0]), float(points1[-1][0][1])+0.3, "th(%d): %.1lf%%(%d samples)" % (param,float(points1[-1][0][1]),int(points1[-1][0][0]))))
  final_dotx.append( float(points1[-1][0][0]) )
  final_doty.append( float(points1[-1][0][1]) )

ax.scatter(final_dotx, final_doty, color='black')
  # plt.show()  
  # xxx

last_pty = 0
for ptx, pty, text in final_points:
  cur_pty = max( pty, last_pty+0.2)
  ax.text( ptx, cur_pty, text )
  last_pty = cur_pty

plt.xlabel('Sample number')
plt.ylabel('Accuracy(%)')

plt.title('Percentile threshold (prob th 60%)',fontsize='xx-large',fontweight='bold')
plt.legend()

plt.show()
