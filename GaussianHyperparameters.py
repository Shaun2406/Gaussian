# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 16:27:29 2018

@author: smd118
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as plim
import pandas as pd
import sklearn.gaussian_process as gp
plt.close("all")

Kernel_Params = pd.read_csv('C:\\WinPython-64bit-3.5.4.1Qt5\\Gaussian\\Kernel_Params.csv')
#Kernel_Params['Gt'][95] = 1.12
#Kernel_Params['Gt'][210] = 1.12
#plt.figure()
#plt.plot(Kernel_Params['Gt'], Kernel_Params['k_SIt'], 'kx')

Kernel_Params.hist(column='k_SIt', bins = 25)
Kernel_Params.hist(column='k_Pt', bins = 25)
Kernel_Params.hist(column='k_ut', bins = 25)
'''
fig = plt.figure()
H, xedges, yedges  = np.histogram2d(Kernel_Params['Gt'], Kernel_Params['k_SIt'], bins = 20)
H = H.T
ax = fig.add_subplot(111, title='NonUniformImage: interpolated', aspect='equal', xlim=xedges[[0, -1]], ylim=yedges[[0, -1]])
im = plim.NonUniformImage(ax, interpolation='bilinear')
xcenters = (xedges[:-1] + xedges[1:]) / 2
ycenters = (yedges[:-1] + yedges[1:]) / 2
im.set_data(xcenters, ycenters, H)
ax.images.append(im)
plt.show()
'''