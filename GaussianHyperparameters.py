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
import scipy.stats as stats
plt.close("all")

Kernel_Params = pd.read_csv('C:\\WinPython-64bit-3.5.4.1Qt5\\Gaussian\\Kernel_Params_Base.csv')
#Kernel_Params['Gt'][95] = 1.12
#Kernel_Params['Gt'][210] = 1.12
#plt.figure()
#plt.plot(Kernel_Params['Gt'], Kernel_Params['k_SIt'], 'kx')

for i in range(len(Kernel_Params)):
    if Kernel_Params['k_Pt'][i] == 0.02:
        Kernel_Params['k_Pt'][i] = 20

Kernel_Params['k_SIt'] = np.log10(Kernel_Params['k_SIt'])
Kernel_Params['k_ut'] = np.log10(Kernel_Params['k_ut'])
Kernel_Params['k_Pt'] = np.log10(Kernel_Params['k_Pt'])
Kernel_Params['k_constant'] = np.log10(Kernel_Params['k_constant'])

Kernel_Params.hist(column='k_SIt', bins = 100)
Kernel_Params.hist(column='k_ut', bins = 100)
Kernel_Params.hist(column='k_Pt', bins = 50)
Kernel_Params.hist(column='k_constant', bins = 100)
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

SI_data = Kernel_Params['k_SIt']
SI_data = SI_data[SI_data < 4]
SI_data = SI_data[SI_data > 1]
SI_model = stats.norm.fit(SI_data)
x_SI = np.linspace(-1,5,100)
pdf_fitted_SI = stats.norm.pdf(x_SI, loc = SI_model[0], scale = SI_model[1])


plt.figure(1)
plt.plot(x_SI,pdf_fitted_SI*125,'r-')
print(str("%.2f"%np.mean(Kernel_Params['Score'])) + '% of points within confidence limits, on average')

u_data = Kernel_Params['k_ut']
u_data = u_data[u_data < 3.3]
u_data = u_data[u_data > 0.5]

u_model = stats.norm.fit(u_data)
x_u = np.linspace(0, 3.5, 100)
pdf_fitted_u = stats.norm.pdf(x_u, loc = u_model[0], scale = u_model[1])

plt.figure(2)
plt.plot(x_u,pdf_fitted_u*90,'r-')

p_data = Kernel_Params['k_Pt']
p_data = p_data[p_data < 1.3]
p_data = p_data[p_data > -3]
p_model = stats.norm.fit(p_data, loc = 0)
x_p = np.linspace(-5,1.5,100)
pdf_fitted_p = stats.norm.pdf(x_p, loc = p_model[0], scale = p_model[1])

plt.figure(3)
plt.plot(x_p,pdf_fitted_p*150,'r-')

c_data = Kernel_Params['k_constant']
c_data = c_data[c_data < 1]
c_data = c_data[c_data > -3]
c_model = stats.norm.fit(c_data, loc = 0)
x_c = np.linspace(-3,1,100)
pdf_fitted_c = stats.norm.pdf(x_c, loc = c_model[0], scale = c_model[1])

plt.figure(4)
plt.plot(x_c,pdf_fitted_c*110,'r-')

#STAR2015    1 -  544
#STAR      545 - 1227
#SPRINT   1228 - 1620
#GYULA    1621 - 1668