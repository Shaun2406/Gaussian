# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 12:22:04 2018

@author: smd118
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.gaussian_process as gp
plt.close("all")

GlucData = pd.read_csv('C:\\WinPython-64bit-3.5.4.1Qt5\\Gaussian\\GlucPatient.csv')
kernel = gp.kernels.ConstantKernel(0.5) * gp.kernels.RBF([1000, 1, 100])
GPG = gp.GaussianProcessRegressor(kernel=kernel, optimizer = 'fmin_l_bfgs_b',alpha = 0.016, n_restarts_optimizer=10, normalize_y=True)
t = GlucData['t'].values.reshape(-1, 1)
G = np.log10(GlucData['Gt']).values.reshape(-1, 1)
I = GlucData['ut'].values.reshape(-1, 1)

INP = np.hstack([t, G, I])
SI = np.log10(GlucData['SIt']).values.reshape(-1, 1)

SI1 = np.log10(GlucData['SIt+1']).values.reshape(-1, 1)

GPG.fit(INP,SI)

t_sample = np.linspace(250,7500,1000).reshape(-1, 1)
G_sample = np.interp(t_sample[:,0], t[:,0], G[:,0]).reshape(-1, 1)
I_sample = np.interp(t_sample[:,0], t[:,0], I[:,0]).reshape(-1, 1)

SAM = np.hstack([t_sample, G_sample, I_sample])

y_pred, sigma = GPG.predict(SAM, return_std=True)

plt.figure()
plt.plot(t_sample, 10**y_pred, 'b--')
plt.plot(t, 10**SI, 'kx')
plt.plot(t+60, 10**SI1, 'gx')
plt.gca().fill_between(t_sample.flat, 10**(y_pred.flat-1.645*sigma), 10**(y_pred.flat+1.645*sigma), color="#dddddd")
plt.title('GP fit of SI')
plt.ylabel('SI, L/mU/min')
plt.xlabel('t, minutes')
        
plt.figure()
plt.plot(t_sample,y_pred, 'b--')
plt.plot(t, SI, 'kx')
plt.plot(t+60, SI1, 'gx')
plt.gca().fill_between(t_sample.flat, (y_pred.flat-1.645*sigma), (y_pred.flat+1.645*sigma), color="#dddddd")        
plt.title('GP fit of SI')
plt.ylabel('log(SI)')
plt.xlabel('t, minutes')
#plt.plot([875, 875], [-3.75, -3.25], 'k--')
#plt.plot([2900, 2900], [-3.75, -3.25], 'k--')
#plt.plot([4925, 4925], [-3.75, -3.25], 'k--')

