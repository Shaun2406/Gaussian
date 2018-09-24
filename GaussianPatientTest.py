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
SIData = pd.read_csv('C:\\WinPython-64bit-3.5.4.1Qt5\\Gaussian\\SIData.csv')

Kernels = ({'t': [1000, 1, 10000], 'ut': [1, 0.001, 1000], 'Pt': [0.1, 0.000001, 1], 'Gt': [0.5, 0.001, 10], 'SIt': [1, 0.001, 1000]})
t = GlucData['t'].values.reshape(-1,1)
Features = ['t', 'ut']
Kernel_Width = []
Kernel_Bound = []

SI = np.log10(GlucData['SIt']).values.reshape(-1, 1)
GlucData['Gt'] = np.log10(GlucData['Gt'])
Input = np.zeros([len(t),0])
Sample = np.linspace(250, 7500, 1000).reshape(-1, 1)
for i in range(len(Features)):
    Kernel_Width.append(Kernels[Features[i]][0])
    Kernel_Bound.append(Kernels[Features[i]][1:])
    Input = np.hstack([Input, GlucData[Features[i]].values.reshape(-1,1)])
    if Features[i] != 't':
        Sample = np.hstack([Sample, np.interp(Sample[:,0:1], t[:,0], GlucData[Features[i]].values.reshape(-1,1)[:,0])])
kernel = gp.kernels.ConstantKernel(0.5) * gp.kernels.RBF(Kernel_Width, Kernel_Bound)
#kernel = gp.kernels.ConstantKernel(np.sqrt(0.047250314695719962)) * gp.kernels.RBF([385.66955375, 150.21253574])
GPG = gp.GaussianProcessRegressor(kernel=kernel, optimizer = 'fmin_l_bfgs_b', alpha = 0.02, n_restarts_optimizer=10, normalize_y=True)
#'fmin_l_bfgs_b'

#Input = Input[0:15,:]
GPG.fit(Input, SI)
RBF_Params = GPG.kernel_.get_params()['k2']
print('Constant Kernel Value = %.4f' % GPG.kernel_.get_params()['k1__constant_value'])
print('RBF Kernel Values =', end=" ")
if len(Features) > 1:
    for i in RBF_Params.length_scale:
        print('%.4f' % i, end=" ")
else:
    print('%.4f' % RBF_Params.length_scale, end=" ")

y_pred, sigma = GPG.predict(Sample, return_std=True)

plt.figure()
plt.plot(Sample[:,0:1], 10**y_pred, 'b--')
plt.plot(SIData['t'], SIData['SI'], 'g--')
plt.plot(t, 10**SI, 'kx')
plt.gca().fill_between(Sample[:,0:1].flat, 10**(y_pred.flat-1.645*sigma), 10**(y_pred.flat+1.645*sigma), color="#dddddd")
plt.title('GP fit of SI')
plt.ylabel('SI, L/mU/min')
plt.xlabel('t, minutes')
plt.ylim([0, 0.002])
        
plt.figure()
plt.plot(Sample[:,0:1],y_pred, 'b--')
plt.plot(SIData['t'], np.log10(SIData['SI']), 'g--')
plt.plot(t, SI, 'kx')
plt.gca().fill_between(Sample[:,0:1].flat, (y_pred.flat-1.645*sigma), (y_pred.flat+1.645*sigma), color="#dddddd")        
plt.title('GP fit of SI')
plt.ylabel('log(SI)')
plt.xlabel('t, minutes')
plt.ylim([-4.5, -2.75])

