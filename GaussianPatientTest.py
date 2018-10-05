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

Kernel_Params = pd.DataFrame()
Kernels = ({'t': [1000, 100, 10000], 'ut': [1, 0.001, 1000], 'Pt': [0.1, 0.000001, 1], 'Gt': [0.5, 0.001, 10], 'SIt': [1, 0.001, 1000]})
Features = ['t', 'ut', 'Pt']
Kernel_Width = []
Kernel_Bound = []
for i in range(len(Features)):
    Kernel_Width.append(Kernels[Features[i]][0])
    Kernel_Bound.append(Kernels[Features[i]][1:])        

for i in range(544):
    print(i)
    GlucData = pd.read_csv('C:\\WinPython-64bit-3.5.4.1Qt5\\Gaussian\\Data\\Data_' + str(i) + '.csv')
    SIData = pd.read_csv('C:\\WinPython-64bit-3.5.4.1Qt5\\Gaussian\\Data\\SI_' + str(i) + '.csv')
    GlucData = GlucData[GlucData['SIt'] > 0]
    if GlucData.shape[0] <= 1:
        continue

    t = GlucData['t'].values.reshape(-1,1)

    SI = np.log10(GlucData['SIt']).values.reshape(-1, 1)
    GlucData['Gt'] = np.log10(GlucData['Gt'])
    Input = np.zeros([len(t),0])
    Sample = np.arange(t[0], t[-1], 10).reshape(-1, 1)
    for j in range(len(Features)):
        Input = np.hstack([Input, GlucData[Features[j]].values.reshape(-1,1)])
        if Features[j] != 't':
            Sample = np.hstack([Sample, np.interp(Sample[:,0:1], t[:,0], GlucData[Features[j]].values.reshape(-1,1)[:,0])])
    kernel = gp.kernels.ConstantKernel(0.5) * gp.kernels.RBF(Kernel_Width, Kernel_Bound)
    GPG = gp.GaussianProcessRegressor(kernel=kernel, optimizer = 'fmin_l_bfgs_b', alpha = 0.02, n_restarts_optimizer=20, normalize_y=True)

    GPG.fit(Input, SI)
    RBF_Params = GPG.kernel_.get_params()['k2']
    '''
    print('Constant Kernel Value = %.4f' % GPG.kernel_.get_params()['k1__constant_value'], end=" ")
    print('RBF Kernel Values =', end=" ")
    if len(Features) > 1:
        for k in RBF_Params.length_scale:
            print('%.4f' % k, end=" ")
    else:
        print('%.4f' % RBF_Params.length_scale, end=" ")
    print()
    '''
    y_pred, sigma = GPG.predict(Sample, return_std=True)
    
    A = ((t-t[0])/10).astype(int)
    if A[-1] == len(Sample):
        A[-1] = A[-1] - 1
    SI_pred = y_pred[A[:,0]]
    Sigma_pred = sigma[A[:,0]]
    Score = np.sum(abs(SI-SI_pred)[:,0] < 1.645*Sigma_pred)/len(SI)*100
    
    Kernel_Result= pd.DataFrame({'k_SIt': RBF_Params.length_scale[0], 'k_ut': RBF_Params.length_scale[1], 'k_Pt': RBF_Params.length_scale[2], 'Score': Score, 'Gt': np.mean(GlucData['Gt']), 'SI': np.mean(GlucData['SIt']), 't': t[-1], 'Gender': GlucData['Gender'].values[0], 'Operative': GlucData['Operative'].values[0]}, index = [i])
    Kernel_Params = Kernel_Params.append(Kernel_Result)
    '''
    plt.figure()
    plt.plot(Sample[:,0:1], 10**y_pred, 'b--')
    plt.plot(SIData['t'], SIData['SI'], 'g--')
    plt.plot(t, 10**SI, 'kx')
    plt.gca().fill_between(Sample[:,0:1].flat, 10**(y_pred.flat-1.645*sigma), 10**(y_pred.flat+1.645*sigma), color="#dddddd")
    plt.title('GP fit of SI')
    plt.ylabel('SI, L/mU/min')
    plt.xlabel('t, minutes')
             
    plt.figure()
    plt.plot(Sample[:,0:1],y_pred, 'b--')
    plt.plot(SIData['t'], np.log10(SIData['SI']), 'g--')
    plt.plot(t, SI, 'kx')
    plt.gca().fill_between(Sample[:,0:1].flat, (y_pred.flat-1.645*sigma), (y_pred.flat+1.645*sigma), color="#dddddd")        
    plt.title('GP fit of SI')
    plt.ylabel('log(SI)')
    plt.xlabel('t, minutes')
    '''
#Kernel_Params.to_csv('C:\\WinPython-64bit-3.5.4.1Qt5\\Gaussian\\Kernel_Params.csv')