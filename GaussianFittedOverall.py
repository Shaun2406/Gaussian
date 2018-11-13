# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 12:22:04 2018

@author: smd118
"""
#from time import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as spnt
import pandas as pd
import sklearn.gaussian_process as gp    
from Patients import patients as Patient_List

plt.close("all")

Kernel_Params = pd.DataFrame()
Kernels = ({'t': [1000, 0.001, 10000], 'ut': [20, 0.001, 2000], 'Pt': [0.02, 0.000001, 20], 'Gt': [0.5, 0.001, 10], 'SIt': [1, 0.001, 1000], 'Patient#': [1, 0.001, 1000]})
Features = ['t', 'ut', 'Gt']
Kernel_Width = []
Kernel_Bound = []
for i in range(len(Features)):
    Kernel_Width.append(Kernels[Features[i]][0])
    Kernel_Bound.append(Kernels[Features[i]][1:])        

GlucOverall = pd.DataFrame()

for i in range(1668):
    GlucData = pd.read_csv('C:\\WinPython-64bit-3.5.4.1Qt5\\Gaussian\\Data\\Data_' + str(i+1) + '.csv')
    GlucData = GlucData[GlucData['SIt'] > 0]
    GlucData = GlucData[GlucData['Gt'] > 0]
    GlucData = GlucData.reset_index()
    #PatientNo = pd.DataFrame({'PatientNo': i*np.ones([len(GlucData), 1])})
    #GlucData = GlucData.merge(PatientNo)
    
    GlucOverall = GlucOverall.append(GlucData)
GlucOverall.to_csv('C:\\WinPython-64bit-3.5.4.1Qt5\\Gaussian\\GlucOverall.csv')
GlucOverall = GlucOverall.reset_index()
GlucOverall = GlucOverall.loc[0:1000]

t = GlucOverall['t'].values.reshape(-1,1)
SI = np.log10(GlucOverall['SIt']).values.reshape(-1, 1)
Gt = np.log10(GlucOverall['Gt']).values.reshape(-1, 1)
    
Input = np.zeros([len(t),0])

for j in range(len(Features)):
    Input = np.hstack([Input, GlucOverall[Features[j]].values.reshape(-1,1)])
kernel = gp.kernels.ConstantKernel(0.5) * gp.kernels.RBF(Kernel_Width, Kernel_Bound)
GPG = gp.GaussianProcessRegressor(kernel=kernel, optimizer = 'fmin_l_bfgs_b', alpha = 0.002, n_restarts_optimizer=5, normalize_y=True)
GPG.fit(Input, SI)

RBF_Params = GPG.kernel_.get_params()['k2']

print('Constant Kernel Value = %.4f' % GPG.kernel_.get_params()['k1__constant_value'], end=" ")
print('RBF Kernel Values =', end=" ")
if len(Features) > 1:
    for k in RBF_Params.length_scale:
        print('%.4f' % k, end=" ")
else:
    print('%.4f' % RBF_Params.length_scale, end=" ")
print()


Patient = 8

GlucData = pd.read_csv('C:\\WinPython-64bit-3.5.4.1Qt5\\Gaussian\\Data\\Data_' + str(Patient) + '.csv')
SIData = pd.read_csv('C:\\WinPython-64bit-3.5.4.1Qt5\\Gaussian\\Data\\SI_' + str(Patient) + '.csv')
t_patient = GlucData['t'].values.reshape(-1,1)
SI_patient = np.log10(GlucData['SIt']).values.reshape(-1, 1)
Gt_patient = np.log10(GlucData['Gt']).values.reshape(-1, 1)
Sample = np.arange(t_patient[0], t_patient[-1], 1).reshape(-1, 1)

Input_Patient = np.zeros([len(t_patient),0])

for j in range(len(Features)):
    Input_Patient = np.hstack([Input_Patient, GlucData[Features[j]].values.reshape(-1,1)])
    if Features[j] != 't':
        Sample = np.hstack([Sample, np.interp(Sample[:,0:1], t_patient[:,0], GlucData[Features[j]].values.reshape(-1,1)[:,0])])

y_pred, sigma = GPG.predict(Sample, return_std=True)


for i in range(4):
    if any(GlucData['Patient'][0] in x for x in Patient_List[i]):
        CV_DataSet = i+1
        CV_Table = np.loadtxt('C:\\WinPython-64bit-3.5.4.1Qt5\\MATLAB Tables\\Table_3D_1H_' + str(CV_DataSet), delimiter=",")
CV_Table = np.delete(CV_Table,0,0)
CV_Lower = np.reshape(CV_Table[:,2],[400,400])
CV_Upper = np.reshape(CV_Table[:,6],[400,400])


bound_5 = spnt.RectBivariateSpline(np.linspace(0.2, 1.4, 400), np.linspace(-8.5, -1.5, 400), CV_Lower)
bound_95 = spnt.RectBivariateSpline(np.linspace(0.2, 1.4, 400), np.linspace(-8.5, -1.5, 400), CV_Upper)
Bounds = [[], []]
Bounds[0] = np.zeros(len(Input_Patient))
Bounds[1] = np.zeros(len(Input_Patient))
for i in range(len(Input_Patient)):
    Bounds[0][i] = bound_5(Gt_patient[i], SI_patient[i])
    Bounds[1][i] = bound_95(Gt_patient[i], SI_patient[i])
    
Bounds[0] = np.repeat(Bounds[0],2)
Bounds[1] = np.repeat(Bounds[1],2)

A = ((t_patient-t_patient[0])/10).astype(int)
if A[-1] == len(Sample):
    A[-1] = A[-1] - 1
SI_pred = y_pred[A[:,0]]
Sigma_pred = sigma[A[:,0]]
Score = np.sum(abs(SI_patient-SI_pred)[:,0] < 1.645*Sigma_pred)/len(SI_patient)*100
print(Score)

t_predict = np.arange(t_patient[0], t_patient[-1], 1).reshape(-1, 1)
Score = np.sum(abs(np.log10(SIData['SI'].values[t_predict-60])-y_pred[:,0].reshape(-1,1)) < 1.645*sigma.reshape(-1,1))/len(t_predict)*100
print(str('%.2f' % Score) + '% of points are within the 90% credible interval bounds')

y_pred_fwd = np.zeros(len(y_pred))
sigma_fwd = np.zeros(len(sigma))

kernel_out = gp.kernels.ConstantKernel(GPG.kernel_.get_params()['k1__constant_value']) * gp.kernels.RBF(RBF_Params.length_scale)

for i in range(0,len(Input_Patient)-1):
    GPG_A = gp.GaussianProcessRegressor(kernel=kernel_out, optimizer = None, normalize_y = True, alpha = 0.001)
    GPG_A.fit(Input_Patient[0:i+1,:], SI_patient[0:i+1,:])
    A, B = GPG_A.predict(Sample[int(Input_Patient[i,0]-t_predict[0]):int(Input_Patient[i+1,0]-t_predict[0]),:], return_std = True)
    y_pred_fwd[int(t_patient[i]-t_patient[0]):int(t_patient[i+1]-t_patient[0])] = A[0,:]
    sigma_fwd[int(t_patient[i]-t_patient[0]):int(t_patient[i+1]-t_patient[0])] = B
    
Score_fwd = np.sum(abs(np.log10(SIData['SI'].values[t_predict-60])-y_pred_fwd.reshape(-1,1)) < 1.645*sigma_fwd.reshape(-1,1))/len(t_predict)*100
print(str('%.2f' % Score_fwd) + '% of points are within the 90% forward credible interval bounds')

t_plot = np.repeat(t_patient,2)
t_plot = np.delete(t_plot,0)
t_plot = np.append(t_plot,t_plot[-1])
plt.figure()
plt.plot(Sample[:,0:1], 10**y_pred, color="#000099", linestyle = '--')
#plt.plot(SIData['t'], SIData['SI'], 'g--')
plt.plot(t_patient, 10**SI_patient, 'kx')
plt.gca().fill_between((t_plot).flat, (10**Bounds[0]).flat, (10**Bounds[1]).flat, color="#dddddd")
plt.gca().fill_between(Sample[:,0:1].flat, 10**(y_pred_fwd.flat-1.645*sigma_fwd), 10**(y_pred_fwd.flat+1.645*sigma_fwd), color="#000099", alpha=0.7)            
plt.gca().fill_between(Sample[:,0:1].flat, 10**(y_pred.flat-1.645*sigma), 10**(y_pred.flat+1.645*sigma), color="#999999", alpha=0.7)

plt.title('GP fit of SI')
plt.ylabel('SI, L/mU/min')
plt.xlabel('t, minutes')
         
plt.figure()
plt.plot(Sample[:,0:1],y_pred, color="#000099", linestyle = '--')
#plt.plot(SIData['t'], np.log10(SIData['SI']), color="#990099", linestyle = '--')
plt.plot(t_patient, SI_patient, 'kx')
plt.gca().fill_between((t_plot).flat, (Bounds[0].flat), (Bounds[1].flat), color="#dddddd")
plt.gca().fill_between(Sample[:,0:1].flat, (y_pred_fwd.flat-1.645*sigma_fwd), (y_pred_fwd.flat+1.645*sigma_fwd), color="#000099", alpha=0.7)    
plt.gca().fill_between(Sample[:,0:1].flat, (y_pred.flat-1.645*sigma), (y_pred.flat+1.645*sigma), color="#999999", alpha=0.7)
      
plt.title('GP fit of SI')
plt.ylabel('log(SI)')
plt.xlabel('t, minutes')
#Kernel_Params.to_csv('C:\\WinPython-64bit-3.5.4.1Qt5\\Gaussian\\Kernel_Params_Base.csv')