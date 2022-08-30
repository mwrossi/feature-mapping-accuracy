# -*- coding: utf-8 -*-
"""
This script contains the FUNCTIONS needed for 'synthetic_bedrock_maps_main.py'

DESCRIPTION:
    The functions here are used to build sythetic 'truth' and 'model' grids.
    Bedrock is placed in the landscape as square tors using different
    assumptions for how error is structured in the scene. It also includes
    functions to calculate accuracy and other grid metrics.
    
FUNCTION DESCRIPTIONS
    1: 'kernel'           Tor centres to bedrock using scipy.signal.convolve2d
    2: 'tor_location'     Iterate over number of tors needed using 'kernel'
    3: 'generate_grid'    Create bedrock map using 'tor_location'
    4: 'model_offset'     Create model grid for translational offset of truth
    5: 'model_rand_err'   Create model grid for random error on truth
    6: 'accuracy_metrics' Calculate TP, FP, FN, TN, F1-score, MCC, nMCC
    7: 'edge_to_area'     Calculate edge to area ratio on given grid

REQUIREMENTS:
    libraries: NumPy, SciPy 

ASSOCIATED MANUSCRIPT:
    Rossi, M.W., in review, Short Communication: Evaluating the accuracy of 
    binary classifiers for geomorphic applications: Earth Surface Dynamics.

last edited by mwrossi on 08.30.2022
"""

import numpy as np
from scipy import signal as sig
from scipy import ndimage as ndi

## BUILD TORS USING 2D CONVOLUTION TOR CENTRES FOR GIVEN SIZE ##
def kernel(ksize,arr):
    k = np.ones((ksize, ksize))
    arrc = sig.convolve2d(arr, k, mode='same', boundary='wrap')
    arrc[arrc >= 1] = 1
    
    return arrc

## BEDROCK MAP USING TOR LOCATIONS FOR A GIVEN BEDROCK FRACTION ##
def tor_location(leng, frac, scale, seed_no):
    z = np.zeros((leng,leng))

    tor_no = round((leng*leng*(1-frac))/(scale**2), 0)
    
    i=1
    ratio=0.
    
    #while i<20:
    while ratio < 0.995 or ratio > 1.005:
        if i==1:
            tor_no = tor_no
        else:
            tor_no = round(tor_no*ratio)
        #print(tor_no)
        tor_dens = tor_no/(leng*leng)
        
        z = np.zeros((leng,leng))
        np.random.seed(seed_no)
        z += np.random.rand(leng,leng)         
        z[z <= (1-tor_dens)] = 0
        z[z > (1-tor_dens)] = 1
        z = kernel(ksize=scale, arr=z) 
        
        ratio = (1-frac)/(0.00000000001+np.sum(z)/(leng**2))
        #print(ratio)
        
        if i==50:
            break

        i+=1
    
    if ratio < 0.995 or ratio > 1.005:
        print('Did not converge in fifty iterations.')
    
    return z

## GENERATE BEDROCK GRID ##
def generate_grid(leng, frac, seed_no, scale):
    # TRUTH grid
    z = np.zeros((leng,leng))
    np.random.seed(seed_no)
    z += np.random.rand(leng,leng)

    z = tor_location(leng,frac,scale,seed_no) 
    frac_actual = np.size(z[z==1])/(np.size(z))
    
    return [z, frac_actual];
    

## OFFSET INPUT GRID BY DN ##
def model_offset(z,dn):
    [m,n]=z.shape
    zn = np.zeros([m,n])
        
    for j in range(n):
        if j<dn:
            zn[:,j] = z[:,(n-dn+j)]
        else:
            zn[:,j] = z[:,j-dn]
        
    frac_actual = np.size(zn[zn==1])/(np.size(zn))
    
    return [zn, frac_actual]            

## MODIFY INPUT GRID USING A CONSTANT ERROR RATE ##
def model_rand_err(z,err,seed_no):
    [m,n]=z.shape
    zn = np.zeros([m,n])
    ze = np.zeros([m,n])
    
    np.random.seed(seed_no)
    ze += np.random.rand(m,n)
    ze[ze>=err]=1    
    ze[ze<err]=0
    ze=1-ze
     
    #print(ze)   
    for i in range(m):
        for j in range(n):
            if ze[i,j]==1:
                zn[i,j]=z[i,j]+2
            else:
                zn[i,j]=z[i,j]

    zn[zn==2] = 1
    zn[zn==3] = 0
    
    frac_actual = np.size(zn[zn==1])/(np.size(zn))
    
    return [zn, frac_actual]            


## CALCULATE F!-SCORE, MCC, and nMCC ##
def accuracy_metrics(zm,zt):
    # Build Confusion Matrix
    zclass = (zm[:]-zt[:])+(3*zt[:])+1
    TN = float(np.count_nonzero(zclass==1))
    TP = float(np.count_nonzero(zclass==4))
    FN = float(np.count_nonzero(zclass==3))
    FP = float(np.count_nonzero(zclass==2))
    
    # F1-score and MCC
    F1 = (2*TP) / ((2*TP) + FP + FN) 
    MCC = ((TP*TN)-(FP*FN)) / np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    nMCC = (MCC+1)/2
    
    return [F1, nMCC, zclass, TN, TP, FN, FP];

## CALCULATE EDGE TO AREA RATIO ON BEDROCK MAP ##
def edge_to_area(zm):
    ker = np.array(([0,1,0],[1,0,1],[0,1,0]))
    area = np.sum(zm)
    edges = np.sum(zm*(4 - ndi.correlate(zm,ker,mode='wrap')))
    ratio = edges / area
    
    return ratio
    
    