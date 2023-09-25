"""
This is the MAIN SCRIPT for evaluating how pixel-level accuracy metrics
record the error structure between 'truth' and 'model' data across large 
gradients in feature prevalence. 

DESCRIPTION:
    This script generates synthetic 'truth' and 'model' data representing
    features in typical scenes. The user specifies size of the scene
    and the error structure of the model data. The main results are the
    F1-score and normalized MCC for a range of values of feature fraction.
    Results are plotted and saved as a .csv file in the local directory.
    
MODEL SCENARIOS
    1: 'con' assumes a constant feature fraction regardless of truth
    2: 'ind' assumes location of incipient features independent of truth, 
        though matches the scene-level feature fraction
    3: 'ran' assumes a uniform, random error rate regardless of truth, thus 
        it DOES NOT enforce that scene-level fractions match
    4: 'sys' assumes a uniform translational offset of with wrap-around
        boundaries, thus it DOES enforce that scene-level fractions match
    5: 'sys + ran' assumes both random error {3} and systematic error {4}, thus
        it DOES NOT enforce that scene-level fractions match

REQUIREMENTS:
    packages: numpy 1.23.5, scipy 1.10.0, matplotlib 3.7.0
    code: synthetic_bedrock_maps_functions.py       
"""

import numpy as np
import matplotlib.pyplot as plt
import synthetic_feature_maps_functions as gf

## USER INPUTS ##
l = 100             # grid length [pixels]
scl = 10            # incipient feature length [pixels]
sflag = 5           # 1: 'con', 2: 'ind', 3: 'ran', 4: 'sys', 5: 'sys + ran' 
fmap = 0.5          # draw map at bedrock fraction = fmap
con = 1             # uniform FEATURE fraction [only used for sflag: {1}]
off = 1             # offset [pixels] [only used for sflag: {4,5}]
rand = 0.05         # random error rate [only used for sflag: {3,5}]

## INITIALIZE ARRAYS ##
f = np.linspace(start=0.01, stop=0.99, num=99)  # target FEATURE fractions
ft = np.zeros(len(f))                           # true FEATURE fractions
fm = np.zeros(len(f))                           # model FEATURE fractions
F1 = np.zeros(len(f))                           # F1-score
F1a = np.zeros(len(f))                          # F1-score (swapped)
F1macro = np.zeros(len(f))                      # F1-score (macro)
nMCC = np.zeros(len(f))                         # normalized MCC 
TN = np.zeros(len(f))                           # True Negatives 
TP = np.zeros(len(f))                           # True Positives
FP = np.zeros(len(f))                           # False Positives 
FN = np.zeros(len(f))                           # False Negatives
e2a = np.zeros(len(f))                          # Edge to Area Ratio


## MAIN SCRIPT ##
for i in range(len(f)):
    [zt, ft[i]] = gf.generate_grid(leng=l, frac=f[i], seed_no=1, scale=scl) 
    
    ## TRUTH-MODEL SCENARIOS ##
    ## CONSTANT FRACTION - Truth varies but model does not ##
    if sflag==1:
        [zm, fm[i]] = gf.generate_grid(leng=l, frac=con, seed_no=2, scale=scl)
    
    ## FRACTION MATCH - Tor location independent of truth ##
    elif sflag==2:
        [zm, fm[i]] = gf.generate_grid(leng=l, frac=f[i], seed_no=2, scale=scl)      
    
    ## RANDOM ERROR - Error set by constant error frequency ##
    elif sflag==3:
        [zm, fm[i]] = gf.model_rand_err(zt,err=rand,seed_no=2)
    
    ## SYSTEMATIC ERROR - Error set by translational offset  of dn [pixels] ##
    elif sflag==4:
        [zm, fm[i]]  = gf.model_offset(zt,dn=off)
        
    ## COMBINED ERROR - Systematic translaional error plus random error ##
    elif sflag==5:
        [zi, fm[i]]  = gf.model_offset(zt,dn=off)
        [zm, fm[i]]  = gf.model_rand_err(zi,err=rand,seed_no=2)
    
    else:
        print('You have not entered a model scenario.')
    
    ## CALCULATE ACCURACY METRICS ##
    [F1[i], nMCC[i], zclass, TN[i], TP[i], FN[i], FP[i]] = gf.accuracy_metrics(zm,zt)
    F1a[i] = (2*TN[i]) / ((2*TN[i]) + FN[i] + FP[i])
    F1macro[i] = (F1[i]+F1a[i])/2
    e2a[i] = gf.edge_to_area(zm)
        
    ## SAVE CLASSIFIED GRID FOR SPECIFIED BEDROCK FRACTION FOR PLOTTING ##
    check = ft[i] / fmap    
    if check>0.95 and check<1.05:
        zclass_save = zclass
        #print(check)

## CALCULATE ROOT MEAN SQUARED ERROR BETWEEN MODEL AND TRUTH ##
x_sq_err = (ft-fm)**2
x_RMSD = np.sqrt(np.sum(x_sq_err)/len(f))
print(x_RMSD)

## PLOT RESULTS ##
fig1, (ax1, ax2) = plt.subplots(1, 2)    

## PLOT TOP PANEL - Classified map at fraction ~mflag ##
im = ax1.imshow(zclass_save, interpolation=None, cmap=plt.cm.viridis)
cbar = fig1.colorbar(im, ax=ax1, ticks=[1,2,3,4])
cbar.ax.set_yticklabels(['TN', 'FP', 'FN', 'TP'], fontsize=20)

## FORMAT TOP PANEL ##
ax1.set(xlim=(0,l), ylim=(0,l), xlabel='X [m]', ylabel='Y [m]')
ax1.set_xlabel('X [m]', fontsize=26, weight='bold')
ax1.set_ylabel('Y [m]', fontsize=26, weight='bold')
ax1.set_aspect('auto')
#ax1.set_title(f'{mflag*100:0.0f}% bedrock', fontsize=28, style='italic', pad=16)
ax1.text(0.535, 0.07, f'{fmap*100:0.0f}% bedrock', 
         bbox={'fc': 'white', 'pad': 10, 'alpha': 0.9}, 
         fontsize=20, transform=ax1.transAxes)
ax1.tick_params(axis='both', which='major', labelsize=18)

## PLOT BOTTOM PANEL - Accuracy as a function of bedrock fraction ##
ax2.plot(ft, F1, label='F1-score', c = 'k')
ax2.plot(ft, nMCC, label='nMCC', c = 'r')

## FORMAT TOP PANEL ##
ax2.set(xlim=(0,1), ylim=(0,1))
ax2.set_xlabel('Bedrock Fraction', fontsize=26, weight='bold')
ax2.set_ylabel('Accuracy Scores', fontsize=26, weight='bold')
ax2.legend(loc='lower right', frameon=True, fontsize=20)
ax2.tick_params(axis='both', which='major', labelsize=16)

## SET FIGURE SIZE ##
fig1.set_size_inches(16, 6)
plt.show()

## SAVE DATA TO .CSV FILE ##
np.savetxt(f'acc_l_{l}_scl_{scl}_off_{off}.csv', np.c_[fm, F1, nMCC, TN, TP, FN, FP, e2a, F1macro], delimiter=',')

