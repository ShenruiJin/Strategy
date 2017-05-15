# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 15:09:57 2016

@author: asanjari
"""

#first array is always the underlying
#plt.plot(arrayWeight)   #blue is the weight ratio
#plt.plot(100*arrayM1)  #green is the underlying
#plt.plot(100*arrayM2)  #red is the Implied

#obviously you going to see the green line very close to zero all the time.


import numpy as np

nonzeroBULL=np.nonzero(newStrat['UBull performance without W'][:])[0]

##########        BULL
nBull=100 #period for updating weights
N=len(nonzeroBULL)

K=N/nBull
arrayWeightBull=np.zeros(K)
arrayM1Bull=np.zeros(K)
arrayM2Bull=np.zeros(K)

arrayalphabull=np.zeros(K)
arraybetabull=np.zeros(K)

for k in range(K):
    
    i=k*nBull
    j=(k+1)*nBull 
    
    array1=newStrat['UBull performance without W'][nonzeroBULL]
    array2=newStrat['IBull performance without W'][nonzeroBULL]
    
    array1=array1[i:j]
    array2=array2[i:j]
    
    m1=np.mean(array1)
    m2=np.mean(array2)
    
    cov=np.cov(array1,array2)[0,1]
    va=np.cov(array1,array2)[0,0]
    vb=np.cov(array1,array2)[1,1]
    
    #print m1 ,'  ' ,m2,'   ',va,'    ',vb,'    ',cov
    arrayM1Bull[k]=m1
    arrayM2Bull[k]=m2
    
    arrayalphabull[k]=(m1*vb-m2*cov)/(va*vb - cov*cov)
    arraybetabull[k]=(m2*va-m1*cov)/(va*vb - cov*cov)
    
    arrayWeightBull[k]=(m1*vb-m2*cov)/(m2*va-m1*cov)

import matplotlib.pyplot as plt


plt.plot(arrayalphabull[1:])
plt.plot(arraybetabull[1:])

plt.plot(arrayWeightBull[1:])
plt.plot(1000*arrayM1Bull[1:])
plt.plot(1000*arrayM2Bull[1:])

#in bull market alpha tend to be more positive
#in the bull market beta tend to be more negative


nonzeroBEAR=np.nonzero(newStrat['UBear performance without W'][:])[0]
##########        BEAR
nBear=25 #period for updating weights

N=len(nonzeroBEAR) 

K=N/nBear
arrayWeightBear=np.zeros(K)
arrayM1Bear=np.zeros(K)
arrayM2Bear=np.zeros(K)

arrayalphabear=np.zeros(K)
arraybetabear=np.zeros(K)

for k in range(K):
    i=k*nBear
    j=(k+1)*nBear     
    
    array1=newStrat['UBear performance without W'][nonzeroBEAR]
    array2=newStrat['IBear performance without W'][nonzeroBEAR]
    
    array1=array1[i:j]
    array2=array2[i:j]
    
    m1=np.mean(array1)
    m2=np.mean(array2)
    
    cov=np.cov(array1,array2)[0,1]
    va=np.cov(array1,array2)[0,0]
    vb=np.cov(array1,array2)[1,1]
    
    #print m1 ,'  ' ,m2,'   ',va,'    ',vb,'    ',cov
    arrayM1Bear[k]=m1
    arrayM2Bear[k]=m2
    
    arrayalphabear[k]=(m1*vb-m2*cov)/(va*vb - cov*cov)
    arraybetabear[k]=(m2*va-m1*cov)/(va*vb - cov*cov)
    
    arrayWeightBear[k]=(m1*vb-m2*cov)/(m2*va-m1*cov)


plt.plot(arrayalphabear[1:])
plt.plot(arraybetabear[1:])

plt.plot(arrayWeightBear[1:])   #blue
plt.plot(1000*arrayM1Bear[1:])  #green
plt.plot(1000*arrayM2Bear[1:])  #red


#in bear market alpha tend to be positive and beta close to zero







import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from os import chdir
import os
import sqlite3
import scipy

from win32com.client import Dispatch


#**********************************
# IMPORT NYC ENGINEERING LIBRAIRIES
#**********************************

import sys

os.chdir(r'K:\ED_ExcelTools\Transfert\Structuring\Proprietary Indices\Python Script\US Structuring Libraries')


from  DB_functions import *
from  class_CalendarUS import *
from  CBOE_Rolled_Contracts import *
from  class_Strategy import *











os.chdir(r'K:\ED_ExcelTools\Transfert\Structuring\Proprietary Indices\Python Script\US Structuring Libraries')

from Implied_vs_Historical_Volatility_Strategy import *

Universe="US"

dctContracts={}
dctWeights={}

if Universe == "TargetBeta":
    """ Signal Calculus"""
    dctContracts['Implied Vol'] = 'VIX'
    dctContracts['Underlying Index'] = 'SPX'
    """ Trading Long """
    dctContracts['Underlying Long'] = 'SPX'
    dctContracts['Implied Vol Long'] = 'UX1R'
    dctWeights['Underlying Long'] = 1.0
    dctWeights['Implied Vol Long'] = 0
    """Trading Short"""
    dctContracts['Underlying Short'] = 'ES1R'
    dctContracts['Implied Vol Short'] = 'UX1R'
    dctWeights['Underlying Short'] =0.0
    dctWeights['Implied Vol Short'] =0.5
if Universe == "US":
    """ Signal Calculus"""
    dctContracts['Implied Vol'] = 'VIX'
    dctContracts['Underlying Index'] = 'SPX'
    """ Trading Long """
    dctContracts['Underlying Long'] = 'SPX'
    dctContracts['Implied Vol Long'] = 'UX1R'
    dctWeights['Underlying Long'] = 1.0
    dctWeights['Implied Vol Long'] = 0
    """Trading Short"""
    dctContracts['Underlying Short'] = 'ES1R'
    dctContracts['Implied Vol Short'] = 'UX1R'
    dctWeights['Underlying Short'] =-1.0
    dctWeights['Implied Vol Short'] =0.5
    
if Universe =="UE":
    """ Signal Calculus"""
    dctContracts['Implied Vol']='V2X'
    dctContracts['Underlying Index']='SX5E'
    """ Trading Long """
    dctContracts['Underlying Long']='SX5E'
    dctContracts['Implied Vol Long']='FVS1R'
    dctWeights['Underlying Long'] = 1.0
    dctWeights['Implied Vol Long'] = 0.0
    """Trading Short"""
    dctContracts['Underlying Short']='VG1R'
    dctContracts['Implied Vol Short']='FVS1R'
    dctWeights['Underlying Short'] =0.0
    dctWeights['Implied Vol Short'] = 0.5


dtBegin =datetime.datetime(2004,1,1)
dtEnd =datetime.datetime.today()

perc_window=200
percentile=0.1
BA_Underlying=0*0.05/100
BA_Cover=0*0.1/100





def Implied_vs_Historical_Volatility_Strategy(dctContracts,dctWeights,percentile,perc_window,dtBegin,dtEnd,BA_Underlying,BA_Cover):
    percentile*=100
    """ LOAD PRICES """
    contract_list = [dctContracts['Implied Vol'],dctContracts['Underlying Index'],dctContracts['Underlying Long'],dctContracts['Implied Vol Long'],dctContracts['Underlying Short'],dctContracts['Implied Vol Short']]
    newStrat=Asset_Price_getPrices(contract_list,['Last_Price'],dtBegin,dtEnd,False,np.nan)
        
    if dctWeights['Implied Vol Short'] ==0 and dctWeights['Implied Vol Long'] ==0:
        newStrat[dctContracts['Implied Vol Short']] =1
        newStrat[dctContracts['Implied Vol Long']]=1 
    newStrat=newStrat.dropna()
    
    """ HISTORICAL VOLATILITY CALCULUS """
    newStrat[dctContracts['Underlying Index']+' vol']=ntx_rolling_std(newStrat,"Last_Price."+dctContracts['Underlying Index'],20,True)
    newStrat['Vol diff']=newStrat["Last_Price."+dctContracts['Implied Vol']]-newStrat[dctContracts['Underlying Index']+' vol']*100

    newStrat['Trigger']=0.0
    
    for k in range(perc_window,len(newStrat)):
        newStrat.loc[newStrat.index[k],'Trigger']=np.percentile(newStrat['Vol diff'][k-perc_window+1:k+1],percentile)
        
    newStrat=newStrat[newStrat.index>=newStrat.index[perc_window]]   
    
    newStrat['Index']=100.0
    newStrat['Benchmark']=100.0 
    newStrat['Underlying Exposure'] = 0.0
    newStrat['Cover Exposure'] = 0.0
    newStrat['bull performance'] = 0.0
    newStrat['bear performance'] = 0.0
    newStrat['performance'] = 0.0
    newStrat['UBear performance without W']=0.0
    newStrat['IBear performance without W']=0.0
    newStrat['UBull performance without W']=0.0
    newStrat['IBull performance without W']=0.0
    
    
    
    Bearcounter=1
    Bullcounter=1
    for k in range(2,len(newStrat)):
        
        Trigger = newStrat.loc[newStrat.index[k-2],'Trigger']
        
        perfUBull=0.0
        perfIBull=0.0
        perfUBear=0.0
        perfIBear=0.0
        
        if newStrat.loc[newStrat.index[k-2],'Vol diff']>Trigger:
            #""" bullish """
            newStrat.loc[newStrat.index[k-1],'Underlying Exposure']= dctWeights['Underlying Long']
            newStrat.loc[newStrat.index[k-1],'Cover Exposure'] = dctWeights['Implied Vol Long']
            perf = dctWeights['Underlying Long'] * (newStrat.loc[newStrat.index[k],"Last_Price."+dctContracts['Underlying Long']]/newStrat.loc[newStrat.index[k-1],"Last_Price."+dctContracts['Underlying Long']]-1)
            perf += dctWeights['Implied Vol Long']*(newStrat.loc[newStrat.index[k],"Last_Price."+dctContracts['Implied Vol Long']]/newStrat.loc[newStrat.index[k-1],"Last_Price."+dctContracts['Implied Vol Long']]-1)
            
            perfUBull =  (newStrat.loc[newStrat.index[k],"Last_Price."+dctContracts['Underlying Long']]/newStrat.loc[newStrat.index[k-1],"Last_Price."+dctContracts['Underlying Long']]-1)
            perfIBull = (newStrat.loc[newStrat.index[k],"Last_Price."+dctContracts['Implied Vol Long']]/newStrat.loc[newStrat.index[k-1],"Last_Price."+dctContracts['Implied Vol Long']]-1)
            
            #updating weights using M1,M2 and ratio in the Bull market
            #in this situation, the underlyn has a smooth almost constant behavior compare to Implied which usually is negative
            
            Bullcounter=Bullcounter+1
            if(Bullcounter%nBull==0):
                
                l=Bullcounter/nBull
                if (l>len(arrayM1Bull)-1) : break
                m1=arrayM1Bull[l-1]
                m2=arrayM2Bull[l-1]
                alpha=arrayalphabull[l-1]
                beta=arraybetabull[l-1]
                ratio=arrayWeightBull[l-1]
                #updating BULL weights
                
                dctWeights['Underlying Long']=alpha/20
                dctWeights['Implied Vol Long']=beta/20
                
                    
                
        else:
            #""" bearish """
            newStrat.loc[newStrat.index[k-1],'Underlying Exposure'] = dctWeights['Underlying Short']
            newStrat.loc[newStrat.index[k-1],'Cover Exposure'] = dctWeights['Implied Vol Short']
            perf = dctWeights['Underlying Short'] * (newStrat.loc[newStrat.index[k],"Last_Price."+dctContracts['Underlying Short']]/newStrat.loc[newStrat.index[k-1],"Last_Price."+dctContracts['Underlying Short']]-1)
            perf += dctWeights['Implied Vol Short']* (newStrat.loc[newStrat.index[k],"Last_Price."+dctContracts['Implied Vol Short']]/newStrat.loc[newStrat.index[k-1],"Last_Price."+dctContracts['Implied Vol Short']]-1)
            
            perfUBear = (newStrat.loc[newStrat.index[k],"Last_Price."+dctContracts['Underlying Short']]/newStrat.loc[newStrat.index[k-1],"Last_Price."+dctContracts['Underlying Short']]-1)
            perfIBear = (newStrat.loc[newStrat.index[k],"Last_Price."+dctContracts['Implied Vol Short']]/newStrat.loc[newStrat.index[k-1],"Last_Price."+dctContracts['Implied Vol Short']]-1)
            
            #updating weights using M1,M2 and ratio in the Bear market, alpha look zero
            Bearcounter=Bearcounter+1
            if (Bearcounter%nBear==0):
                
                
                l=Bearcounter/nBear    
                if (l>len(arrayM1Bear)-1) : break
                m1=arrayM1Bear[l-1]
                m2=arrayM2Bear[l-1]
                ratio=arrayWeightBear[l-1]
                
                alpha=arrayalphabear[l-1]
                beta=arraybetabear[l-1]
                
                dctWeights['Underlying Short']=alpha/20
                dctWeights['Implied Vol Short']=beta/20
                
                #updating BEAR weights
        if k>2:
            UnderlyingTraded = np.abs(newStrat.loc[newStrat.index[k-1],'Underlying Exposure']-newStrat.loc[newStrat.index[k-2],'Underlying Exposure'])
            CoverTraded = np.abs(newStrat.loc[newStrat.index[k-1],'Cover Exposure']-newStrat.loc[newStrat.index[k-2],'Cover Exposure'])
        else:
            UnderlyingTraded = np.abs(newStrat.loc[newStrat.index[k-1],'Underlying Exposure'])
            CoverTraded =  np.abs(newStrat.loc[newStrat.index[k-1],'Cover Exposure'])
                
        perf -=  UnderlyingTraded * BA_Underlying 
        perf -=  CoverTraded * BA_Cover
        
        
        perV1 = dctWeights['Underlying Long'] * (newStrat.loc[newStrat.index[k],"Last_Price."+dctContracts['Underlying Long']]/newStrat.loc[newStrat.index[k-1],"Last_Price."+dctContracts['Underlying Long']]-1)
        perV1 += dctWeights['Implied Vol Long']*(newStrat.loc[newStrat.index[k],"Last_Price."+dctContracts['Implied Vol Long']]/newStrat.loc[newStrat.index[k-1],"Last_Price."+dctContracts['Implied Vol Long']]-1)
        
        perV2 = dctWeights['Underlying Short'] * (newStrat.loc[newStrat.index[k],"Last_Price."+dctContracts['Underlying Short']]/newStrat.loc[newStrat.index[k-1],"Last_Price."+dctContracts['Underlying Short']]-1)
        perV2 += dctWeights['Implied Vol Short']* (newStrat.loc[newStrat.index[k],"Last_Price."+dctContracts['Implied Vol Short']]/newStrat.loc[newStrat.index[k-1],"Last_Price."+dctContracts['Implied Vol Short']]-1)
        
        newStrat.loc[newStrat.index[k],'bull performance']=perV1
        newStrat.loc[newStrat.index[k],'bear performance']=perV2
        newStrat.loc[newStrat.index[k],'performance']=perf
        
        newStrat.loc[newStrat.index[k],'UBear performance without W']=perfUBear
        newStrat.loc[newStrat.index[k],'IBear performance without W']=perfIBear
        newStrat.loc[newStrat.index[k],'UBull performance without W']=perfUBull
        newStrat.loc[newStrat.index[k],'IBull performance without W']=perfIBull
        
        newStrat.loc[newStrat.index[k],'Benchmark']=newStrat.loc[newStrat.index[k-1],'Benchmark']*(newStrat.loc[newStrat.index[k],"Last_Price."+dctContracts['Underlying Index']]/newStrat.loc[newStrat.index[k-1],"Last_Price."+dctContracts['Underlying Index']])
        
        
        newStrat.loc[newStrat.index[k],'Index']=newStrat.loc[newStrat.index[k-1],'Index']*(1+perf)
        
    return newStrat





















