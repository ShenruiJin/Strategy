# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 11:33:46 2016

@author: asanjari
"""


#*****************************************************************************
#                           IMPORTS
#*****************************************************************************

#*************************
# IMPORT PYTHON LIBRAIRIES
#*************************


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

#******************************************************
# DEFINING THE UNIVERSE AND SETTING THE INITIAL WEIGHTS
#******************************************************

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

risk=20

percentile*=100
""" LOAD PRICES """
contract_list = [dctContracts['Implied Vol'],dctContracts['Underlying Index'],dctContracts['Underlying Long'],dctContracts['Implied Vol Long'],dctContracts['Underlying Short'],dctContracts['Implied Vol Short']]
#newStrat=Asset_Price_getPrices(contract_list,['Last_Price'],dtBegin,dtEnd,False,np.nan)
path = r'C:\Users\asanjari\Documents\data.csv'
newStrat= pd.read_csv(path)





if dctWeights['Implied Vol Short'] ==0 and dctWeights['Implied Vol Long'] ==0:
    newStrat[dctContracts['Implied Vol Short']] =1
    newStrat[dctContracts['Implied Vol Long']]=1 
newStrat=newStrat.dropna()
    
""" HISTORICAL VOLATILITY CALCULUS """
newStrat[dctContracts['Underlying Index']+' vol']=ntx_rolling_std(newStrat,"Last_Price."+dctContracts['Underlying Index'],20,True)
newStrat['Vol diff']=newStrat["Last_Price."+dctContracts['Implied Vol']]-newStrat[dctContracts['Underlying Index']+' vol']*100


#adding the trigger
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

Aalphabear=np.array(dctWeights['Underlying Short'])
Abetabear=np.array(dctWeights['Implied Vol Short'])
Aalphabull=np.array(dctWeights['Underlying Long'])
Abetabull=np.array(dctWeights['Implied Vol Long'])

BEARindeces=[]
BULLindeces=[]


for k in range(2,len(newStrat)):
    Trigger = newStrat.loc[newStrat.index[k-2],'Trigger']
    
    perfUBull=0.0
    perfIBull=0.0
    perfUBear=0.0
    perfIBear=0.0
    
    
    ############################
    ########finding the weights
    ############################
    
    periodBull=20 #period for updating weights
    periodBear=20 #period for updating weights
    
    Nbear=len(BEARindeces)
    Nbull=len(BULLindeces)
    
    
    if (Nbear%periodBear==0 and Nbear>0 ):
        
        array1=np.array([])
        array2=np.array([])
        
        thisBear=BEARindeces[-periodBear:]
        for ind in thisBear:
             array1=np.append(array1,newStrat.loc[newStrat.index[ind],  'UBear performance without W'])
             array2=np.append(array2,newStrat.loc[newStrat.index[ind],  'IBear performance without W'])
        
        
        
        
        m1=np.mean(array1)
        m2=np.mean(array2)
        cov=np.cov(array1,array2)[0,1]
        va=np.cov(array1,array2)[0,0]
        vb=np.cov(array1,array2)[1,1]
    
        alphabear=(m1*vb-m2*cov)/(va*vb - cov*cov)
        betabear=(m2*va-m1*cov)/(va*vb - cov*cov)
        
        Aalphabear=np.append(Aalphabear,alphabear)
        Abetabear=np.append(Abetabear,betabear)
        
        dctWeights['Underlying Short']=alphabear/risk
        dctWeights['Implied Vol Short']=betabear/risk
        
    
    if (Nbull%periodBull==0 and Nbull>0 ):
        array1=np.array(0)
        array2=np.array(0)
        
        thisBull=BULLindeces[-periodBull:]
        for ind in thisBull:
             array1=np.append(array1,newStrat.loc[newStrat.index[ind],  'UBull performance without W'])
             array2=np.append(array2,newStrat.loc[newStrat.index[ind],  'IBull performance without W'])
        
        
        m1=np.mean(array1)
        m2=np.mean(array2)
        cov=np.cov(array1,array2)[0,1]
        va=np.cov(array1,array2)[0,0]
        vb=np.cov(array1,array2)[1,1]
    
        alphabull=(m1*vb-m2*cov)/(va*vb - cov*cov)
        betabull=(m2*va-m1*cov)/(va*vb - cov*cov)
        
        Aalphabull=np.append(Aalphabull,alphabull)
        Abetabull=np.append(Abetabull,betabull)
        
                
        
        dctWeights['Underlying Long']=alphabull/risk 
        dctWeights['Implied Vol Long']=betabull/risk
    



    if newStrat.loc[newStrat.index[k-2],'Vol diff']>Trigger:
        #""" bullish """
        BULLindeces.append(k-2)
                
        newStrat.loc[newStrat.index[k-1],'Underlying Exposure']= dctWeights['Underlying Long']
        newStrat.loc[newStrat.index[k-1],'Cover Exposure'] = dctWeights['Implied Vol Long']
        perf = dctWeights['Underlying Long'] * (newStrat.loc[newStrat.index[k],"Last_Price."+dctContracts['Underlying Long']]/newStrat.loc[newStrat.index[k-1],"Last_Price."+dctContracts['Underlying Long']]-1)
        perf += dctWeights['Implied Vol Long']*(newStrat.loc[newStrat.index[k],"Last_Price."+dctContracts['Implied Vol Long']]/newStrat.loc[newStrat.index[k-1],"Last_Price."+dctContracts['Implied Vol Long']]-1)
            
        perfUBull =  (newStrat.loc[newStrat.index[k],"Last_Price."+dctContracts['Underlying Long']]/newStrat.loc[newStrat.index[k-1],"Last_Price."+dctContracts['Underlying Long']]-1)
        perfIBull = (newStrat.loc[newStrat.index[k],"Last_Price."+dctContracts['Implied Vol Long']]/newStrat.loc[newStrat.index[k-1],"Last_Price."+dctContracts['Implied Vol Long']]-1)
            
            
            
    else:
        #""" bearish """        
        BEARindeces.append(k-2)
        
        newStrat.loc[newStrat.index[k-1],'Underlying Exposure'] = dctWeights['Underlying Short']
        newStrat.loc[newStrat.index[k-1],'Cover Exposure'] = dctWeights['Implied Vol Short']
        perf = dctWeights['Underlying Short'] * (newStrat.loc[newStrat.index[k],"Last_Price."+dctContracts['Underlying Short']]/newStrat.loc[newStrat.index[k-1],"Last_Price."+dctContracts['Underlying Short']]-1)
        perf += dctWeights['Implied Vol Short']* (newStrat.loc[newStrat.index[k],"Last_Price."+dctContracts['Implied Vol Short']]/newStrat.loc[newStrat.index[k-1],"Last_Price."+dctContracts['Implied Vol Short']]-1)
        
        perfUBear = (newStrat.loc[newStrat.index[k],"Last_Price."+dctContracts['Underlying Short']]/newStrat.loc[newStrat.index[k-1],"Last_Price."+dctContracts['Underlying Short']]-1)
        perfIBear = (newStrat.loc[newStrat.index[k],"Last_Price."+dctContracts['Implied Vol Short']]/newStrat.loc[newStrat.index[k-1],"Last_Price."+dctContracts['Implied Vol Short']]-1)
        
        
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
 

import matplotlib.pyplot as plt

plt.plot(newStrat['Benchmark'][:])
plt.plot(newStrat['Index'][:])




plt.plot(Aalphabear/20)
plt.plot(Abetabear/20)


plt.plot(Aalphabull/20)
plt.plot(Abetabull/20)