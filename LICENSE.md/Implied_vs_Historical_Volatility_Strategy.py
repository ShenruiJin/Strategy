# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 13:16:30 2016


         HISTORICAL VOLATILITY VS IMPLIED VOLATILITY STRATEGY


@author: tzercher
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
os.chdir(r'K:\ED_ExcelTools\Transfert\Structuring\Proprietary Indices\Python Script\US Structuring Libraries')

from  class_CalendarUS import * 
from  CBOE_Rolled_Contracts import *
from  DB_functions import *
from  class_Strategy import *




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
    
    for k in range(2,len(newStrat)):
        Trigger = newStrat.loc[newStrat.index[k-2],'Trigger']
        if newStrat.loc[newStrat.index[k-2],'Vol diff']>Trigger:
            """ bullish """
            newStrat.loc[newStrat.index[k-1],'Underlying Exposure']= dctWeights['Underlying Long']
            newStrat.loc[newStrat.index[k-1],'Cover Exposure'] = dctWeights['Implied Vol Long']
            perf = dctWeights['Underlying Long'] * (newStrat.loc[newStrat.index[k],"Last_Price."+dctContracts['Underlying Long']]/newStrat.loc[newStrat.index[k-1],"Last_Price."+dctContracts['Underlying Long']]-1)
            perf += dctWeights['Implied Vol Long']*(newStrat.loc[newStrat.index[k],"Last_Price."+dctContracts['Implied Vol Long']]/newStrat.loc[newStrat.index[k-1],"Last_Price."+dctContracts['Implied Vol Long']]-1)
            
            
        else:
            """ bearish """
            newStrat.loc[newStrat.index[k-1],'Underlying Exposure'] = dctWeights['Underlying Short']
            newStrat.loc[newStrat.index[k-1],'Cover Exposure'] = dctWeights['Implied Vol Short']
            perf = dctWeights['Underlying Short'] * (newStrat.loc[newStrat.index[k],"Last_Price."+dctContracts['Underlying Short']]/newStrat.loc[newStrat.index[k-1],"Last_Price."+dctContracts['Underlying Short']]-1)
            perf += dctWeights['Implied Vol Short']* (newStrat.loc[newStrat.index[k],"Last_Price."+dctContracts['Implied Vol Short']]/newStrat.loc[newStrat.index[k-1],"Last_Price."+dctContracts['Implied Vol Short']]-1)
        
        if k>2:
            UnderlyingTraded = np.abs(newStrat.loc[newStrat.index[k-1],'Underlying Exposure']-newStrat.loc[newStrat.index[k-2],'Underlying Exposure'])
            CoverTraded = np.abs(newStrat.loc[newStrat.index[k-1],'Cover Exposure']-newStrat.loc[newStrat.index[k-2],'Cover Exposure'])
        else:
            UnderlyingTraded = np.abs(newStrat.loc[newStrat.index[k-1],'Underlying Exposure'])
            CoverTraded =  np.abs(newStrat.loc[newStrat.index[k-1],'Cover Exposure'])
                
        perf -=  UnderlyingTraded * BA_Underlying 
        perf -=  CoverTraded * BA_Cover
        
        newStrat.loc[newStrat.index[k],'Benchmark']=newStrat.loc[newStrat.index[k-1],'Benchmark']*(newStrat.loc[newStrat.index[k],"Last_Price."+dctContracts['Underlying Index']]/newStrat.loc[newStrat.index[k-1],"Last_Price."+dctContracts['Underlying Index']])
        
        
        newStrat.loc[newStrat.index[k],'Index']=newStrat.loc[newStrat.index[k-1],'Index']*(1+perf)
        
    return newStrat






