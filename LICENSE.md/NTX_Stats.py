# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 10:36:54 2016

NTX Statistics functions

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
import os

#**********************************
# IMPORT NYC ENGINEERING LIBRAIRIES
#**********************************
os.chdir(r'H:\Local_Code')

from class_CalendarUS import * 
from  CBOE_Rolled_Contracts import *
from  DB_functions import *
from  class_Strategy import *


def ntx_rolling_std(newStrat,column,iwindow,bMean):
    data=newStrat.copy()
    data['Log Return']=np.log(data[column].pct_change()+1)
    if bMean:
        data['Mean']=pd.rolling_mean(data['Log Return'],window=int(iwindow))
    else:
        data['Mean']=0.0    
    data['std']=iwindow*pd.rolling_mean((data['Log Return'])**2,window=int(iwindow))/(iwindow-1)-(iwindow)*data['Mean']**2/(iwindow-1)
    data['std']=np.sqrt(252*data['std'])
    return data['std']

def ntx_rolling_cov(newStrat,column1,column2,iwindow,bMean):
    data=newStrat.copy()
    data['Log Return1']=np.log(data[column1].pct_change()+1)
    data['Log Return2']=np.log(data[column2].pct_change()+1)
    if bMean:
        data['Mean1']=pd.rolling_mean(data['Log Return1'],window=int(iwindow))
        data['Mean2']=pd.rolling_mean(data['Log Return2'],window=int(iwindow))
    else:
        data['Mean1']=0.0   
        data['Mean2']=0.0 
        
    data['cov']=iwindow*pd.rolling_mean(data['Log Return1']*data['Log Return2'],window=int(iwindow))/(iwindow-1)-iwindow*data['Mean1']*data['Mean2']/(iwindow-1)
    data['cov']=data['cov']
    
    return data['cov']
    
def ntx_rolling_corr(newStrat,column1,column2,iwindow,bMean):
    data=newStrat.copy()
    
    data['vol1']=np.sqrt(ntx_rolling_cov(data,column1,column1,iwindow,bMean))
    data['vol2']=np.sqrt(ntx_rolling_cov(data,column2,column2,iwindow,bMean))
    
    data['Log Return1']=np.log(data[column1].pct_change()+1)
    data['Log Return2']=np.log(data[column2].pct_change()+1)
    if bMean:
        data['Mean1']=pd.rolling_mean(data['Log Return1'],window=int(iwindow))
        data['Mean2']=pd.rolling_mean(data['Log Return2'],window=int(iwindow))
    else:
        data['Mean1']=0.0   
        data['Mean2']=0.0 
        
    data['corr']=iwindow*pd.rolling_mean(data['Log Return1']*data['Log Return2'],window=int(iwindow))/(iwindow-1)-iwindow*data['Mean1']*data['Mean2']/(iwindow-1)
    data['corr']=data['corr']/(data['vol1']*data['vol2'])
    return data['corr']
    
    


    
