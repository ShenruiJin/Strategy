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
from itertools import takewhile
#**********************************
# IMPORT NYC ENGINEERING LIBRAIRIES
#**********************************
os.chdir(r'K:\ED_ExcelTools\Transfert\Structuring\Proprietary Indices\Python Script\US Structuring Libraries')

from class_CalendarUS import * 
from CBOE_Rolled_Contracts import *
from DB_functions import *
from class_Strategy import *

from ExcessReturn import Excess_Return
from NTX_Stats import *
from RiskPremia_beta import *


def MRP_US_beta(strategies,risk_premia,rate,dateBegin,dateEnd ):
    #**********************************
    #            LOAD  DATA 
    #**********************************
    calendar=Calendar(dateBegin,dateEnd).BusinessDaysFixingDates
    newStrat=pd.DataFrame({'Date':calendar})
    newStrat.set_index('Date',inplace=True)
    
    #****** COMPUTE RISK PREMIA WITH BETA and EXCESS RETURN VS SPXT*******
    for strat in risk_premia:
        Contract_list=['SPXT',strat]
        dftemp=RiskPremia_beta(dateBegin,dateEnd,Contract_list,rate,1)
        dftemp.columns=['SPXT',strat,'Rebal']
        dftemp=Excess_Return(dftemp,strat,'SPXT')
        newStrat[strat]=dftemp['ExcessReturn']


    newStrat=newStrat.dropna()
    
    for strat in strategies:
    
        ticker = strat
        calendar = newStrat.index.tolist()
        dftemp=Asset_Price_getPrices([ticker],['Last_Price'],dateBegin,dateEnd,False,calendar).dropna()
        stratfees = Strategy(dftemp.index,dftemp['Last_Price.'+ticker])
        stratfees=stratfees.FeesIndex(0.00)
        newStrat[ticker]=stratfees['strat']
    
        newStrat=newStrat.dropna()


    #********************************************
    #    MOMENTUM CALCULUS, STRATEGY SELECTION
    #********************************************
    newStrat['Rebal']=0
    for k in range(1,len(newStrat)):
        if newStrat.index[k].month != newStrat.index[k-1].month:
            newStrat['Rebal'][k]=1
    
    PerfNXSRLOVU=(newStrat['NXSRLOVU']/newStrat['NXSRLOVU'].shift(19)).shift(1).tolist()
    PerfNXSRSMCU=(newStrat['NXSRSMCU']/newStrat['NXSRSMCU'].shift(19)).shift(1).tolist()
    PerfNXSRVALU=(newStrat['NXSRVALU']/newStrat['NXSRVALU'].shift(19)).shift(1).tolist()
    PerfNXSRHIDU=(newStrat['NXSRHIDU']/newStrat['NXSRHIDU'].shift(19)).shift(1).tolist()
    
    Perfstrat={}
    for  strat in strategies:
        Perfstrat[strat]= (newStrat[strat]/newStrat[strat].shift(19)).shift(1).tolist()

    dfMomentum=pd.DataFrame({'NXSRLOVU':PerfNXSRLOVU,'NXSRSMCU':PerfNXSRSMCU,'NXSRVALU':PerfNXSRVALU,'NXSRHIDU':PerfNXSRHIDU},index =newStrat.index)

    for  strat in strategies:
        dfMomentum[strat]=Perfstrat[strat]
        
    dfMomentum=dfMomentum.dropna()

    dfStrategy1 = dfMomentum.idxmax(axis=1).tolist()
    for i in range(0,len(dfStrategy1)):
        dfMomentum[dfStrategy1[i]][i]=-100
        
    dfStrategy2 = dfMomentum.idxmax(axis=1).tolist()
    for i in range(0,len(dfStrategy2)):
        dfMomentum[dfStrategy2[i]][i]=-100
        
    dfStrategy3 = dfMomentum.idxmax(axis=1).tolist()
    for i in range(0,len(dfStrategy3)):
        dfMomentum[dfStrategy3[i]][i]=-100  
             
    dfStrategy=pd.DataFrame({'Strategy1':dfStrategy1,'Strategy2':dfStrategy2,'Strategy3':dfStrategy3},index=dfMomentum.index)
    

    #********************************************
    #    INDEX CALCULUS
    #********************************************

    rebalDate = dfStrategy.index[0]
    Strategy1 =dfStrategy.loc[rebalDate,'Strategy1']
    Strategy2 =dfStrategy.loc[rebalDate,'Strategy2']
    Strategy3 =dfStrategy.loc[rebalDate,'Strategy3']
    
    newStrat=newStrat[newStrat.index>=dfStrategy.index[0]]
    
    newStrat['Index']=100.0
    rebalDate = newStrat.index[0]
    newStrat['Strategy1']=""
    newStrat['Strategy2']=""
    newStrat['Strategy3']=""
    
    for k in range(1,len(newStrat)):
        dtToday = newStrat.index[k]
        rebalDate= newStrat.index[k-1]
        perf =(1.0*newStrat.loc[dtToday,Strategy1]/newStrat.loc[rebalDate,Strategy1]+0.0*newStrat.loc[dtToday,Strategy2]/newStrat.loc[rebalDate,Strategy2]+0.0*newStrat.loc[dtToday,Strategy3]/newStrat.loc[rebalDate,Strategy3]-1)
        newStrat.loc[dtToday,'Index']= newStrat.loc[rebalDate,'Index']*(1+perf)
        newStrat['Strategy1'][k-1]=Strategy1
        newStrat['Strategy2'][k-1]=Strategy2
        newStrat['Strategy3'][k-1]=Strategy3
        Strategy1 =dfStrategy.loc[dtToday,'Strategy1']
        Strategy2 =dfStrategy.loc[dtToday,'Strategy2']
        Strategy3 =dfStrategy.loc[dtToday,'Strategy3']
        
    
    return newStrat

