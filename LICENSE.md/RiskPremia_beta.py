# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 15:32:20 2015

                           RISK PREMIA FUNCTIONS

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
from itertools import takewhile
#**********************************
# IMPORT NYC ENGINEERING LIBRAIRIES
#**********************************
os.chdir(r'K:\ED_ExcelTools\Transfert\Structuring\Proprietary Indices\Python Script\US Structuring Libraries')

from class_CalendarUS import * 
from DB_functions import *
from class_Strategy import *
from PortfolioAllocation import *
from NTX_Stats import *


def RiskPremia_beta(dateBegin,dateEnd,Contract_list,Rate,dBorrow):
    #********************************************
    # OPEN DATABASE, LOAD PRICES,CLOSE DB
    #********************************************
    if  type(Rate)==str:
        Rate_temp = Rate
        Rate= []
        Rate.append(Rate_temp)
    calendar=Asset_Price_getPrices([Contract_list[0]],['Last_Price'],dateBegin,dateEnd,False,np.nan).index.tolist()
    newStrat=Asset_Price_getPrices(Contract_list,['Last_Price'],dateBegin,dateEnd,False,calendar)
    newStrat=newStrat.dropna()
        
    #load rates 
    dftemp = Asset_Price_getPrices([Rate[0]],['Last_Price'],dateBegin,dateEnd,False,np.nan)
    newStrat['Last_Price.'+str(Rate[0])]=dftemp['Last_Price.'+str(Rate[0])]
    newStrat=newStrat.fillna(method='ffill')
    #********************************************
    #               BETA CALCULUS
    #********************************************
    newStrat['Days']=0
    for k in range(1,len(newStrat)):
        newStrat['Days'][k]=(newStrat.index[k]-newStrat.index[k-1]).days
        

    newStrat[Contract_list[0]+".Vol"]=pd.rolling_mean(newStrat['Days']*np.log(newStrat['Last_Price.'+str(Contract_list[0])].pct_change()+1)**2,window=120)
    for strat in Contract_list:
        if strat!=Contract_list[0]: 
            newStrat[strat+".Beta"]=pd.rolling_mean(newStrat['Days']*np.log(newStrat['Last_Price.'+str(Contract_list[0])].pct_change()+1)*np.log(newStrat['Last_Price.'+str(strat)].pct_change()+1),window=120)/(newStrat[Contract_list[0]+".Vol"])
            newStrat[strat+".Beta"]=1/newStrat[strat+".Beta"]
            newStrat[strat+".Beta"]=newStrat[strat+".Beta"].map(lambda x: min(x,2))
            newStrat[strat+".Beta"]=newStrat[strat+".Beta"].map(lambda x: max(x,0.5))
    #***************************************************
    #               MONTHLY REBALANCING BETA RATIO
    #***************************************************

    for strat in Contract_list:
        if strat!=Contract_list[0]: 
            newStrat[strat+".IndexBeta"]=100.0
    newStrat['Rebal']=0
    RebalIndex = 121
    while newStrat.index[RebalIndex].month ==newStrat.index[RebalIndex-1].month:
        RebalIndex+=1
    RebalBeta= RebalIndex-1
    iFirstDate=RebalIndex
    rate =(newStrat["Last_Price."+Rate[0]][RebalIndex]+dBorrow)/(100*365)
    
    for k in range(RebalIndex+1,len(newStrat)):
        if newStrat.index[k-1].month!=newStrat.index[k-2].month:
            RebalIndex=k-1
            RebalBeta=k-2
            newStrat['Rebal'][k-1]=1
            rate =(newStrat["Last_Price."+Rate[0]][k-1]+1)/(100*365)
        for strat in Contract_list:
            if strat!=Contract_list[0]: 
                beta =newStrat[strat+".Beta"][RebalBeta]
                perf = beta  * (newStrat["Last_Price."+strat][k]/newStrat["Last_Price."+strat][RebalIndex]-1)
                borrow_cost=  min(0,1-beta)*rate*(newStrat.index[k]-newStrat.index[RebalIndex]).days#/newStrat[strat+".IndexBeta"][RebalIndex]
                newStrat[strat+".IndexBeta"][k]=newStrat[strat+".IndexBeta"][RebalIndex]*(1+perf+ borrow_cost)
                
    for strat in Contract_list:    
        if strat!=Contract_list[0]:
            del newStrat["Last_Price."+strat]
            del newStrat[strat+".Beta"]
    del newStrat["Last_Price."+Rate[0]]
    del newStrat[Contract_list[0]+".Vol"]
    del newStrat["Days"] 
    newStrat=newStrat[newStrat.index[iFirstDate]:newStrat.index[len(newStrat)-1]]
    return newStrat
    

def ShortRiskPremia(dateBegin,dateEnd,Contract_list,Rate):
    #********************************************
    # OPEN DATABASE, LOAD PRICES,CLOSE DB
    #********************************************
    chdir(r'K:\ED_ExcelTools\Transfert\Structuring\Proprietary Indices\Python Script\US Structuring Libraries')
    
    calendar=Asset_Price_getPrices([Contract_list[0]],['Last_Price'],dateBegin,dateEnd,False,np.nan).index.tolist()
    newStrat=Asset_Price_getPrices(Contract_list,['Last_Price'],dateBegin,dateEnd,False,calendar)
    newStrat=newStrat.fillna(method='ffill')    
    #load rates 
    dftemp = Asset_Price_getPrices([Rate[0]],['Last_Price'],dateBegin,dateEnd,False,np.nan)
    newStrat['Last_Price.'+Rate[0]]=dftemp['Last_Price.'+Rate[0]]
    newStrat=newStrat.fillna(method='ffill')

    sRate=Rate[0]
    #********************************************
    # DEFEES THE INDEX, then beta, then fees
    #********************************************
    risk_premia = []
    calendar=Calendar(dateBegin,dateEnd)
    for i in range(1,len(Contract_list)):
        risk_premia.append(Contract_list[i])
    for strat in risk_premia:
        dftemp = pd.DataFrame(index=newStrat.index)
        dftemp['Last_Price.'+strat]=newStrat['Last_Price.'+strat]
        dftemp=dftemp.dropna()
        #defees
        dftemp=Strategy(dftemp.index, dftemp['Last_Price.'+strat]).WithoutFeesIndex_RiskPremia(0.25/100,calendar)  
        #second defees (equivalent to fees..):
        dftemp=Strategy( dftemp.index, dftemp['Last_Price.'+strat]).WithoutFeesIndex(0.25/100)  
        
        #ADD BETA
        dftemp['Last_Price.'+'SPXT']=newStrat['Last_Price.'+'SPXT']
        dftemp=Beta_Allocation_Short(dftemp,'Last_Price.'+'SPXT',['strat'],sRate,1)
        
        
    return dftemp
    
  
        