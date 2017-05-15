# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 15:32:20 2015

                           VARIANCE SWAP MEAN REVERSION STRATEGY

@author: tzercher/jblinot
"""
#*****************************************************************************
#                           IMPORTS
#*****************************************************************************


#*************************
# IMPORT PYTHON LIBRAIRIES
#*************************

import pandas as pd
import numpy as np
import datetime
import os
import sqlite3
#**********************************
# IMPORT NYC ENGINEERING LIBRAIRIES
#**********************************
os.chdir(r'K:\ED_ExcelTools\Transfert\Structuring\Proprietary Indices\Python Script\US Structuring Libraries')

from class_CalendarUS import * 
from DB_functions import *
from class_Strategy import *

def MeanReversion(Contract_list,nbBusinessDays,MAwindow,dateBegin,dateEnd, Leverage,BAsk):
    
    Index = Contract_list[0]
    Rates12M = Contract_list[1]
    Overnight =  Contract_list[2]
    
    #**********************************
    #            LOAD  DATA 
    #**********************************

    dateBegintemp=datetime.datetime(1998,1,1)
    dftemp=Asset_Price_getPrices(Index,['Last_Price'],dateBegintemp,dateEnd,False,np.nan)

    dftemp['MA200']= pd.rolling_mean(dftemp['Last_Price.'+str(Index)],MAwindow)
    dftemp['logReturn']=np.log(dftemp['Last_Price.'+str(Index)].pct_change()+1.0)**2.0
    dftemp['vol3m']=0.0
    for k in range(0,len(dftemp)):
        dftemp.loc[dftemp.index[k],'vol3m'] = np.sqrt(252)*np.sqrt(dftemp['logReturn'][k-20:k].mean())   
    dftemp=dftemp[dftemp.index>=dateBegin]
    calendar=dftemp.index.tolist()
    
    newStrat=Asset_Price_getPrices(Contract_list,['Last_Price'],dateBegintemp,dateEnd,False,calendar)
    newStrat=newStrat.fillna(method='ffill')
    newStrat=newStrat.dropna()
 
    
    #**********************************
    #            STAT CALCULUS 
    #**********************************
    
    newStrat['MA200'] = dftemp['MA200']
    newStrat['logReturn'] = dftemp['logReturn']
    newStrat['vol3m'] = dftemp['vol3m']
    newStrat['Leverage'] = Leverage * 1.0 / (newStrat['vol3m']**2.0)
    newStrat['WeekDay']=newStrat.index.map(lambda x: x.weekday())     
    newStrat['isMondayOrFriday'] = newStrat['WeekDay'].apply(lambda x: 0 if (x == 0 or x == 4) else 1)
    
    newStrat['Dates_Format']=newStrat.index
    sumVarSwap = 0.0
    
    for i in range(1,nbBusinessDays):
        sumVarSwap += np.log(newStrat['Last_Price.'+str(Index)].shift(2)/newStrat['Last_Price.'+str(Index)].shift(i+2))
        
    newStrat['delta'] = -1.0/5.0*newStrat['Leverage'].shift(1)* sumVarSwap 
    newStrat=newStrat[np.isnan(newStrat['delta'])==False]
    newStrat['delta']=newStrat['delta'].map(lambda x: min(2.5,max(-2.5,x)))
    newStrat['delta_cap']=newStrat['delta']/newStrat['Last_Price.'+str(Index)].shift(2)
    
    
    for i in range(0,len(newStrat)):
        if (newStrat['MA200'][i-2]<newStrat['Last_Price.'+str(Index)][i-2]):
            newStrat['delta_cap'][i] = max(-0.0,newStrat['delta_cap'][i])
        else:
            newStrat['delta_cap'][i] = min(0.0,newStrat['delta_cap'][i])
            
            
    
    
    newStrat['Index'] = 100.0
    strikeindex = 0
    emprunt = 0
    position = 0
    fees = 0
    
    for i in range(strikeindex+1,len(newStrat)):
        diffDay = (newStrat['Dates_Format'][i] - newStrat['Dates_Format'][i-1]).days
        diffDayfromInception = (newStrat['Dates_Format'][i]-newStrat['Dates_Format'][strikeindex]).days
        emprunt += 0.10 * (newStrat['Last_Price.'+str(Overnight)][i-1]/100.0* min(0,(1.0 - np.abs(newStrat['delta_cap'][i])  + 10.0*(newStrat['Index'][i-1]/newStrat['Index'][strikeindex]-1.0))) * diffDay/365.0) 
        initialMargin = - 0.10 * newStrat['Last_Price.'+str(Rates12M)][strikeindex]/100.0 *diffDayfromInception/365.0 
    
        diffPrices = (newStrat['Last_Price.'+str(Index)][i]-newStrat['Last_Price.'+str(Index)][i-1]) 
        position += newStrat['delta_cap'][i-1] * diffPrices
        
        
        diffDelta  = newStrat['delta_cap'][i]-newStrat['delta_cap'][i-1]
        fees += (diffDelta * (diffPrices - np.sign(diffDelta) * BAsk))
    
        newStrat['Index'][i] = newStrat['Index'][strikeindex]*(1+ position + 0*initialMargin + 0*emprunt + fees)
        
        if (newStrat['Dates_Format'][i]).day != (newStrat['Dates_Format'][i-1]).day:
            strikeindex = i
            emprunt = 0
            position = 0
            fees = 0
        
    return newStrat
    
def MeanReversion_Vol(Contract_list,nbBusinessDays,MAwindow,dateBegin,dateEnd,Leverage,BAsk):
    
    Index = Contract_list[0]
    Rates12M = Contract_list[1]
    Overnight =  Contract_list[2]
    #**********************************
    #            LOAD  DATA 
    #**********************************
    chdir(r'K:\ED_ExcelTools\Transfert\Structuring\Proprietary Indices\Python Script\US Structuring Libraries')
    #we try to find the last database

    dateBegintemp=datetime.datetime(1998,1,1)
    dftemp=Asset_Price_getPrices(Index,['Last_Price'],dateBegintemp,dateEnd,False,np.nan)
    dftemp['MA200']= pd.rolling_mean(dftemp['Last_Price.'+str(Index)],MAwindow)
    dftemp['logReturn']=np.log(dftemp['Last_Price.'+str(Index)].pct_change()+1.0)**2.0
    dftemp['vol3m']=0.0
    for k in range(0,len(dftemp)):
        dftemp.loc[dftemp.index[k],'vol3m'] = np.sqrt(252)*np.sqrt(dftemp['logReturn'][k-20:k].mean())   
    dftemp=dftemp[dftemp.index>=dateBegin]
    calendar=dftemp.index.tolist()
    newStrat=Asset_Price_getPrices(Contract_list,['Last_Price'],dateBegintemp,dateEnd,False,calendar)
    newStrat=newStrat.fillna(method='ffill')
    newStrat=newStrat.dropna()

    
    #**********************************
    #            STAT CALCULUS 
    #**********************************
    
    newStrat['MA200'] = dftemp['MA200']
    newStrat['logReturn'] = dftemp['logReturn']
    newStrat['vol3m'] = dftemp['vol3m']
    newStrat['Leverage'] = Leverage * 1.0 / (newStrat['vol3m']**2.0)
    newStrat['WeekDay']=newStrat.index.map(lambda x: x.weekday())     
    newStrat['isMondayOrFriday'] = newStrat['WeekDay'].apply(lambda x: 0 if (x == 0 or x == 4) else 1)
    
    newStrat['Dates_Format']=newStrat.index
    sumVarSwap = 0.0
    for i in range(1,nbBusinessDays):
        sumVarSwap += np.log(newStrat['Last_Price.'+str(Index)].shift(2)/newStrat['Last_Price.'+str(Index)].shift(i+2))
        
    newStrat['delta'] = -1.0/5.0*newStrat['Leverage'].shift(1)* sumVarSwap 
    newStrat=newStrat[np.isnan(newStrat['delta'])==False]
    newStrat['delta']=newStrat['delta'].map(lambda x: min(2.0,max(-2.0,x)))
    newStrat['delta'] = newStrat['delta'].fillna(0)
    newStrat['delta_cap']=newStrat['delta']/newStrat['Last_Price.'+str(Index)].shift(2)
    for i in range(0,len(newStrat)):
        if (newStrat['MA200'][i-2]<newStrat['Last_Price.'+str(Index)][i-2]):
            newStrat['delta_cap'][i] = max(-0.0,newStrat['delta_cap'][i])
        else:
            newStrat['delta_cap'][i] = min(0.0,newStrat['delta_cap'][i])
        
        
    newStrat['Index'] = 100.0
    strikeindex = 0
    emprunt = 0
    position = 0
    fees = 0
    for i in range(strikeindex+1,len(newStrat)):
        diffDay = (newStrat['Dates_Format'][i] - newStrat['Dates_Format'][i-1]).days
        diffDayfromInception = (newStrat['Dates_Format'][i]-newStrat['Dates_Format'][strikeindex]).days
        emprunt += 0.10 * (newStrat['Last_Price.'+Overnight][i-1]/100.0* min(0,(1.0 - np.abs(newStrat['delta_cap'][i])  + 10.0*(newStrat['Index'][i-1]/newStrat['Index'][strikeindex]-1.0))) * diffDay/365.0) 
        initialMargin = - 0.10 * newStrat['Last_Price.'+Rates12M][strikeindex]/100.0 *diffDayfromInception/365.0 
    
        diffPrices = (newStrat['Last_Price.'+str(Index)][i]-newStrat['Last_Price.'+str(Index)][i-1]) 
        position += newStrat['delta_cap'][i-1] * diffPrices
        
        diffDelta  = newStrat['delta_cap'][i]-newStrat['delta_cap'][i-1]
        fees += (diffDelta * (diffPrices - np.sign(diffDelta)*BAsk*newStrat['Last_Price.'+str(Index)][i-1]))
        newStrat['Index'][i] = newStrat['Index'][strikeindex]*(1+ position + 0*initialMargin + 0*emprunt + fees)
        
        if (newStrat['Dates_Format'][i]).day != (newStrat['Dates_Format'][i-1]).day:
            strikeindex = i
            emprunt = 0
            position = 0
            fees = 0
        
    return newStrat