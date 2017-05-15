# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 15:32:20 2015

                                            SHORT VOL STRATEGY

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
import matplotlib.pyplot as plt
import datetime
import os
from os import chdir
import sqlite3
#**********************************
# IMPORT NYC ENGINEERING LIBRAIRIES
#**********************************
os.chdir(r'K:\ED_ExcelTools\Transfert\Structuring\Proprietary Indices\Python Script\US Structuring Libraries')

from class_CalendarUS import * 
from CBOE_Rolled_Contracts import *
from DB_functions import *
from class_Strategy import *


def Short_Vol_Strategy(sRegion,tick_up,tick_down,limit_up,limit_down,vol_window,MA_window,dBidAskSpread,MarginCall,iDelay,dateBegin,dateEnd):
    #****************************
    # CONTRACTS MANAGEMENT     **
    #****************************
    if sRegion == "US":
        Contract_list=["UX1","UX2","UX3","UX4","UX5","UX6","UX7","UX8",'VIX','US0012M','US00ON']
        volIndex = 'VIX'
    else:
        Contract_list=["FVS1","FVS2","FVS3","FVS4","FVS5","FVS6","FVS7","FVS8","V2X","EONIA","EUR012M"]
        volIndex = 'V2X'   
    Contract1=Contract_list[0]
    Contract2=Contract_list[1]
    Contract3=Contract_list[2]
    Contract4=Contract_list[3]
    Contract5=Contract_list[4]
    Contract6=Contract_list[5]
    Contract7=Contract_list[6]
    Contract8=Contract_list[7]
    Contract9=Contract_list[8]
    Contract10=Contract_list[9]
    
    #********************************************
    # OPEN DATABASE, LOAD PRICES,CLOSE DB
    #********************************************
    chdir(r'K:\ED_ExcelTools\Transfert\Structuring\Proprietary Indices\Python Script\US Structuring Libraries')
    calendar=Calendar(dateBegin, dateEnd).BusinessDaysFixingDates 
    newStrat=Asset_Price_getPrices(Contract_list,['Last_Price'],dateBegin,dateEnd,False,calendar)
    
    newStrat=newStrat.fillna(method='ffill')
    newStrat=newStrat.dropna()
    
    #****************************************
    # OPEN DATA + VIX STATISTICS CALCULUS  **
    #****************************************
    newStrat['Dates_Format']=newStrat.index
    newStrat['Year']=newStrat['Dates_Format'].map(lambda x: x.year)
    newStrat['Vol Index MA']=pd.rolling_mean(newStrat["Last_Price."+volIndex],window=MA_window)
    newStrat['Vol Index Trend']=newStrat['Vol Index MA'].diff()
    newStrat['Vol Index Vol']=np.sqrt(252)*np.sqrt((vol_window-1)*pd.rolling_mean((np.log(newStrat["Last_Price."+volIndex].pct_change()+1))**2,window=(vol_window-1))/(vol_window-2))

    #****************************************
    # NTX ST and MT index Calculus         **
    #****************************************
    #ST: uses contract 1 and contract 2
    #MT: uses contract 2 and contract 3
    ST_Strat=US_ST_Rolled_Vol_Index_NTX(Contract1,Contract2,Contract3,dateBegin,dateEnd)
    ST_Strat.Index.set_index('Dates_Format',inplace=True)
    newStrat[Contract1+".Weight"]=ST_Strat.Index['W1']
    newStrat[Contract2+".Weight"]=ST_Strat.Index['W2']
    newStrat['Is Rebalancing']=ST_Strat.Index['IsRoll']

    #****************************************
    #  SPOT AND CONTANGO SIGNAL            **
    #****************************************
    newStrat[Contract1+"Modif"]=newStrat[Contract1+".Weight"]*newStrat["Last_Price."+Contract1]+newStrat[Contract2+".Weight"]*newStrat["Last_Price."+Contract2]
    newStrat[Contract2+"Modif"]=newStrat[Contract1+".Weight"]*newStrat["Last_Price."+Contract2]+newStrat[Contract2+".Weight"]*newStrat["Last_Price."+Contract3]
    
    """ spot signal: Beta Calculus""" # compute the beta with respect to the Vol index (beta 20BD)
    newStrat['Spot Signal 1']=-252*(vol_window-1)*pd.rolling_mean(np.log(newStrat["Last_Price."+volIndex].pct_change()+1)*np.log(newStrat[Contract1+"Modif"].pct_change()+1),window=(vol_window-1))/((vol_window-2)*newStrat['Vol Index Vol']**2)
#    """ spot signal: spot expected pnl Calculus""" # beta * trend
    newStrat['Spot Signal 1']*=newStrat['Vol Index Trend']

    """ contango signal: expected daily pnl (slope) """
    newStrat['Contango Signal 1']=-(newStrat[Contract1+"Modif"]-newStrat[Contract2+"Modif"])/20
    
    #****************************************
    #  WEIGHT CALCULUS                     **
    #****************************************
    """weight calculus """
    newStrat['ST weight']=0.0

    for k in range(21,len(newStrat)):
        if newStrat['Contango Signal 1'][k]+newStrat['Spot Signal 1'][k]>0:
            newStrat['ST weight'][k]= newStrat['ST weight'][k-1]-tick_down
            newStrat['ST weight'][k]=max(newStrat['ST weight'][k],limit_down)
        else:
            newStrat['ST weight'][k]= newStrat['ST weight'][k-1]+tick_up
            newStrat['ST weight'][k]=min(newStrat['ST weight'][k],limit_up)

    
    #***************************************** 
    #******** CREATE THE INDEX           *****
    #*****************************************
    
    newStrat['Index']=100.0
    newStrat['Contract1Expo']=0.0
    newStrat['Contract2Expo']=0.0

    newStrat['Contract1Traded']=0.0
    newStrat['Contract2Traded']=0.0

    newStrat['Contract1Number']=0.0
    newStrat['Contract2Number']=0.0

    newStrat['Perf Expo']=0.0
    newStrat['Perf Trading']=0.0
    newStrat['Perf OV']=0.0
    newStrat['Perf Rates']=0.0
  
    Strike_Index =22
    Rate_rebal = newStrat.loc[newStrat.index[Strike_Index],"Last_Price."+Contract9]/100
    Rate_Cost=0.0
    Overnight_Cost=0.0
    
    for k in range(23,len(newStrat)):
        ##############################################################
        #
        # IN THIS LOOP WE CALCULATE THE PERFORMANCE BETWEEN K-1 AND K 
        #
        ##############################################################
        
        #*************************************************** 
        #********Check rate performance (annual rebalancing)               
        #***************************************************
        if newStrat.loc[newStrat.index[k-1],'Dates_Format'].year !=newStrat.loc[newStrat.index[k-2],'Dates_Format'].year:
            Strike_Index =k-1
            Rate_rebal = newStrat.loc[newStrat.index[Strike_Index],"Last_Price."+Contract9]/100  
        Nbdays = (newStrat.loc[newStrat.index[k],'Dates_Format']-newStrat.loc[newStrat.index[k-1],'Dates_Format']).days
        Rate_Cost = MarginCall*(newStrat.loc[newStrat.index[Strike_Index],'Index']/newStrat.loc[newStrat.index[k-1],'Index'])* Rate_rebal*Nbdays/365
        
        #******************************************************** 
        #********Get yesterday's contract prices   and expo    **
        #********************************************************
        """ Attention: at the close of a rebal day: contract are rolled
            U2 becomes U1 etc
            For example if t-1 is a Rebal day, on t our performance on a contract U1
            is W1 * U1 / U2 to take the rolling cost into account """
        """ IMPORTANT :
        It = I_t-1 *(1+ expo1*perf + ... ) => It_-1 and expo are not known before the close on t-1 !
        a t-1: 
        => the level of the Index is assessed (using assessed values of the underlying contract)
        => the ContractExpo are then calculated using those approximations = TRADING RISK
        However: the signal is used with a 1-day (iDelay=1) delay because it is unsure trading can assess it the same way
        """    
            
        ST_expo = newStrat.loc[newStrat.index[k-1-iDelay],'ST weight'] #iDelay => 1: use the day before's data
        if newStrat.loc[newStrat.index[k-1],'Is Rebalancing']==1:
            #Get PRICES
            Contract1_Yesterday=newStrat.loc[newStrat.index[k-1],"Last_Price."+Contract2]
            Contract2_Yesterday=newStrat.loc[newStrat.index[k-1],"Last_Price."+Contract3]
            #Compute expo
            Contract1Expo =ST_expo*(newStrat.loc[newStrat.index[k-1],Contract2+".Weight"]*Contract2_Yesterday)/(newStrat.loc[newStrat.index[k-1],Contract1+".Weight"]*Contract1_Yesterday+newStrat.loc[newStrat.index[k-1],Contract2+".Weight"]*Contract2_Yesterday)
            Contract2Expo=0
            #The exposure we have at the close on k-1
            newStrat.loc[newStrat.index[k-1],'Contract1Expo']= Contract1Expo
            newStrat.loc[newStrat.index[k-1],'Contract2Expo']= Contract2Expo
            Total_Expo = np.abs(Contract1Expo)+np.abs(Contract2Expo)
            #The number of contracts we had before trading on k-1
            NbContract1_BeforeTrading = newStrat.loc[newStrat.index[k-2],'Contract2Number']
            NbContract2_BeforeTrading = 0.0
            #The number of contracts we have at the close on k-1
            newStrat.loc[newStrat.index[k-1],'Contract1Number']=Contract1Expo*newStrat.loc[newStrat.index[k-1],'Index']/Contract1_Yesterday
            newStrat.loc[newStrat.index[k-1],'Contract2Number']=Contract2Expo*newStrat.loc[newStrat.index[k-1],'Index']/Contract2_Yesterday  
            #The exposure we have on k-1 before trading
            Contract1Expo_BeforeTrading=NbContract1_BeforeTrading*Contract1_Yesterday/newStrat.loc[newStrat.index[k-1],'Index']
            Contract2Expo_BeforeTrading=0.0 
        else:
            #Get PRICES
            Contract1_Yesterday=newStrat.loc[newStrat.index[k-1],"Last_Price."+Contract1]
            Contract2_Yesterday=newStrat.loc[newStrat.index[k-1],"Last_Price."+Contract2]
            #Compute expo
            Contract1Expo =ST_expo*(newStrat.loc[newStrat.index[k-1],Contract1+".Weight"]*Contract1_Yesterday)/(newStrat.loc[newStrat.index[k-1],Contract1+".Weight"]*Contract1_Yesterday+newStrat.loc[newStrat.index[k-1],Contract2+".Weight"]*Contract2_Yesterday)
            Contract2Expo =ST_expo*(newStrat.loc[newStrat.index[k-1],Contract2+".Weight"]*Contract2_Yesterday)/(newStrat.loc[newStrat.index[k-1],Contract1+".Weight"]*Contract1_Yesterday+newStrat.loc[newStrat.index[k-1],Contract2+".Weight"]*Contract2_Yesterday)
            #The exposure we have at the close on k-1
            newStrat.loc[newStrat.index[k-1],'Contract1Expo']= Contract1Expo
            newStrat.loc[newStrat.index[k-1],'Contract2Expo']= Contract2Expo
            Total_Expo = np.abs(Contract1Expo)+np.abs(Contract2Expo)
            #The number of contracts we had before trading on k-1 
            NbContract1_BeforeTrading = newStrat.loc[newStrat.index[k-2],'Contract1Number']
            NbContract2_BeforeTrading = newStrat.loc[newStrat.index[k-2],'Contract2Number']
            #The number of contract we have at the close on k-1
            newStrat.loc[newStrat.index[k-1],'Contract1Number']=Contract1Expo*newStrat.loc[newStrat.index[k-1],'Index']/Contract1_Yesterday
            newStrat.loc[newStrat.index[k-1],'Contract2Number']=Contract2Expo*newStrat.loc[newStrat.index[k-1],'Index']/Contract2_Yesterday
            #The exposure we have on k-1 before trading
            Contract1Expo_BeforeTrading=NbContract1_BeforeTrading*Contract1_Yesterday/newStrat.loc[newStrat.index[k-1],'Index']
            Contract2Expo_BeforeTrading=NbContract2_BeforeTrading*Contract2_Yesterday/newStrat.loc[newStrat.index[k-1],'Index']

        #****************************************
        # CHECK CASH BORROWING
        #****************************************
        Cash =newStrat.loc[newStrat.index[k-1],'Index']-(1-MarginCall)*newStrat.loc[newStrat.index[Strike_Index],'Index']

        if Total_Expo*MarginCall>Cash:
            Emprunt =Total_Expo*MarginCall-np.sign(Cash)*Cash
            Emprunt /=newStrat.loc[newStrat.index[k-1],'Index']
            Rate_overnight=max(newStrat.loc[newStrat.index[k-1],"Last_Price."+Contract10],0)/100
        else:
            Emprunt = 0
            Rate_overnight=0
        Overnight_Cost=Emprunt * Rate_overnight*Nbdays/365
        
        #*************************************************** 
        #******** Decompose Position vs Trading        *****
        #***************************************************

        Contract1Traded=Contract1Expo-Contract1Expo_BeforeTrading
        Contract2Traded=Contract2Expo-Contract2Expo_BeforeTrading
                
        newStrat.loc[newStrat.index[k-1],'Contract1Traded']=Contract1Traded
        newStrat.loc[newStrat.index[k-1],'Contract2Traded']=Contract2Traded

        newStrat.loc[newStrat.index[k-1],'Contract1 Number Traded']=Contract1Traded*newStrat.loc[newStrat.index[k-1],'Index']/ Contract1_Yesterday
        newStrat.loc[newStrat.index[k-1],'Contract2 Number Traded']=Contract2Traded*newStrat.loc[newStrat.index[k-1],'Index']/ Contract2_Yesterday
 
        #****************************************
        # Compute the perf position of yersterday
        #****************************************
        perf_Position=0.0
        perf_Position += Contract1Expo_BeforeTrading *(newStrat.loc[newStrat.index[k],"Last_Price."+Contract1]/Contract1_Yesterday-1)
        perf_Position +=  Contract2Expo_BeforeTrading *(newStrat.loc[newStrat.index[k],"Last_Price."+Contract2]/Contract2_Yesterday-1)

        #***********************************
        # Compute the trading performance
        #***********************************
        perf_trading = 0.0
        perf_trading +=Contract1Traded*(newStrat.loc[newStrat.index[k],"Last_Price."+Contract1]-Contract1_Yesterday-np.sign(Contract1Traded)*dBidAskSpread)/Contract1_Yesterday
        perf_trading +=Contract2Traded*(newStrat.loc[newStrat.index[k],"Last_Price."+Contract2]-Contract2_Yesterday-np.sign(Contract2Traded)*dBidAskSpread)/Contract2_Yesterday

        perf = perf_Position+perf_trading
        newStrat.loc[newStrat.index[k],'Perf Expo']=perf_Position
        newStrat.loc[newStrat.index[k],'Perf Trading']=perf_trading
        newStrat.loc[newStrat.index[k],'Perf OV']=Overnight_Cost
        newStrat.loc[newStrat.index[k],'Perf Rates']=Rate_Cost
        newStrat.loc[newStrat.index[k],'Index']=newStrat.loc[newStrat.index[k-1],'Index']*(1+perf-Rate_Cost-Overnight_Cost)
        if newStrat.loc[newStrat.index[k],'Index']==np.nan:
            newStrat.loc[newStrat.index[k],'Index']=100.0
    
    return newStrat
    
