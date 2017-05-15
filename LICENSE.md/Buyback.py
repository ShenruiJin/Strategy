# -*- coding: utf-8 -*-
"""
Created on Mon May  2 08:34:35 2016

@author: tzercher
"""

#*****************************************************************************
#                           IMPORTS
#*****************************************************************************

#*************************
# IMPORT PYTHON LIBRAIRIES
#*************************
import pandas as pd
import sqlite3
import datetime
import numpy as np
import os
import sqlite3

#**********************************
# IMPORT US STRUCTURING LIBRAIRIES
#**********************************

os.chdir(r'K:\ED_ExcelTools\Transfert\Structuring\Proprietary Indices\Python Script\US Structuring Libraries')

DataBase_path = r'K:\ED_ExcelTools\Transfert\Structuring\Proprietary Indices\Python Script\US Structuring Libraries\Database'
Prism_path = r'K:\ED_ExcelTools\Transfert\Structuring\Proprietary Indices\Python Script\US Structuring Libraries\Database\PrismRequest'

import class_CalendarUS
#from  DataBaseREAD import *
from DB_functions import *
from class_Strategy import *


"""
****  Parameters ****
"""
sUniverse = 'SPX'
dtBegin=datetime.datetime(2007,1,1)
dtEnd=datetime.datetime(2016,3,30)
Rebal="Quaterly"

iTopBB = 100
iTopVol = 60
iTopFCF =20

"""get the SPX calendar"""
referencecalendar = DB_BDH_Call('SPX','Last_Price',dtBegin,dtEnd,False,np.nan).dropna().index.tolist()
newStrat = pd.DataFrame(index=referencecalendar)


""" ******************************* INITIALIZATION : differs only if different rebalancing type *************************"""

""" load all baskets and tickers """
all_tickers = []
underlying_list = []
dates_rebal = []
columns_rename =[]
Baskets = {}

dates_rebal.append(newStrat.index[0])

if Rebal == "Monthly":
    for k in range(1,len(newStrat)): 
        if newStrat.index[k].month != newStrat.index[k-1].month:
            dates_rebal.append(newStrat.index[k])
            underlying_list.append('SPX')
            columns_rename.append('SPX.'+str(newStrat.index[k]))
    """ get the composition for those dates """
    dfbasket = Basket_get_BasketComponents(underlying_list,dates_rebal)
    dfbasket.columns = columns_rename
    for k in range(0,len(columns_rename)):
        tickers = dfbasket[columns_rename[k]].dropna().tolist()
        Baskets[dates_rebal[k]]=tickers
        for i in range(1,len(tickers)):
            all_tickers.append(tickers[i])
            
if Rebal == "Quaterly": 
    for k in range(1,len(newStrat)): 
        if newStrat.index[k].month != newStrat.index[k-1].month:           
            if newStrat.index[k].month in [1,4,7,10]:
                dates_rebal.append(newStrat.index[k])
                underlying_list.append('SPX')
                columns_rename.append('SPX.'+str(newStrat.index[k]))
    """ get the composition for those dates """
    dfbasket = Basket_get_BasketComponents(underlying_list,dates_rebal)
    dfbasket.columns = columns_rename
    for k in range(0,len(columns_rename)):
        tickers = dfbasket[columns_rename[k]].dropna().tolist()
        Baskets[dates_rebal[k]]=tickers
        for i in range(1,len(tickers)):
            all_tickers.append(tickers[i])

if Rebal == "Annual": 
    for k in range(1,len(newStrat)): 
        if newStrat.index[k].year != newStrat.index[k-1].year:  
            dates_rebal.append(newStrat.index[k])
            underlying_list.append('SPX')
            columns_rename.append('SPX.'+str(newStrat.index[k]))
    """ get the composition for those dates """
    dfbasket = Basket_get_BasketComponents(underlying_list,dates_rebal)
    dfbasket.columns = columns_rename
    for k in range(0,len(columns_rename)):
        tickers = dfbasket[columns_rename[k]].dropna().tolist()
        Baskets[dates_rebal[k]]=tickers
        for i in range(1,len(tickers)):
            all_tickers.append(tickers[i])
            

"""initial composition"""
initialBasket  = Basket_get_BasketComponents('SPX',newStrat.index[0])['SPX'].dropna().tolist()
initialBasket = initialBasket[1:len(initialBasket)-1]
Baskets[newStrat.index[0]]=initialBasket 
for i in range(1,len(initialBasket)):
    all_tickers.append(initialBasket[i])
    

""" load all prices """
referencecalendar_prices = DB_BDH_Call('SPX','Last_Price',datetime.datetime(dtBegin.year-1,1,1),dtEnd,False,np.nan).dropna().index.tolist()
dfPrice = DB_BDH_Call(all_tickers,['Last_Price'],dtBegin,dtEnd,False,newStrat.index.tolist())

tickers = dfPrice.columns.tolist()
for k in range(0,len(tickers)):
    tickers[k] = tickers[k][11:len(tickers[k])]
dfPrice.columns = tickers



""" load volatilities """
dfVol = dfPrice.copy()
for column in dfPrice.columns:
    dfVol[column]=ntx_rolling_std(dfPrice,column,30,False)
    
    
    
    
    
    
""" select the top  buybacks over the past years """
dct_BuyBacks = {}
for date in dates_rebal:
    tickers = Baskets[date]
    selection=Buybacks_getAssetBuyBacksTop(initialBasket,datetime.datetime(date.year-1,date.month,date.day),date,iTopBB)
    dct_BuyBacks[date]=selection.index.tolist()
    
""" select the low vol over the past years """    
dct_Vol = {}
for date in dates_rebal:
    tickers = list(dct_BuyBacks[date])
    dfVol_temp = dfVol[dfVol.index <= date].tail(1)
    dfVol_temp = dfVol_temp[tickers]
    tickers = []
    vol=[]
    for k in range(0,len(dfVol_temp.columns)):
        tickers.append(dfVol_temp.columns[k])
        vol.append(dfVol_temp.loc[dfVol_temp.index[0],dfVol_temp.columns[k]])
    dct_Vol[date]= pd.DataFrame({'Tickers':tickers,'Vol':vol}).sort('Vol',ascending = True).head(iTopVol)['Tickers'].tolist()   


""" select the best FCF over the past years """ 
FCF_Tickers = []

for date in dates_rebal:
    tickers = list(dct_Vol[date])
    for ticker in tickers:
        FCF_Tickers.append(ticker)
FCF_Tickers=list(dict.fromkeys(FCF_Tickers).keys()) 
  
df_FCF = DB_BDH_Call(FCF_Tickers,'FCF',datetime.datetime(dtBegin.year-1,1,1),dtEnd,False,np.nan)
df_FCF=df_FCF.fillna(method = 'ffill')   
tickers = df_FCF.columns.tolist()
for k in range(0,len(tickers)):
    tickers[k] = tickers[k][15:len(tickers[k])]
df_FCF.columns =   tickers


dct_FCF={}
for date in dates_rebal:
    tickers = list(dct_Vol[date])
    df_FCF_temp = df_FCF[df_FCF.index <= date].tail(1)
    df_FCF_temp = df_FCF_temp[tickers]
    tickers = []
    FCF=[]
    for k in range(0,len(df_FCF_temp.columns)):
        tickers.append(df_FCF_temp.columns[k])
        FCF.append(df_FCF_temp.loc[df_FCF_temp.index[0],df_FCF_temp.columns[k]])
    dct_FCF[date]= pd.DataFrame({'Tickers':tickers,'FCF':FCF}).sort('FCF',ascending = True).head(iTopFCF)['Tickers'].tolist()   
  
    
newStrat['Index'] = 100.0 


""" Performance dataframe """
dfPerformance = dfPrice  / dfPrice.shift(1) -1
dfPerformance=dfPerformance[dfPerformance.index>=dtBegin]

for k in range(1,len(dates_rebal)):
    Period_begin = dates_rebal[k-1]
    Period_End =  dates_rebal[k]
    icount = 0
    for ticker in dfPerformance.columns:
        if ticker not in dct_FCF[Period_begin]:
            dfPerformance[Period_begin:Period_End][ticker] = 0.0
        else:
            icount+=1
    print(icount)
icount =0
for ticker in dfPerformance.columns:
    if ticker not in dct_FCF[Period_End]:
        dfPerformance[Period_End:dfPerformance.index[len(dfPerformance)-1]][ticker] = 0.0
    else:
        icount+=1
    print(icount)
    
dfPerformance=dfPerformance.fillna(0)            
dfPerformance['Total Performance']     = 0.0 
sum_perf = 0.0       
for ticker in dfPerformance.columns  :  
    sum_perf += dfPerformance[ticker]
sum_perf=sum_perf/iTopFCF +1

newStrat['Index'] = 100 * sum_perf.cumprod()
newStrat['Index'].plot()


dfSPX = DB_BDH_Call('SPX',['Last_Price'],dtBegin,dtEnd,False,newStrat.index.tolist())

dfSPX['Rebased']=100.0
for k in range(1,len(dfSPX)):
    dfSPX['Rebased'][k]=dfSPX['Rebased'][k-1] * dfSPX['Last_Price.SPX'][k] /dfSPX['Last_Price.SPX'][k-1]


newStrat['SPX']=dfSPX['Rebased']
newStrat.plot()


strat= Strategy(newStrat.index,newStrat['Index']).Describe()
strat= Strategy(newStrat.index,newStrat['SPX']).Describe()


newStrat.to_csv(r'C:\Users\tzercher\Desktop\testBuyback.csv')
















