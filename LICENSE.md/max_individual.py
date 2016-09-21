# -*- coding: utf-8 -*-
"""
Created on Tue Aug 09 16:27:38 2016

@author: sjin
"""

import pandas as pd
import sqlite3
import datetime
import numpy as np
import os
import sqlite3
import matplotlib.pyplot as plt
#**********************************
# IMPORT US STRUCTURING LIBRAIRIES
#**********************************
os.chdir(r'C:\Users\sjin\Documents\Local_Code')
DataBase_path = r'C:\Users\sjin\Documents\Local_Code\Database'
Prism_path = r'C:\Users\sjin\Documents\Local_Code\Database\PrismRequest'


import class_CalendarUS 
#########from  DataBaseREAD import *################
from DB_functions import *
from class_Strategy import *
from PortfolioAllocation import *

def Min_Def_Ratio(Covariance_matrix):
    #inequalitys constraints
    cons=({'type': 'eq', 'fun': lambda x:  1 - sum(x)})
   
    #equality constraint 
    bds=[]
    for i in range(0,len(Covariance_matrix)):
        bds.append((0.000000000000001,None))

    #set initial value
    initial_value= np.zeros((len(Covariance_matrix),1),dtype=np.float)
    for i in range(0,len(Covariance_matrix)):
        initial_value[i,0]=1.0/len(Covariance_matrix)

    result =scipy.optimize.minimize(lambda x:-np.dot(np.transpose(x),(np.diagonal(Covariance_matrix)**0.5))/(np.dot(np.transpose(x),np.dot(Covariance_matrix,x))**0.5), x0=initial_value,constraints=cons,bounds=bds).x
    #method="SLSQP",
    optimal_weights=[]
    for i in range(0,len(result)):
        optimal_weights.append(float(result[i]))
    return optimal_weights

def Covariance_matrix2(Calculus_Date,Prices,vol_window,decay_var):
    newStrat=Prices.copy()
    newStrat=newStrat[newStrat.index<=Calculus_Date].tail(vol_window+1)
    Assets = newStrat.columns
    
    Covariance_matrix = np.zeros((len(Assets),len(Assets)),dtype=np.float)
    #compute volatility
    
    decay_list=[]
    for i in range(0,vol_window):
        decay_list.append((1-decay_var)*decay_var**i)
    decay_array=np.array(decay_list)

    for i in range(0,len(Assets)):
        asset=Assets[i]
        Covariance_matrix[i,i]=252*(np.sum(((np.log(newStrat[asset].pct_change()+1))**2).dropna()*decay_array[::-1]))
        
    #compute covariance
    for i in range(0,len(Assets)-1):
        asset1=Assets[i]
        for j in range(i+1,len(Assets)):
            asset2=Assets[j]
            Covariance_matrix[i,j]=252*np.sum((np.log(newStrat[asset2].pct_change()+1)*np.log(newStrat[asset1].pct_change()+1)).dropna()*decay_array[::-1])
            Covariance_matrix[j,i]=Covariance_matrix[i,j]

    #return the covariance matrix
    return Covariance_matrix
#os.chdir(r'K:\ED_ExcelTools\Transfert\Structuring\Proprietary Indices\Python Script\US Structuring Libraries')
#DataBase_path = r'K:\ED_ExcelTools\Transfert\Structuring\Proprietary Indices\Python Script\US Structuring Libraries\Database'
#Prism_path = r'K:\ED_ExcelTools\Transfert\Structuring\Proprietary Indices\Python Script\US Structuring Libraries\Database\PrismRequest'

month_list={1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
decay=0.98
HighLow=0
vol_day=20

for ii in range(1,2):
    newStrat=pd.read_csv(r"C:\Users\sjin\Desktop\diver_data"+str(10)+".csv")
    #newStrat=pd.read_csv(r"C:\Users\sjin\Desktop\diver_data9.csv")
    #newStrat=pd.read_csv(r"C:\Users\sjin\Desktop\diver_data10.csv")
    #newStrat=pd.read_csv(r"C:\Users\sjin\Desktop\diver_data11.csv")
    #newStrat=pd.read_csv(r"C:\Users\sjin\Desktop\diver_data12.csv")
    Amount_Top=len(newStrat.columns)-1
    newStrat_price=pd.DataFrame(columns=['newStrat'])
    newStrat_price.loc[0]=100.0
    rebalance_index=vol_day-30
    newStrat.index=newStrat['Date'].map(lambda x: datetime.datetime.strptime(x,"%m/%d/%Y"))
    diver_ratio=pd.DataFrame(columns=['ratio'])
    del newStrat['Date']
    weight_data=pd.DataFrame(columns=["S"+str(x) for x in range(1,Amount_Top+1)])
    ticker_data=pd.DataFrame(columns=["S"+str(x) for x in range(1,Amount_Top+1)])
    date_series=[]
    rebalance_time=0
    for i in range(vol_day+1,len(newStrat)):
        if (i==vol_day+1):
            monthly_return=np.array(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index]])
            ticker_list=(newStrat.columns[np.argpartition(monthly_return,-Amount_Top)[-Amount_Top:]]) if(HighLow==0) else (newStrat.columns[np.argpartition(monthly_return,Amount_Top-1)[:Amount_Top]])       
            underlying_price=pd.DataFrame()
            for j in ticker_list:
                underlying_price[j]=newStrat[j]
            underlying_price.index=newStrat.index
            rebalance_index=i        
            cov_matrix=Covariance_matrix2(underlying_price.index[i],underlying_price,vol_day,decay)
            weight=np.array(Min_Def_Ratio(cov_matrix))
            newStrat_price.loc[i-vol_day]=newStrat_price.loc[i-vol_day-1]*(1+np.dot(weight,(underlying_price.loc[underlying_price.index[i]]/underlying_price.loc[underlying_price.index[i-1]]-1)))
            diver_ratio.loc[i-vol_day-1]= np.dot(np.transpose(weight),(np.diagonal(cov_matrix)**0.5))/(np.dot(np.transpose(weight),np.dot(cov_matrix,weight))**0.5)    
            weight_data.loc[i-vol_day]=weight
            date_series.append(underlying_price.index[i])
            ticker_data.loc[i-vol_day]=ticker_list
        elif (newStrat.index[i].day!=newStrat.index[i-1].day):
        #(newStrat.index[i].weekday()==4 and newStrat.index[i].day>(month_list[newStrat.index[i].month]-7)): 
        #or (newStrat.index[i].weekday()==1 and newStrat.index[i-1].weekday()!=5 and newStrat.index[i].day>(month_list[newStrat.index[i].month]-7))):
            monthly_return=np.array(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index]])
            ticker_list=(newStrat.columns[np.argpartition(monthly_return,-Amount_Top)[-Amount_Top:]]) if(HighLow==0) else (newStrat.columns[np.argpartition(monthly_return,Amount_Top-1)[:Amount_Top]])  
            underlying_price=pd.DataFrame()
            for j in ticker_list:
                underlying_price[j]=newStrat[j]
            underlying_price.index=newStrat.index
            #rebalance_index=i        
            cov_matrix=Covariance_matrix2(underlying_price.index[i-1],underlying_price,vol_day,decay)
            weight_pot=np.array(Min_Def_Ratio(cov_matrix))
            if sum(abs(weight_pot-weight)>(ii*0.05))>0:
                weight=weight_pot
                newStrat_price.loc[i-vol_day]=newStrat_price.loc[i-vol_day-1]*(1+np.dot(weight,(underlying_price.loc[underlying_price.index[i]]/underlying_price.loc[underlying_price.index[i-1]]-1)))
                rebalance_index=i
                rebalance_time+=1
            else:
                newStrat_price.loc[i-vol_day]=newStrat_price.loc[rebalance_index-vol_day-1]*(1+np.dot(weight,(underlying_price.loc[underlying_price.index[i]]/underlying_price.loc[underlying_price.index[rebalance_index-1]]-1)))
                
#            weight_data.loc[i-vol_day]=weight
#            diver_ratio.loc[i-vol_day-1]= np.dot(np.transpose(weight),(np.diagonal(cov_matrix)**0.5))/(np.dot(np.transpose(weight),np.dot(cov_matrix,weight))**0.5)    
            date_series.append(underlying_price.index[i])
            ticker_data.loc[i-vol_day]=ticker_list
            
        else:
#            weight_spot=weight*underlying_price.loc[underlying_price.index[i-1]]/underlying_price.loc[underlying_price.index[rebalance_index-1]]
#            if sum(abs(weight_spot-weight)>0.05)>0:
#                newStrat_price.loc[i-vol_day]=newStrat_price.loc[i-vol_day-1]*(1+np.dot(weight,(underlying_price.loc[underlying_price.index[i]]/underlying_price.loc[underlying_price.index[i-1]]-1)))
#                rebalance_time+=1
#            else:                
            newStrat_price.loc[i-vol_day]=newStrat_price.loc[rebalance_index-vol_day-1]*(1+np.dot(weight,(underlying_price.loc[underlying_price.index[i]]/underlying_price.loc[underlying_price.index[rebalance_index-1]]-1)))
        #        weight_data.loc[i-vol_day]=weight
        #        diver_ratio.loc[i-vol_day-1]= np.dot(np.transpose(weight),(np.diagonal(cov_matrix)**0.5))/(np.dot(np.transpose(weight),np.dot(cov_matrix,weight))**0.5)    
        
    
    newStrat_price.index=newStrat.index[vol_day:]
#    weight_data.index=date_series
#    diver_ratio.index=date_series
    Strat=Strategy(newStrat_price.index,newStrat_price['newStrat'])
    Strat.Describe()
    #weight_data.to_csv(r"C:\Users\sjin\Desktop\diver_weight.csv")
    #diver_ratio.to_csv(r"C:\Users\sjin\Desktop\diver_ratio.csv")
    Strat.Save(r"C:\Users\sjin\Desktop\Max_test\Trigger_"+str(ii*0.05)+".csv")
    #ticker_data.to_csv(r"C:\Users\sjin\Desktop\tickerFilter.csv")
        #return newStrat_price.loc[newStrat_price.index[-1]][0]
    print rebalance_time
    
    #print diver_return(0.98)
        

    