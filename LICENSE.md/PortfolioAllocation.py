# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 15:32:20 2015

                         PORTFOLIO ALLOCATION FUNCTIONS

@author: tzercher
"""
#*****************************************************************************
#                           IMPORTS
#*****************************************************************************

#*************************
# IMPORT PYTHON LIBRAIRIES
#*************************


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from os import chdir
import sqlite3
import scipy.optimize
from decimal import Decimal
#**********************************
# IMPORT NYC ENGINEERING LIBRAIRIES
#**********************************
os.chdir(r'H:\Local_Code')

from class_CalendarUS import * 
#from DB_functions import *
from class_Strategy import *
from NTX_Stats import *


#**************************************
#** STATISTIC ESTIMATION FUNCTIONS  ***
#**************************************

"""         Covariance_matrix
This function return the covariance matrix
(unbiased estimator, log returns, 0-mean)
"""

def Covariance_matrix(Calculus_Date,Prices,vol_window):
    newStrat=Prices.copy()
    newStrat=newStrat[newStrat.index<=Calculus_Date].tail(vol_window+1)
    Assets = newStrat.columns
    
    Covariance_matrix = np.zeros((len(Assets),len(Assets)),dtype=np.float)
    #compute volatility
    for i in range(0,len(Assets)):
        asset=Assets[i]
        Covariance_matrix[i,i]=252*(vol_window*np.mean((((np.log(newStrat[asset].pct_change()+1))**2)).dropna()/(vol_window-1)))
    #compute covariance
    for i in range(0,len(Assets)-1):
        asset1=Assets[i]
        for j in range(i+1,len(Assets)):
            asset2=Assets[j]
            Covariance_matrix[i,j]=252*vol_window*np.mean((np.log(newStrat[asset2].pct_change()+1)*np.log(newStrat[asset1].pct_change()+1)).dropna())/(vol_window-1)
            Covariance_matrix[j,i]=Covariance_matrix[i,j]

    #return the covariance matrix
    return Covariance_matrix

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


def beta_adjusted_PR(newStrat):
    spx=pd.read_csv(r"H:\Desktop\SPX_data_PR.csv")
    var_name=newStrat.columns[0]
    date_begin=newStrat.index[0]
    date_end=newStrat.index[-1]
    spx.index=spx["Date"].map(lambda x: datetime.datetime.strptime(x,"%m/%d/%Y"))
    del spx["Date"]
    spx=spx.loc[spx.index[spx.index>=date_begin]]
    spx=spx.loc[spx.index[spx.index<=date_end]]
    newStrat=pd.merge(newStrat,spx,how='inner',left_index=True,right_index=True)
    newStrat['Days']=0
    for k in range(1,len(newStrat)):
        newStrat['Days'][k]=(newStrat.index[k]-newStrat.index[k-1]).days
#    newStrat['Days']=1
    newStrat["sum_SPX"]=pd.rolling_sum(newStrat['Days']*np.log(newStrat['SPX Index'].pct_change()+1)**2,window=120)
    newStrat["sum_SPX_SP5"]=pd.rolling_sum(newStrat['Days']*np.log(newStrat['SPX Index'].pct_change()+1)*np.log(newStrat[var_name].pct_change()+1),window=120)
    newStrat["SPX_Vol"]=pd.rolling_mean(newStrat['Days']*np.log(newStrat['SPX Index'].pct_change()+1)**2,window=120)
    newStrat["Cov"]=pd.rolling_mean(newStrat['Days']*np.log(newStrat['SPX Index'].pct_change()+1)*np.log(newStrat[var_name].pct_change()+1),window=120)
    newStrat["Beta"]=pd.rolling_mean(newStrat['Days']*np.log(newStrat['SPX Index'].pct_change()+1)*np.log(newStrat[var_name].pct_change()+1),window=120)/(newStrat["SPX_Vol"])
#    newStrat["Beta"]=1/newStrat["Beta"]
    newStrat["Beta"]=newStrat["Beta"].map(lambda x: min(x,2))
    newStrat["Beta"]=newStrat["Beta"].map(lambda x: max(x,0.5))
    newStrat["IndexBeta"]=100.0
    newStrat['Rebal']=0
    RebalIndex = 121
    while newStrat.index[RebalIndex].month ==newStrat.index[RebalIndex-1].month:
        RebalIndex+=1
    RebalBeta= RebalIndex-1
    iFirstDate=RebalIndex
#    rate =(newStrat["Last_Price."+Rate[0]][RebalIndex]+dBorrow)/(100*365)
    
    for k in range(RebalIndex,len(newStrat)):
        if newStrat.index[k].month!=newStrat.index[k-1].month:
            RebalIndex=k-1
            RebalBeta=k-2
            newStrat['Rebal'][k-1]=1            
            beta =newStrat["Beta"][RebalBeta]
            perf = (newStrat[var_name][k]/newStrat[var_name][RebalIndex]-1)
            borrow_cost=0
#            borrow_cost=  min(0,1-beta)*rate*(newStrat.index[k]-newStrat.index[RebalIndex]).days#/newStrat["IndexBeta"][RebalIndex]
            newStrat["IndexBeta"][k]=newStrat["IndexBeta"][RebalIndex]*(1+perf-beta*(newStrat["SPX Index"][k]/newStrat["SPX Index"][RebalIndex]-1))            

#            rate =(newStrat["Last_Price."+Rate[0]][k-1]+1)/(100*365)
        else:
            beta =newStrat["Beta"][RebalBeta]
            perf = (newStrat[var_name][k]/newStrat[var_name][RebalIndex]-1)
            borrow_cost=0
#            borrow_cost=  min(0,1-beta)*rate*(newStrat.index[k]-newStrat.index[RebalIndex]).days#/newStrat["IndexBeta"][RebalIndex]
            newStrat["IndexBeta"][k]=newStrat["IndexBeta"][RebalIndex]*(1+perf-beta*(newStrat["SPX Index"][k]/newStrat["SPX Index"][RebalIndex]-1))  
                

    newStrat=newStrat[newStrat.index[iFirstDate-1]:newStrat.index[len(newStrat)-1]]
    return newStrat["IndexBeta"]
    


def beta_adjusted_TR(newStrat):
    spx=pd.read_csv(r"H:\Desktop\SPX_data.csv")
    var_name=newStrat.columns[0]
    date_begin=newStrat.index[0]
    date_end=newStrat.index[-1]
    spx.index=spx["Date"].map(lambda x: datetime.datetime.strptime(x,"%m/%d/%Y"))
    del spx["Date"]
    spx=spx.loc[spx.index[spx.index>=date_begin]]
    spx=spx.loc[spx.index[spx.index<=date_end]]
    newStrat=pd.merge(newStrat,spx,how='inner',left_index=True,right_index=True)
    newStrat['Days']=0
    for k in range(1,len(newStrat)):
        newStrat['Days'][k]=(newStrat.index[k]-newStrat.index[k-1]).days
#    newStrat['Days']=1
    newStrat["sum_SPX"]=pd.rolling_sum(newStrat['Days']*np.log(newStrat['SPX Index'].pct_change()+1)**2,window=120)
    newStrat["sum_SPX_SP5"]=pd.rolling_sum(newStrat['Days']*np.log(newStrat['SPX Index'].pct_change()+1)*np.log(newStrat[var_name].pct_change()+1),window=120)
    newStrat["SPX_Vol"]=pd.rolling_mean(newStrat['Days']*np.log(newStrat['SPX Index'].pct_change()+1)**2,window=120)
    newStrat["Cov"]=pd.rolling_mean(newStrat['Days']*np.log(newStrat['SPX Index'].pct_change()+1)*np.log(newStrat[var_name].pct_change()+1),window=120)
    newStrat["Beta"]=pd.rolling_mean(newStrat['Days']*np.log(newStrat['SPX Index'].pct_change()+1)*np.log(newStrat[var_name].pct_change()+1),window=120)/(newStrat["SPX_Vol"])
#    newStrat["Beta"]=1/newStrat["Beta"]
    newStrat["Beta"]=newStrat["Beta"].map(lambda x: min(x,2))
    newStrat["Beta"]=newStrat["Beta"].map(lambda x: max(x,0.5))
    newStrat["IndexBeta"]=100.0
    newStrat['Rebal']=0
    RebalIndex = 121
    while newStrat.index[RebalIndex].month ==newStrat.index[RebalIndex-1].month:
        RebalIndex+=1
    RebalBeta= RebalIndex-1
    iFirstDate=RebalIndex
#    rate =(newStrat["Last_Price."+Rate[0]][RebalIndex]+dBorrow)/(100*365)
    
    for k in range(RebalIndex,len(newStrat)):
        if newStrat.index[k].month!=newStrat.index[k-1].month:
            RebalIndex=k-1
            RebalBeta=k-2
            newStrat['Rebal'][k-1]=1            
            beta =newStrat["Beta"][RebalBeta]
            perf = (newStrat[var_name][k]/newStrat[var_name][RebalIndex]-1)
            borrow_cost=0
#            borrow_cost=  min(0,1-beta)*rate*(newStrat.index[k]-newStrat.index[RebalIndex]).days#/newStrat["IndexBeta"][RebalIndex]
            newStrat["IndexBeta"][k]=newStrat["IndexBeta"][RebalIndex]*(1+perf-beta*(newStrat["SPX Index"][k]/newStrat["SPX Index"][RebalIndex]-1))            

#            rate =(newStrat["Last_Price."+Rate[0]][k-1]+1)/(100*365)
        else:
            beta =newStrat["Beta"][RebalBeta]
            perf = (newStrat[var_name][k]/newStrat[var_name][RebalIndex]-1)
            borrow_cost=0
#            borrow_cost=  min(0,1-beta)*rate*(newStrat.index[k]-newStrat.index[RebalIndex]).days#/newStrat["IndexBeta"][RebalIndex]
            newStrat["IndexBeta"][k]=newStrat["IndexBeta"][RebalIndex]*(1+perf-beta*(newStrat["SPX Index"][k]/newStrat["SPX Index"][RebalIndex]-1))  
                

    newStrat=newStrat[newStrat.index[iFirstDate-1]:newStrat.index[len(newStrat)-1]]
    return newStrat["IndexBeta"]



def top_return_arg(array,num):
    top_index=np.argpartition(array,-num)[-num:]
    other_index=np.argpartition(array,-num)[:-num]
    return (top_index,other_index)
    
def low_return_arg(array,num):
    low_index=np.argpartition(array,num)[:num]
    other_index=np.argpartition(array,num)[num:]
    return (low_index,other_index)

def two_side_return_arg(array,lownum,highnum):
    top_index=top_return_arg(array,highnum)[0]
    low_index=low_return_arg(array,lownum)[0]
    other_list=[]
    for i in range(0,len(array)):
        if (i not in top_index) and (i not in low_index):
            other_list.append(i)
    mid_index=np.array(other_list)
    return (low_index+top_index,mid_index) 

def two_side_filter(array,num,ranking_type):
    if ranking_type=="top":
        return top_return_arg(array,num)
    elif ranking_type=="low":
        return low_return_arg(array,num)
    elif ranking_type=="two side":
        return two_side_return_arg(array,num[0],num[1])
    elif ranking_type=="none":
        return ("empty",range(0,len(array)))
    elif ranking_type=="every X":
        top_list=[]
        low_list=[]
        for i in np.arange(0,int(len(array)),num):
            if array[i]>array[i+1]:
                top_list.append(i)
                low_list.append(i+1)
            else:
                top_list.append(i+1)
                low_list.append(i)
        return (np.array(low_list),np.array(top_list))
    elif ranking_type=="every XX":
        top_list=[]
        low_list=[]
        for i in np.arange(0,int(len(array)),num):
            if array[i]>array[i+1]:
                top_list.append(i)
                low_list.append(i+1)
            else:
                top_list.append(i+1)
                low_list.append(i)
        index2=np.argmin(array[top_list])
        top_list=list(set(top_list)-set([top_list[index2]]))
        low_list=list(set(range(0,len(array)))-set(top_list))
        return (np.array(low_list),np.array(top_list))
    elif ranking_type=="first X":
        index=np.argmax(array[0:num])
        low_list=list(set(range(0,num))-set([index]))
        top_list=list(set(range(0,len(array)))-set(low_list))
        return (np.array(low_list),np.array(top_list))        
    elif ranking_type=="first XX":
        index=np.argmax(array[0:num])
        low_list=list(set(range(0,num))-set([index]))
        top_list=list(set(range(0,len(array)))-set(low_list))
        index2=np.argmin(array[top_list])
        top_list=list(set(top_list)-set([top_list[index2]]))
        low_list=list(set(range(0,len(array)))-set(top_list))
        return (np.array(low_list),np.array(top_list)) 
    elif ranking_type=="Flexible":
        top_list=[]
        index=np.argmax(array[0:num[0]])
        top_list.append(index)
        for i in range(1,len(num)):    
            index=np.argmax(array[num[i-1]:num[i]])
            top_list.append(index+num[i-1])
        low_list=list(set(range(0,len(array)))-set(top_list))
        return (np.array(low_list),np.array(top_list)) 
    elif ranking_type=="Flexible_EW_Type":
        perf_list=np.array([])
        perf_list=np.append(perf_list,np.average(array[0:num[0]]))
        for i in range(1,len(num)):    
            perf_list=np.append(perf_list,np.average(array[num[i-1]:num[i]]))
        index=np.argmin(perf_list)
        if index>0:
            low_list=range(num[index-1],num[index])
        else:
            low_list=range(0,num[0])
        top_list=list(set(range(0,len(array)))-set(low_list))
        return (np.array(low_list),np.array(top_list)) 
"""         Correlation_matrix
This function return the correlation matrix
(unbiased estimator, log returns, 0-mean)
"""
     
def Correlation_matrix(Calculus_Date,Prices,vol_window):
    newStrat=Prices.copy()
    newStrat=newStrat[newStrat.index<=Calculus_Date].tail(vol_window+1)
    Assets = newStrat.columns
    Correlation_matrix = np.zeros((len(Assets),len(Assets)),dtype=np.float)
    #compute volatility
    for i in range(0,len(Assets)):
        asset=Assets[i]
        Correlation_matrix[i,i]=252*(vol_window*np.mean((((np.log(newStrat[asset].pct_change()+1))**2)).dropna()/(vol_window-1)))
    #compute covariance
    for i in range(0,len(Assets)-1):
        asset1=Assets[i]
        for j in range(i+1,len(Assets)):
            asset2=Assets[j]
            Correlation_matrix[i,j]=vol_window*np.mean((np.log(newStrat[asset2].pct_change()+1)*np.log(newStrat[asset1].pct_change()+1)).dropna())/(vol_window-1)
            Correlation_matrix[i,j]=Correlation_matrix[i,j]/(np.sqrt(Correlation_matrix[i,i]*Correlation_matrix[j,j]))
            Correlation_matrix[j,i]=Correlation_matrix[i,j]
    for i in range(0,len(Assets)):
        Correlation_matrix[i,i]=1
    return Correlation_matrix    
    
    
"""         Mean_vector
This function return the mean vector
"""

def Mean_vector(Calculus_Date,Prices,mean_window):
    newStrat=Prices.copy()
    newStrat=newStrat[newStrat.index<=Calculus_Date].tail(mean_window+1)
    Assets = newStrat.columns
    mean_vect = np.zeros((len(Assets),1),dtype=np.float)
    for i in range(0,len(Assets)-1):
        asset1=Assets[i]
        mean_vect[i,0]=np.mean(np.log(newStrat[asset1].pct_change()+1))
    return mean_vect
    

#**************************************
#**         MINIMUM VARIANCE        ***
#**************************************    

def Min_Variance(Covariance_matrix):
    #inequalitys constraints
    cons=({'type': 'eq', 'fun': lambda x:  1 - sum(x)})
   
    #equality constraint 
    bds=[]
    for i in range(0,len(Covariance_matrix)):
        bds.append((0.000000000000001,None))

    #set initial value
    initial_value= np.zeros((len(Covariance_matrix),1),dtype=np.float)
    for i in range(0,len(Covariance_matrix)):
        initial_value[i,0]=1/len(Covariance_matrix)

    result =scipy.optimize.minimize(lambda x:np.dot(np.transpose(x),np.dot(x,Covariance_matrix)), x0=initial_value,method="SLSQP",constraints=cons,bounds=bds).x
    optimal_weights=[]
    for i in range(0,len(result)):
        optimal_weights.append(float(result[i]))
    return optimal_weights
    

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

    result =scipy.optimize.minimize(lambda x:-np.dot(np.transpose(x),(np.diagonal(Covariance_matrix)**0.5))/(np.dot(np.transpose(x),np.dot(Covariance_matrix,x))**0.5), x0=initial_value,method="SLSQP",constraints=cons,bounds=bds).x
    optimal_weights=[]
    for i in range(0,len(result)):
        optimal_weights.append(float(result[i]))
    return optimal_weights

#**************************************
#**    EQUAL RISK CONTRIBUTUTION    ***
#************************************** 
 
def ERP_function_to_minimize(Covariance_matrix,x):
    summation = 0.0
    product = np.dot(Covariance_matrix,x)
    for i in range(0,len(x)):
        for j in range(0,len(x)):
            summation+=(float(x[i])* float(product[i])-float(x[j])* float(product[j]))**2
    return float(summation)
    
def ERC(Covariance_matrix):
    cons=({'type': 'eq', 'fun': lambda x:  1 - sum(x)})
   
    #equality constraint 
    bds=[]
    for i in range(0,len(Covariance_matrix)):
        bds.append((0.000000000000001,0.999999999))
        
     #set initial value
    initial_value= np.zeros((len(Covariance_matrix),1),dtype=np.float)
    initial_value[0,0]=0.5
    initial_value[1,0]=0.5
    result =scipy.optimize.minimize(lambda x:ERP_function_to_minimize(Covariance_matrix,x), x0=initial_value,method="SLSQP",constraints=cons,bounds=bds).x
    optimal_weights=[]
    for i in range(0,len(result)):
        optimal_weights.append(float(result[i]))
    return  optimal_weights

def ERC_Execute_Return_NoneZero(newStrat,Rebalancing_type,MomentumType,Amount_Top,vol_day,day_delay,day_return,threshold,decay,stop_loss,none_zero_day1,none_zero_day2,none_zero_number,none_zero_threshold):
    newStrat_price=pd.DataFrame(columns=['newStrat'])
    newStrat_price.loc[0]=100.0
    rebalance_index=vol_day-30
    rebalance_time=0
    weights_pd=pd.DataFrame(columns=["w" + str(i) for i in range(1,len(newStrat.columns)+1)])

    rebalance_type=0
    if Rebalancing_type=="monthly" or Rebalancing_type=="Monthly" :
        rebalance_type=3
    elif Rebalancing_type=="Quarterly" or Rebalancing_type=="quarterly" :
        rebalance_type=4
    elif Rebalancing_type=="Weekly" or Rebalancing_type=="weekly" :
        rebalance_type=2
    elif Rebalancing_type=="Daily" or Rebalancing_type=="daily" :
        rebalance_type=1   
    
    for i in range(vol_day+day_delay,len(newStrat)):
        if (i==vol_day+day_delay):
            month_return=np.array(newStrat.loc[newStrat.index[i-day_delay]]/newStrat.loc[newStrat.index[i-day_return-day_delay]])
            ticker_index=(np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1]))
            weight=ERC_Weight_Calculator(i,newStrat,ticker_index,vol_day,decay,day_delay)
            rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight
        elif (rebalance_type==1 and newStrat.index[i].day!=newStrat.index[i-1].day) or (rebalance_type==2 and newStrat.index[i].week!=newStrat.index[i-1].week) or (rebalance_type==3 and newStrat.index[i].month!=newStrat.index[i-1].month) or (rebalance_type==4 and newStrat.index[i].month!=newStrat.index[i-1].month and newStrat.index[i].month%3==0) :
            month_return=np.array(newStrat.loc[newStrat.index[i-day_delay]]/newStrat.loc[newStrat.index[i-day_return-day_delay]])
            ticker_index=np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1])
            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-stop_loss) or (sum((newStrat.loc[newStrat.index[i-none_zero_day1]]/newStrat.loc[newStrat.index[i-none_zero_day2]])>1-none_zero_threshold))<none_zero_number:   
                weight_pot=np.zeros(len(newStrat.columns))
            else:
                weight_pot=ERC_Weight_Calculator(i,newStrat,ticker_index,vol_day,decay,day_delay)
            if sum(abs(weight_pot-weight)>(threshold))>0:
                cost=0
                #cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                newStrat_price.loc[i-vol_day-day_delay+1]=(1-cost)*newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
                rebalance_index=i
                rebalance_time+=1
            else:
                newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))               
            weights_pd.loc[i-vol_day-day_delay+1]=weight
        else:
            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-0.9):
                weight_pot=np.zeros(len(newStrat.columns))
                cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight

    newStrat_price.index=newStrat.index[vol_day+day_delay-1:]
    weights_pd.index=newStrat.index[vol_day+day_delay:]
    Strat=Strategy(newStrat_price.index,newStrat_price['newStrat'])
    sharpe_temp=Strat.Sharpe()
    return (sharpe_temp,weights_pd,Strat,newStrat_price)
    
def ERC_Weight_Calculator(i,newStrat,ticker_index,vol_day,decay,day_delay):
    weight=np.zeros(len(newStrat.columns))
    ticker_list=newStrat.columns[ticker_index]    
    underlying_price=pd.DataFrame()
    for j in ticker_list:
        underlying_price[ticker_list]=newStrat[ticker_list]
    underlying_price.index=newStrat.index
    cov_matrix=Covariance_matrix2(underlying_price.index[i-day_delay],underlying_price,vol_day,decay)
    weight[ticker_index]=np.array(ERC(cov_matrix))
    return weight
    
    
    
    
#**************************************
#**    MEAN VARIANCE ALLOCATION     ***
#************************************** 

#// TO DO ......................

#**************************************
#**    OVERLAY ALLOCTION            ***
#************************************** 
def Overlay_Allocation(dfPrice,Index,Overlay,vol_window):
    data= dfPrice.copy()
    data['Vol1']=np.sqrt(252)*pd.rolling_std(np.log(data[Index].pct_change()+1),vol_window)
    data['Vol2']=np.sqrt(252)*pd.rolling_std(np.log(data[Overlay].pct_change()+1),vol_window)
    
    #Allocation 1
    data['W11']=data['Vol2']/(data['Vol2']+data['Vol1'])
    data['W12']=data['Vol1']/(data['Vol2']+data['Vol1'])
    #Allocation 2
    data['W21']=1
    data['W22']=data['Vol1']/data['Vol2']
    data['W22']=data['W22'].fillna(0)
    data['W22']=data['W22'].map(lambda x :max(0,x))
    data['W22']=data['W22'].map(lambda x :min(x,2))
    #quaterly rebalancing
    quaterly=[12,3,6,9]
    data['Rebal']=0
    data['date']=data.index
    for k in range(2,len(data)):
        if data.index[k].month!=data['date'].index[k-1].month:
            if data.index[k].month in quaterly:
                data['Rebal'][k]=1
                
    data['W11 rebal']=0.0
    data['W12 rebal']=0.0
    data['W21 rebal']=0.0
    data['W22 rebal']=0.0
    
    for k in range(2,len(data)):
        if data['Rebal'][k]==1:
            data['W11 rebal'][k]=data['W11'][k]
            data['W12 rebal'][k]=data['W12'][k]
            data['W21 rebal'][k]=data['W21'][k]
            data['W22 rebal'][k]=data['W22'][k]
        else:
            data['W11 rebal'][k]= data['W11 rebal'][k-1]
            data['W12 rebal'][k]= data['W12 rebal'][k-1]
            data['W21 rebal'][k]=  data['W21 rebal'][k-1]
            data['W22 rebal'][k]=data['W22 rebal'][k-1]

    return data

def Overlay_Allocation_execute(dfPrice,Index,Overlay,vol_window):
    data= dfPrice.copy()
    data['Vol1']=np.sqrt(252)*pd.rolling_std(np.log(data[Index].pct_change()+1),vol_window)
    data['Vol2']=np.sqrt(252)*pd.rolling_std(np.log(data[Overlay].pct_change()+1),vol_window)
    
    #Allocation 1
    data['W11']=data['Vol2']/(data['Vol2']+data['Vol1'])
    data['W12']=data['Vol1']/(data['Vol2']+data['Vol1'])
    #Allocation 2
    data['W21']=1
    data['W22']=data['Vol1']/data['Vol2']
    data['W22']=data['W22'].fillna(0)
    data['W22']=data['W22'].map(lambda x :max(0,x))
    data['W22']=data['W22'].map(lambda x :min(x,2))
    #quaterly rebalancing
    quaterly=[12,3,6,9]
    data['Rebal']=0
    data['date']=data.index
    for k in range(2,len(data)):
        if data.index[k].month!=data['date'].index[k-1].month:
            if data.index[k].month in quaterly:
                data['Rebal'][k]=1
                
    data['W11 rebal']=0.0
    data['W12 rebal']=0.0
    data['W21 rebal']=0.0
    data['W22 rebal']=0.0
    
    for k in range(2,len(data)):
        if data['Rebal'][k]==1:
            data['W11 rebal'][k]=data['W11'][k]
            data['W12 rebal'][k]=data['W12'][k]
            data['W21 rebal'][k]=data['W21'][k]
            data['W22 rebal'][k]=data['W22'][k]
        else:
            data['W11 rebal'][k]= data['W11 rebal'][k-1]
            data['W12 rebal'][k]= data['W12 rebal'][k-1]
            data['W21 rebal'][k]=  data['W21 rebal'][k-1]
            data['W22 rebal'][k]=data['W22 rebal'][k-1]
            
    strikeIndex = 83
    data['Allocation']=100.0
    for k in range(84,len(data)):
        perf= data['W22 rebal'][k-1]*(data[Overlay][k]/data[Overlay][strikeIndex]-1)+data['W21 rebal'][k-1]*(data[Index][k]/data[Index][strikeIndex]-1)
        data['Allocation'][k]=data['Allocation'][strikeIndex]*(1+perf)
        if data['Rebal'][k]==1:
            strikeIndex=k
    data=data[data.index>=data.index[83]]        
    return data
    
#**************************************
#**    BETA    ALLOCTION            ***
#**************************************     
    
def Beta_Allocation(dfPrice,Index,Contract_list,Rate,dBorrow):
    newStrat=dfPrice.copy()
    #********************************************
    # OPEN DATABASE, LOAD PRICES,CLOSE DB
    #********************************************
    chdir(r'K:\ED_ExcelTools\Transfert\Structuring\Proprietary Indices\Python Script\US Structuring Libraries')
    lDB=[]
    lCreated_at = []
    for element in os.listdir():
        if element.endswith(".db"):
            lDB.append(element)
            lCreated_at.append(os.path.getmtime(element))  
    database = pd.DataFrame({'File':lDB,'Created_at':lCreated_at})
    database=database.sort('Created_at',ascending = False)
    dbName = database.loc[database.index[0],'File']
    conn = sqlite3.connect(dbName)
    c=conn.cursor()
    #load rates 
    dateBegin=newStrat.index[0]
    dateEnd=newStrat.index[len(newStrat)-1]
    dftemp = getAssetPrices(Rate,dateBegin,dateEnd,c)
    newStrat[Rate]=dftemp['Price']
    newStrat=newStrat.fillna(method='ffill')
    conn.close()
    #********************************************
    #               BETA CALCULUS
    #********************************************
    newStrat['Days']=0
    for k in range(1,len(newStrat)):
        newStrat['Days'][k]=(newStrat.index[k]-newStrat.index[k-1]).days
        
    newStrat[Index+".Vol"]=pd.rolling_mean(newStrat['Days']*np.log(newStrat[Index].pct_change()+1)**2,window=120)
    
    for strat in Contract_list:
        newStrat[strat+".Beta"]=pd.rolling_mean(newStrat['Days']*np.log(newStrat[Index].pct_change()+1)*np.log(newStrat[strat].pct_change()+1),window=120)/(newStrat[Index+".Vol"])
        newStrat[strat+".Beta"]=1/newStrat[strat+".Beta"]
        newStrat[strat+".Beta"]=newStrat[strat+".Beta"].map(lambda x: min(x,2))
        newStrat[strat+".Beta"]=newStrat[strat+".Beta"].map(lambda x: max(x,0.5))
    #***************************************************
    #               MONTHLY REBALANCING BETA RATIO
    #***************************************************

    for strat in Contract_list:
        newStrat[strat+".IndexBeta"]=100.0
    newStrat['Rebal']=0
    RebalIndex = 121
    while newStrat.index[RebalIndex].month ==newStrat.index[RebalIndex-1].month:
        RebalIndex+=1
    RebalBeta= RebalIndex-1
    iFirstDate=RebalIndex
    rate =(newStrat[Rate][RebalIndex]+1)/(100*365)
    
    for k in range(RebalIndex+1,len(newStrat)):
        if newStrat.index[k-1].month!=newStrat.index[k-2].month:
            RebalIndex=k-1
            RebalBeta=k-2
            newStrat['Rebal'][k-1]=1
            rate =(newStrat[Rate][k-1]+dBorrow)/(100*365)
        for strat in Contract_list:
 
            beta =newStrat[strat+".Beta"][RebalBeta]
            perf = beta  * (newStrat[strat][k]/newStrat[strat][RebalIndex]-1)
            borrow_cost=  min(0,1-beta)*rate*(newStrat.index[k]-newStrat.index[RebalIndex]).days#/newStrat[strat+".IndexBeta"][RebalIndex]
            newStrat[strat+".IndexBeta"][k]=newStrat[strat+".IndexBeta"][RebalIndex]*(1+perf+ borrow_cost)
            
    for strat in Contract_list:    
        if strat!=Contract_list[0]:
            del newStrat[strat]
            del newStrat[strat+".Beta"]
    del newStrat[Rate]
    del newStrat[Index+".Vol"]
    del newStrat["Days"] 
    newStrat=newStrat[newStrat.index[iFirstDate]:newStrat.index[len(newStrat)-1]]
    return newStrat    
 
def Beta_Allocation_Short(dfPrice,Index,Contract_list,Rate,dBorrow):
    newStrat=dfPrice.copy()
    #********************************************
    # OPEN DATABASE, LOAD PRICES,CLOSE DB
    #********************************************
    chdir(r'K:\ED_ExcelTools\Transfert\Structuring\Proprietary Indices\Python Script\US Structuring Libraries')
    lDB=[]
    lCreated_at = []
    for element in os.listdir():
        if element.endswith(".db"):
            lDB.append(element)
            lCreated_at.append(os.path.getmtime(element))  
    database = pd.DataFrame({'File':lDB,'Created_at':lCreated_at})
    database=database.sort('Created_at',ascending = False)
    dbName = database.loc[database.index[0],'File']
    conn = sqlite3.connect(dbName)
    c=conn.cursor()
    #load rates 
    dateBegin=newStrat.index[0]
    dateEnd=newStrat.index[len(newStrat)-1]
    dftemp = getAssetPrices(Rate,dateBegin,dateEnd,c)
    newStrat[Rate]=dftemp['Price']
    newStrat=newStrat.fillna(method='ffill')
    conn.close()
    #********************************************
    #               BETA CALCULUS
    #********************************************
    newStrat['Days']=0
    for k in range(1,len(newStrat)):
        newStrat['Days'][k]=(newStrat.index[k]-newStrat.index[k-1]).days
        
    newStrat[Index+".Vol"]=pd.rolling_mean(newStrat['Days']*np.log(newStrat[Index].pct_change()+1)**2,window=120)
    
    for strat in Contract_list:
        newStrat[strat+".Beta"]=pd.rolling_mean(newStrat['Days']*np.log(newStrat[Index].pct_change()+1)*np.log(newStrat[strat].pct_change()+1),window=120)/(newStrat[Index+".Vol"])
        newStrat[strat+".Beta"]=1/newStrat[strat+".Beta"]
        newStrat[strat+".Beta"]=newStrat[strat+".Beta"].map(lambda x: min(x,2))
        newStrat[strat+".Beta"]=newStrat[strat+".Beta"].map(lambda x: max(x,0.5))
    #***************************************************
    #               MONTHLY REBALANCING BETA RATIO
    #***************************************************

    for strat in Contract_list:
        newStrat[strat+".IndexBeta"]=100.0
    newStrat['Rebal']=0
    RebalIndex = 121
    while newStrat.index[RebalIndex].month ==newStrat.index[RebalIndex-1].month:
        RebalIndex+=1
    RebalBeta= RebalIndex-1
    iFirstDate=RebalIndex
    rate =(newStrat[Rate][RebalIndex]+1)/(100*365)
    
    for k in range(RebalIndex+1,len(newStrat)):
        if newStrat.index[k-1].month!=newStrat.index[k-2].month:
            RebalIndex=k-1
            RebalBeta=k-2
            newStrat['Rebal'][k-1]=1
            rate =(newStrat[Rate][k-1]+dBorrow)/(100*365)
        for strat in Contract_list:
 
            beta =newStrat[strat+".Beta"][RebalBeta]
            perf = beta  * (newStrat[strat][k]/newStrat[strat][RebalIndex]-1)
            
            
            borrow_cost=  min(0,1-beta)*rate*(newStrat.index[k]-newStrat.index[RebalIndex]).days#/newStrat[strat+".IndexBeta"][RebalIndex]
            newStrat[strat+".IndexBeta"][k]=newStrat[strat+".IndexBeta"][RebalIndex]*(1-perf+ 0*borrow_cost)
            
    for strat in Contract_list:    
        if strat!=Contract_list[0]:
            del newStrat[strat]
            del newStrat[strat+".Beta"]
    del newStrat[Rate]
    del newStrat[Index+".Vol"]
    del newStrat["Days"] 
    newStrat=newStrat[newStrat.index[iFirstDate]:newStrat.index[len(newStrat)-1]]
    return newStrat    

#**************************************
#**    EQUAL EXPOSURE               ***
#**************************************
def Equal_Exposure_daily(dfPrice,ContractList):
    newStrat = dfPrice.copy()    
    newStrat['Index']=100.0
    coef = 1.0/len(ContractList)
    for k in range(1,len(newStrat.index)):
        perf = 0
        dtToday=newStrat.index[k]
        dtYesterday=newStrat.index[k-1]
        for strat in ContractList:
            perf+=coef*(newStrat.loc[dtToday,strat]/newStrat.loc[dtYesterday,strat]-1)
        newStrat.loc[dtToday,'Index']   =newStrat.loc[dtYesterday,'Index']*(1+perf)
    return  newStrat

    
def Equal_Exposure_monthly(dfPrice,ContractList):
    newStrat = dfPrice.copy()    
    newStrat['Index']=100.0
    coef = 1.0/len(ContractList)
    dtYesterday=newStrat.index[0]
    for k in range(1,len(newStrat.index)):
        if newStrat.index[k].month==newStrat.index[k-1].month:
            perf = 0
            dtToday=newStrat.index[k]
            for strat in ContractList:
                perf+=coef*(newStrat.loc[dtToday,strat]/newStrat.loc[dtYesterday,strat]-1)
            newStrat.loc[dtToday,'Index']   =newStrat.loc[dtYesterday,'Index']*(1+perf)
        else:
            perf = 0
            dtToday=newStrat.index[k]
            dtYesterday=newStrat.index[k-1]
            for strat in ContractList:
                perf+=coef*(newStrat.loc[dtToday,strat]/newStrat.loc[dtYesterday,strat]-1)
            newStrat.loc[dtToday,'Index']   =newStrat.loc[dtYesterday,'Index']*(1+perf)
    return  newStrat

def Equal_Exposure_quarterly(dfPrice,ContractList):
    newStrat = dfPrice.copy()    
    newStrat['Index']=100.0
    coef = 1.0/len(ContractList)
    dtYesterday=newStrat.index[0]
    m_list=[3,6,9,12]
    for k in range(1,len(newStrat.index)):
        if newStrat.index[k].month==newStrat.index[k-1].month and (newStrat.index[k].month in m_list) :
            perf = 0
            dtToday=newStrat.index[k]
            for strat in ContractList:
                perf+=coef*(newStrat.loc[dtToday,strat]/newStrat.loc[dtYesterday,strat]-1)
            newStrat.loc[dtToday,'Index']   =newStrat.loc[dtYesterday,'Index']*(1+perf)
        else:
            perf = 0
            dtToday=newStrat.index[k]
            dtYesterday=newStrat.index[k-1]
            for strat in ContractList:
                perf+=coef*(newStrat.loc[dtToday,strat]/newStrat.loc[dtYesterday,strat]-1)
            newStrat.loc[dtToday,'Index']   =newStrat.loc[dtYesterday,'Index']*(1+perf)
    return  newStrat
    
    
def Equal_Exposure_weekly(dfPrice,ContractList):
    newStrat = dfPrice.copy()    
    newStrat['Index']=100.0
    coef = 1.0/len(ContractList)
    dtYesterday=newStrat.index[0]
    for k in range(1,len(newStrat.index)):
        if newStrat.index[k].week==newStrat.index[k-1].week:
            perf = 0
            dtToday=newStrat.index[k]
            for strat in ContractList:
                perf+=coef*(newStrat.loc[dtToday,strat]/newStrat.loc[dtYesterday,strat]-1)
            newStrat.loc[dtToday,'Index']   =newStrat.loc[dtYesterday,'Index']*(1+perf)
        else:
            perf = 0
            dtToday=newStrat.index[k]
            dtYesterday=newStrat.index[k-1]
            for strat in ContractList:
                perf+=coef*(newStrat.loc[dtToday,strat]/newStrat.loc[dtYesterday,strat]-1)
            newStrat.loc[dtToday,'Index']   =newStrat.loc[dtYesterday,'Index']*(1+perf)
    return  newStrat
    
    
    
    

#**************************************
#**             MOMENTUM            ***
#************************************** 

""" Daily,weekly,monthly or quaterly rebalancing
    Selects the best performing strategy on the last observation period
    Invests in the selected strategy until the end of the period
"""

def Momentum_Allocation(dfPrice,ContractList,Rebalancing_type,iObservation_period):

    newStrat = dfPrice.copy()                
    #MOMENTUM PART 

    perf_vector={}
    for k in range(0,len(ContractList)):
        strat = ContractList[k] 
        perf_vector[strat]=(newStrat[strat]/newStrat[strat].shift(iObservation_period)).shift(1).tolist()

    dfMomentum=pd.DataFrame(index =newStrat.index)
    for k in range(0,len(ContractList)):
            strat = ContractList[k] 
            dfMomentum[strat]=perf_vector[strat]
            
    dfMomentum=dfMomentum.dropna()
    
    dfStrategy1 = dfMomentum.idxmax(axis=1).to_frame()
    dfStrategy1.columns=['Strategy1']

    dfStrategy1['Rebal']=0
    #REBALANCING MANAGEMENT 
    if Rebalancing_type=="Daily" or Rebalancing_type=="daily":
        for k in range(1,len(dfStrategy1)):
            if dfStrategy1.index[k].day != dfStrategy1.index[k-1].day:
                dfStrategy1.loc[dfStrategy1.index[k],'Rebal']=1       
    if Rebalancing_type=="Weekly" or Rebalancing_type=="weekly":
        for k in range(1,len(dfStrategy1)):
            if dfStrategy1.index[k].week != dfStrategy1.index[k-1].week:
                dfStrategy1.loc[dfStrategy1.index[k],'Rebal']=1
                
    if Rebalancing_type=="Monthly" or Rebalancing_type=="monthly":
        for k in range(1,len(dfStrategy1)):
            if dfStrategy1.index[k].month != dfStrategy1.index[k-1].month:
                dfStrategy1.loc[dfStrategy1.index[k],'Rebal']=1
                
    if Rebalancing_type=="Quarterly" or Rebalancing_type=="quarterly":
        quaterly=[12,3,6,9]
        for k in range(1,len(dfStrategy1)):
            if dfStrategy1.index[k].month != dfStrategy1.index[k-1].month:
                if dfStrategy1.index[k].month in quaterly:
                    dfStrategy1.loc[dfStrategy1.index[k],'Rebal']=1
    
    
    
    for i in range(1,len(dfStrategy1)):
        if dfStrategy1.loc[dfStrategy1.index[i],'Rebal']!=1:
            dfStrategy1.loc[dfStrategy1.index[i],'Strategy1']=dfStrategy1.loc[dfStrategy1.index[i-1],'Strategy1']
    dfStrategy1=dfStrategy1['Strategy1'].to_frame()
    return dfStrategy1
    
def Momentum_execute(dfPrice,ContractList,Rebalancing_type,iObservation_period):    
    dfStrategy=Momentum_Allocation(dfPrice,ContractList,Rebalancing_type,iObservation_period)
    newStrat = dfPrice.copy()
    newStrat['Rebal']=0
    #REBALANCING LANAGEMENT 
    if Rebalancing_type=="Daily" or Rebalancing_type=="daily":
        for k in range(1,len(newStrat)):
            if newStrat.index[k].day != newStrat.index[k-1].day:
                newStrat.loc[newStrat.index[k],'Rebal']=1       
    if Rebalancing_type=="Weekly" or Rebalancing_type=="weekly":
        for k in range(1,len(newStrat)):
            if newStrat.index[k].week != newStrat.index[k-1].week:
                newStrat.loc[newStrat.index[k],'Rebal']=1
                
    if Rebalancing_type=="Monthly" or Rebalancing_type=="monthly":
        for k in range(1,len(newStrat)):
            if newStrat.index[k].month != newStrat.index[k-1].month:
                newStrat.loc[newStrat.index[k],'Rebal']=1
                
    if Rebalancing_type=="Quarterly" or Rebalancing_type=="quarterly":
        quaterly=[12,3,6,9]
        for k in range(1,len(newStrat)):
            if newStrat.index[k].month != newStrat.index[k-1].month:
                if newStrat.index[k].month in quaterly:
                    newStrat.loc[newStrat.index[k],'Rebal']=1
                    
    #********************************************
    #    INDEX CALCULUS
    #********************************************
    firstRebal = dfStrategy.index[0]
    rebalDate = firstRebal
    Strategy1 =dfStrategy.loc[rebalDate,'Strategy1']
    newStrat=newStrat[newStrat.index>=rebalDate]   
    newStrat['Strategy_selected']=Strategy1
    newStrat['Index']=100.0
    
    for k in range(1,len(newStrat)):
        dtToday = newStrat.index[k]
        rebalDate= newStrat.index[k-1]
        perf = newStrat.loc[dtToday, Strategy1]/newStrat.loc[rebalDate, Strategy1]-1
        newStrat.loc[dtToday,'Index']= newStrat.loc[rebalDate,'Index']*(1+perf)
        newStrat.loc[newStrat.index[k-1],'Strategy_selected']=Strategy1
        if newStrat.loc[dtToday,'Rebal']==1:
            rebalDate=dtToday
            Strategy1 =dfStrategy.loc[rebalDate,'Strategy1']
    return newStrat
    
    
    

def Sharpe_Momentum_Allocation(dfPrice,ContractList,Rebalancing_type,iObservation_period):
    newStrat = dfPrice.copy()
    newStrat['Rebal']=0
    #REBALANCING MANAGEMENT 
    if Rebalancing_type=="Daily" or Rebalancing_type=="daily":
        for k in range(1,len(newStrat)):
            if newStrat.index[k].day != newStrat.index[k-1].day:
                newStrat.loc[newStrat.index[k],'Rebal']=1       
    if Rebalancing_type=="Weekly" or Rebalancing_type=="weekly":
        for k in range(1,len(newStrat)):
            if newStrat.index[k].week != newStrat.index[k-1].week:
                newStrat.loc[newStrat.index[k],'Rebal']=1
                
    if Rebalancing_type=="Monthly" or Rebalancing_type=="monthly":
        for k in range(1,len(newStrat)):
            if newStrat.index[k].month != newStrat.index[k-1].month:
                newStrat.loc[newStrat.index[k],'Rebal']=1
                
    if Rebalancing_type=="Quarterly" or Rebalancing_type=="quarterly":
        quaterly=[12,3,6,9]
        for k in range(1,len(newStrat)):
            if newStrat.index[k].month != newStrat.index[k-1].month:
                if newStrat.index[k].month in quaterly:
                    newStrat.loc[newStrat.index[k],'Rebal']=1
                    
    #MOMENTUM PART 

    dfMomentum=pd.DataFrame(index =newStrat.index)
    for k in range(0,len(ContractList)):
        for i in range(iObservation_period+2,len(newStrat)):
            strat = ContractList[k]
            if newStrat.loc[newStrat.index[i],'Rebal']==1:                                         
                dfMomentum.loc[dfMomentum.index[i],strat]=Sharpe_Calculator(newStrat.loc[newStrat.index[(i-1-iObservation_period):(i-1)],strat],np.array([1]))
            else:
                dfMomentum.loc[dfMomentum.index[i],strat]=0.0
#    dfMomentum=dfMomentum.dropna()
    
    dfStrategy1 = dfMomentum.idxmax(axis=1,skipna=True).tolist()
#    for i in range(0,len(dfStrategy1)):
#        dfMomentum[dfStrategy1[i]][i]=-100
        
    dfStrategy=pd.DataFrame({'Strategy1':dfStrategy1},index=dfMomentum.index)
    dfStrategy=dfStrategy.dropna()
    return dfStrategy
    
def Sharpe_Momentum_execute(dfPrice,ContractList,Rebalancing_type,iObservation_period):    
    dfStrategy=Sharpe_Momentum_Allocation(dfPrice,ContractList,Rebalancing_type,iObservation_period)
    newStrat = dfPrice.copy()
    newStrat['Rebal']=0
    #REBALANCING LANAGEMENT 
    if Rebalancing_type=="Daily" or Rebalancing_type=="daily":
        for k in range(1,len(newStrat)):
            if newStrat.index[k].day != newStrat.index[k-1].day:
                newStrat.loc[newStrat.index[k],'Rebal']=1       
    if Rebalancing_type=="Weekly" or Rebalancing_type=="weekly":
        for k in range(1,len(newStrat)):
            if newStrat.index[k].week != newStrat.index[k-1].week:
                newStrat.loc[newStrat.index[k],'Rebal']=1
                
    if Rebalancing_type=="Monthly" or Rebalancing_type=="monthly":
        for k in range(1,len(newStrat)):
            if newStrat.index[k].month != newStrat.index[k-1].month:
                newStrat.loc[newStrat.index[k],'Rebal']=1
                
    if Rebalancing_type=="Quarterly" or Rebalancing_type=="quarterly":
        quaterly=[12,3,6,9]
        for k in range(1,len(newStrat)):
            if newStrat.index[k].month != newStrat.index[k-1].month:
                if newStrat.index[k].month in quaterly:
                    newStrat.loc[newStrat.index[k],'Rebal']=1
                    
    #********************************************
    #    INDEX CALCULUS
    #********************************************
    firstRebal = dfStrategy.index[0]
    rebalDate = firstRebal
    Strategy1 =dfStrategy.loc[rebalDate,'Strategy1']
    newStrat=newStrat[newStrat.index>=rebalDate]   
    newStrat['Strategy_selected']=Strategy1
    newStrat['Index']=100.0
    
    for k in range(1,len(newStrat)):
        dtToday = newStrat.index[k]
        rebalDate= newStrat.index[k-1]
        perf = newStrat.loc[dtToday, Strategy1]/newStrat.loc[rebalDate, Strategy1]-1
        newStrat.loc[dtToday,'Index']= newStrat.loc[rebalDate,'Index']*(1+perf)
        newStrat.loc[newStrat.index[k-1],'Strategy_selected']=Strategy1
        if newStrat.loc[dtToday,'Rebal']==1:
            rebalDate=dtToday
            Strategy1 =dfStrategy.loc[rebalDate,'Strategy1']
    return newStrat

#**************************************
#**             RAINBOW             ***
#************************************** 
          

def Rainbow_Allocation(dfPrice,ContractList,Rebalancing_type,iObservation_period,SelectionLength):
    newStrat = dfPrice.copy()
    newStrat['Rebal']=0
    #REBALANCING MANAGEMENT 
    if Rebalancing_type=="Daily" or Rebalancing_type=="daily":
        for k in range(1,len(newStrat)):
            if newStrat.index[k].day != newStrat.index[k-1].day:
                newStrat.loc[newStrat.index[k],'Rebal']=1       
    if Rebalancing_type=="Weekly" or Rebalancing_type=="weekly":
        for k in range(1,len(newStrat)):
            if newStrat.index[k].week != newStrat.index[k-1].week:
                newStrat.loc[newStrat.index[k],'Rebal']=1
                
    if Rebalancing_type=="Monthly" or Rebalancing_type=="monthly":
        for k in range(1,len(newStrat)):
            if newStrat.index[k].month != newStrat.index[k-1].month:
                newStrat.loc[newStrat.index[k],'Rebal']=1
                
    if Rebalancing_type=="Quarterly" or Rebalancing_type=="quarterly":
        quaterly=[12,3,6,9]
        for k in range(1,len(newStrat)):
            if newStrat.index[k].month != newStrat.index[k-1].month:
                if newStrat.index[k].month in quaterly:
                    newStrat.loc[newStrat.index[k],'Rebal']=1
                    
    #MOMENTUM PART 

    perf_vector={}
    for k in range(0,len(ContractList)):
        strat = ContractList[k] 
        perf_vector[strat]=(newStrat[strat]/newStrat[strat].shift(iObservation_period)).shift(1).tolist()

    dfMomentum=pd.DataFrame(index =newStrat.index)
    for k in range(0,len(ContractList)):
            strat = ContractList[k] 
            dfMomentum[strat]=perf_vector[strat]
            
    dfMomentum=dfMomentum.dropna()
    
    dctStrategy={}
    for k in range(0,SelectionLength):
        dctStrategy[k]=dfMomentum.idxmax(axis=1).tolist()
        for i in range(0,len(dctStrategy[k])):
            dfMomentum[dctStrategy[k][i]][i]=-100 # because he want to make the one we already pick up not largest anymore. so he give it -100.
    
    dfStrategy=pd.DataFrame(index=dfMomentum.index)

    for k in range(0,SelectionLength):
        dfStrategy['Strategy'+str(k)]=dctStrategy[k]
    return dfStrategy
    
    
def Rainbow_execute(dfPrice,ContractList,Rebalancing_type,iObservation_period,RainbowCoef):    
    SelectionLength=len(RainbowCoef)
    dfStrategy=Rainbow_Allocation(dfPrice,ContractList,Rebalancing_type,iObservation_period,SelectionLength)
    newStrat = dfPrice.copy()
    newStrat['Rebal']=0
    #REBALANCING LANAGEMENT 
    if Rebalancing_type=="Daily" or Rebalancing_type=="daily":
        for k in range(1,len(newStrat)):
            if newStrat.index[k].day != newStrat.index[k-1].day:
                newStrat.loc[newStrat.index[k],'Rebal']=1       
    if Rebalancing_type=="Weekly" or Rebalancing_type=="weekly":
        for k in range(1,len(newStrat)):
            if newStrat.index[k].week != newStrat.index[k-1].week:
                newStrat.loc[newStrat.index[k],'Rebal']=1
                
    if Rebalancing_type=="Monthly" or Rebalancing_type=="monthly":
        for k in range(1,len(newStrat)):
            if newStrat.index[k].month != newStrat.index[k-1].month:
                newStrat.loc[newStrat.index[k],'Rebal']=1
                
    if Rebalancing_type=="Quarterly" or Rebalancing_type=="quarterly":
        quaterly=[12,3,6,9]
        for k in range(1,len(newStrat)):
            if newStrat.index[k].month != newStrat.index[k-1].month:
                if newStrat.index[k].month in quaterly:
                    newStrat.loc[newStrat.index[k],'Rebal']=1
                    
    #********************************************
    #    INDEX CALCULUS
    #********************************************
    firstRebal = dfStrategy.index[0]
    rebalDate = firstRebal
    dctStrategy={}
    for i in range(0,SelectionLength):
        dctStrategy[i]=dfStrategy.loc[rebalDate,'Strategy'+str(i)]

    newStrat=newStrat[newStrat.index>=rebalDate]    
    for i in range(0,SelectionLength):
         newStrat["Strategy Selected "+ str(i)]=dfStrategy.loc[rebalDate,'Strategy'+str(i)]
    newStrat['Index']=100.0
    
    for k in range(1,len(newStrat)):
        dtToday = newStrat.index[k]
        for i in range(0,SelectionLength):
            newStrat.loc[newStrat.index[k-1],"Strategy Selected "+ str(i)]=dfStrategy.loc[rebalDate,'Strategy'+str(i)]
        perf=0
        for i in range(0,SelectionLength): 
            strat = dctStrategy[i]
            coef = RainbowCoef[i]
            perf += coef*(newStrat.loc[dtToday,  strat]/newStrat.loc[rebalDate,  strat]-1)
        newStrat.loc[dtToday,'Index']= newStrat.loc[rebalDate,'Index']*(1+perf)
        if newStrat.loc[dtToday,'Rebal']==1:
            rebalDate=dtToday
            for i in range(0,SelectionLength):
                dctStrategy[i]=dfStrategy.loc[rebalDate,'Strategy'+str(i)]
    return newStrat    

def Excess_Return(newStrat,Long,Short):    
    calendar =Calendar(datetime.datetime(1990,1,1),datetime.datetime(2020,1,1)) 
    newStrat['ExcessReturn']=100.0
    
    if newStrat.index[0] !=calendar.MonthFirstBusinessDay(newStrat.index[0]):
        StrikeIndex=1
        while newStrat.index[StrikeIndex].month  ==  newStrat.index[StrikeIndex-1].month:
            StrikeIndex+=1
    else:
        StrikeIndex=0
    iFirstDate=StrikeIndex
    
    for k in range(StrikeIndex+1,len(newStrat)):
        perf =newStrat[Long][k]/newStrat[Long][StrikeIndex]-newStrat[Short][k]/newStrat[Short][StrikeIndex]
        newStrat['ExcessReturn'][k]=newStrat['ExcessReturn'][StrikeIndex]*(1+perf)
        if newStrat.index[k].month  !=  newStrat.index[k-1].month:
            StrikeIndex=k
    newStrat=newStrat[newStrat.index[iFirstDate]:newStrat.index[len(newStrat)-1]]
    return newStrat['ExcessReturn']
    
def Add_Long(newStrat,Long1,Long_added):    
    calendar =Calendar(datetime.datetime(1990,1,1),datetime.datetime(2020,1,1)) 
    newStrat['ExcessReturn']=100.0
    
    if newStrat.index[0] !=calendar.MonthFirstBusinessDay(newStrat.index[0]):
        StrikeIndex=1
        while newStrat.index[StrikeIndex].month  ==  newStrat.index[StrikeIndex-1].month:
            StrikeIndex+=1
    else:
        StrikeIndex=0
    iFirstDate=StrikeIndex
    
    for k in range(StrikeIndex+1,len(newStrat)):
        perf =newStrat[Long1][k]/newStrat[Long1][StrikeIndex]+newStrat[Long_added][k]/newStrat[Long_added][StrikeIndex]-2
        newStrat['ExcessReturn'][k]=newStrat['ExcessReturn'][StrikeIndex]*(1+perf)
        if newStrat.index[k].month  !=  newStrat.index[k-1].month:
            StrikeIndex=k
    newStrat=newStrat[newStrat.index[iFirstDate]:newStrat.index[len(newStrat)-1]]
    return newStrat
         
         
def VaR(Return,weight,alpha):
    Return=Return.dropna()
    perf=np.dot(Return,np.transpose(weight))
    port_VaR=-np.percentile(perf,(float(Decimal('1.0')-Decimal(str(alpha)))*100.0))
    return port_VaR

def CVaR(Return,weight,alpha):
    port_VaR=-VaR(Return,weight,alpha)
    length=len(Return.index)
    asset_num=len(Return.columns)
    excess_return=np.dot(weight,np.transpose(Return.as_matrix()))-np.ones((1,length))*port_VaR
    out_return=excess_return[excess_return<0]
    port_CVaR=port_VaR+(1.0/length)*(1.0/(1.0-alpha))*sum(out_return)
    return -port_CVaR
    

def CVaROptimization(ScenRets, R0, VaR0=None, beta=None,  UB=None, LB=None):
#    %
#    %
#    % The function estimates the optimal portfolio weights that minimize CVaR
#    % under a given target return R0
#    %
#    %INPUTS: ScenRets: Portfolio returns matrix
#    %       R0: The target return
#    %       beta:The confidence level between 0.9 and 0.999
#    %       LB, UB the upper and lower bound for the optimal weights. For example If
#    %       you imput UB=.25 none of the stocks can consist more than the 25% of the
#    %       portfolio. 
#    %       VaR0= the initial guess for the portfolio VaR
#    %
#    %OUTPUTS: fval = CVaR of the optimal portfolio
#    %         w= the weights of the optimal portfolio, The last element in w
#    %         equals the VaR of the optimal portfolio
#    %
#    %---------------- INPUT ARGUMENTS--------------------------------------
#    % The function accepts 6 imputs however only the two first are required
#    % If you dont supply the 6 argument then LB=0 (no short positions)
#    % If you dont supply the 5 argument then UB=1
#    % If you dont supply the 4 argument then beta=0.95
#    % If you dont supply the 3 argument VaR0 equals the HS VaR of the equally weighted portfolio

    J=len(ScenRets.index)
    nAssets=len(ScenRets.columns)
    w0=(1/nAssets)*np.ones((1,nAssets))
    if LB is None:
        LB=0.0;
    if UB is None:
        UB=1.0;
    if beta is None:
       beta=0.95;
    if VaR0 is None:
       VaR0=VaR(ScenRets,w0,beta);
    if beta>1 or beta<0.9:
        print('The confidence level beta = 1 - alpha, should be in (0.9 0.99)')
    if LB>=UB:
        print('The LB has to be smaller than UB')
    if UB>1:
        print('The upper bound should be less than 1')
    if LB<-1:
        print('The lower bound should be greater than -1')

    i=range(1,nAssets)
    
#    % the objective function
    cons=({'type': 'eq', 'fun': lambda x: 1 - sum(x)},
           {'type':'ineq','fun':lambda x: np.dot(x,np.transpose(np.array(ScenRets.mean(0))))-R0})
    bds=[]
    for i in range(0,nAssets):
        bds.append((0.000001,0.9999))
    initial_value= np.ones((1,nAssets),dtype=np.float)*(1.0/nAssets)

    result =scipy.optimize.minimize(lambda x:CVaR(ScenRets,x,beta), x0=initial_value,constraints=cons,bounds=bds).x
    optimal_weights=[]
    for i in range(0,len(result)):
        optimal_weights.append(float(result[i]))

    return optimal_weights
    
#############################################################
#############################################################
#############################################################




def Maximum_Diverisification_Strategy_MOZAIC_StopLoss(newStrat,Rebalancing_type,MomentumType,Amount_Top,vol_day,day_delay,day_return,threshold,decay,stop_loss):
    newStrat_price=pd.DataFrame(columns=['newStrat'])
    newStrat_price.loc[0]=100.0
    rebalance_index=vol_day-30
    rebalance_time=0
    weights_pd=pd.DataFrame(columns=["w" + str(i) for i in range(1,len(newStrat.columns)+1)])
    countDown=0
    rebalance_type=0
    if Rebalancing_type=="monthly" or Rebalancing_type=="Monthly" :
        rebalance_type=3
    elif Rebalancing_type=="Quarterly" or Rebalancing_type=="quarterly" :
        rebalance_type=4
    elif Rebalancing_type=="Weekly" or Rebalancing_type=="weekly" :
        rebalance_type=2
    elif Rebalancing_type=="Daily" or Rebalancing_type=="daily" :
        rebalance_type=1   
    
    for i in range(vol_day+day_delay,len(newStrat)):
        if (i==vol_day+day_delay):
            month_return=np.array(newStrat.loc[newStrat.index[i-day_delay]]/newStrat.loc[newStrat.index[i-day_return-day_delay]])
            ticker_index=np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1])
            weight=MD_weight_calculation(i,newStrat,ticker_index,vol_day,decay,day_delay)
            rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight
        elif  countDown==0 or (countDown==-1 and (rebalance_type==1 and newStrat.index[i].day!=newStrat.index[i-1].day) or (rebalance_type==2 and newStrat.index[i].week!=newStrat.index[i-1].week) or (rebalance_type==3 and newStrat.index[i].month!=newStrat.index[i-1].month) or (rebalance_type==4 and newStrat.index[i].month!=newStrat.index[i-1].month and newStrat.index[i].month%3==0)) :
            month_return=np.array(newStrat.loc[newStrat.index[i-day_delay]]/newStrat.loc[newStrat.index[i-day_return-day_delay]])
            ticker_index=np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1])
            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-stop_loss):
                weight_pot=np.zeros(len(newStrat.columns))
                countDown=5
            else:
                weight_pot=MD_weight_calculation(i,newStrat,ticker_index,vol_day,decay,day_delay)
            if sum(abs(weight_pot-weight)>(threshold))>0:
                cost=0
                #cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                newStrat_price.loc[i-vol_day-day_delay+1]=(1-cost)*newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
                rebalance_index=i
                rebalance_time+=1
            else:
                newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))               
            weights_pd.loc[i-vol_day-day_delay+1]=weight
            if countDown>-1:
                countDown=countDown-1
        else:
            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-stop_loss):
                weight_pot=np.zeros(len(newStrat.columns))
                #cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                rebalance_index=i
                countDown=5
            if countDown>-1:
                countDown=countDown-1
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight

    newStrat_price.index=newStrat.index[vol_day+day_delay-1:]
    weights_pd.index=newStrat.index[vol_day+day_delay:]
    Strat=Strategy(newStrat_price.index,newStrat_price['newStrat'])
    sharpe_temp=Strat.Sharpe()
    return (sharpe_temp,weights_pd,Strat,newStrat_price)


def Maximum_Diverisification_Strategy_Rules_book(newStrat,Rebalancing_type,MomentumType,Amount_Top,vol_day,day_delay,day_return,threshold,decay,stop_loss):
    newStrat_price=pd.DataFrame(columns=['newStrat'])
    newStrat_price.loc[0]=100.0
    rebalance_index=vol_day-30
    rebalance_time=0
    weights_pd=pd.DataFrame(columns=["w" + str(i) for i in range(1,len(newStrat.columns)+1)])

    rebalance_type=0
    if Rebalancing_type=="monthly" or Rebalancing_type=="Monthly" :
        rebalance_type=3
    elif Rebalancing_type=="Quarterly" or Rebalancing_type=="quarterly" :
        rebalance_type=4
    elif Rebalancing_type=="Weekly" or Rebalancing_type=="weekly" :
        rebalance_type=2
    elif Rebalancing_type=="Daily" or Rebalancing_type=="daily" :
        rebalance_type=1   
    
    for i in range(vol_day+day_delay,len(newStrat)):
        if (i==vol_day+day_delay):
            month_return=np.array(newStrat.loc[newStrat.index[i-day_delay]]/newStrat.loc[newStrat.index[i-day_return-day_delay]])
            ticker_index=np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[0])
            weight=MD_weight_calculation_Rules_book(i,newStrat,ticker_index,vol_day,decay,day_delay)
            rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight
        elif (rebalance_type==1 and newStrat.index[i].day!=newStrat.index[i-1].day) or (rebalance_type==2 and newStrat.index[i].week!=newStrat.index[i-1].week) or (rebalance_type==3 and newStrat.index[i].month!=newStrat.index[i-1].month) or (rebalance_type==4 and newStrat.index[i].month!=newStrat.index[i-1].month and newStrat.index[i].month%3==0) :
            month_return=np.array(newStrat.loc[newStrat.index[i-day_delay]]/newStrat.loc[newStrat.index[i-day_return-day_delay]])
            ticker_index=np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[0])
            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-stop_loss):
                weight_pot=np.zeros(len(newStrat.columns))
            else:
                weight_pot=MD_weight_calculation_Rules_book(i,newStrat,ticker_index,vol_day,decay,day_delay)
            if sum(abs(weight_pot-weight)>(threshold))>0:
                cost=0
                #cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                newStrat_price.loc[i-vol_day-day_delay+1]=(1-cost)*newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
                rebalance_index=i
                rebalance_time+=1
            else:
                newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))               
            weights_pd.loc[i-vol_day-day_delay+1]=weight
        else:
            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-0.9):
                weight_pot=np.zeros(len(newStrat.columns))
                cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight

    newStrat_price.index=newStrat.index[vol_day+day_delay-1:]
    weights_pd.index=newStrat.index[vol_day+day_delay:]
    Strat=Strategy(newStrat_price.index,newStrat_price['newStrat'])
    sharpe_temp=Strat.Sharpe()
    return (sharpe_temp,weights_pd,Strat,newStrat_price)


#(newStrat1[newStrat1.columns[[0,1,3,7]]],"weekly","low",1,120,4,1,0.1,0.95,0.02,2,5,1,0.001*0)
def Maximum_Diverisification_Strategy_Return_NoneZero(newStrat,Rebalancing_type,MomentumType,Amount_Top,vol_day,day_delay,day_return,threshold,decay,stop_loss,none_zero_day1,none_zero_day2,none_zero_number,none_zero_threshold):
    newStrat_price=pd.DataFrame(columns=['newStrat'])
    newStrat_price.loc[0]=100.0
    rebalance_index=vol_day-30
    rebalance_time=0
    weights_pd=pd.DataFrame(columns=["w" + str(i) for i in range(1,len(newStrat.columns)+1)])

    rebalance_type=0
    if Rebalancing_type=="monthly" or Rebalancing_type=="Monthly" :
        rebalance_type=3
    elif Rebalancing_type=="Quarterly" or Rebalancing_type=="quarterly" :
        rebalance_type=4
    elif Rebalancing_type=="Weekly" or Rebalancing_type=="weekly" :
        rebalance_type=2
    elif Rebalancing_type=="Daily" or Rebalancing_type=="daily" :
        rebalance_type=1   
    
    for i in range(vol_day+day_delay,len(newStrat)):
        if (i==vol_day+day_delay):
            month_return=np.array(newStrat.loc[newStrat.index[i-day_delay]]/newStrat.loc[newStrat.index[i-day_return-day_delay]])
            ticker_index=(np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[0]))
            weight=MD_weight_calculation_Rules_book(i,newStrat,ticker_index,vol_day,decay,day_delay)
            rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight
        elif (rebalance_type==1 and newStrat.index[i].day!=newStrat.index[i-1].day) or (rebalance_type==2 and newStrat.index[i].week!=newStrat.index[i-1].week) or (rebalance_type==3 and newStrat.index[i].month!=newStrat.index[i-1].month) or (rebalance_type==4 and newStrat.index[i].month!=newStrat.index[i-1].month and newStrat.index[i].month%3==0) :
            month_return=np.array(newStrat.loc[newStrat.index[i-day_delay]]/newStrat.loc[newStrat.index[i-day_return-day_delay]])
            ticker_index=np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[0])
            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-stop_loss) or (sum((newStrat.loc[newStrat.index[i-none_zero_day1]]/newStrat.loc[newStrat.index[i-none_zero_day2]])>1-none_zero_threshold))<none_zero_number:   
                weight_pot=np.zeros(len(newStrat.columns))
            else:
                weight_pot=MD_weight_calculation_Rules_book(i,newStrat,ticker_index,vol_day,decay,day_delay)
            if sum(abs(weight_pot-weight)>(threshold))>0:
                cost=0
                #cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                newStrat_price.loc[i-vol_day-day_delay+1]=(1-cost)*newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
                rebalance_index=i
                rebalance_time+=1
            else:
                newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))               
            weights_pd.loc[i-vol_day-day_delay+1]=weight
        else:
            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-0.9):
                weight_pot=np.zeros(len(newStrat.columns))
                cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight

    newStrat_price.index=newStrat.index[vol_day+day_delay-1:]
    weights_pd.index=newStrat.index[vol_day+day_delay:]
    Strat=Strategy(newStrat_price.index,newStrat_price['newStrat'])
    sharpe_temp=Strat.Sharpe()
    return (sharpe_temp,weights_pd,Strat,newStrat_price)


def Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec(newStrat,Rebalancing_type,MomentumType,Amount_Top,vol_day,day_delay,day_return,threshold,decay,stop_loss,none_zero_day1,none_zero_day2,none_zero_number,none_zero_threshold):
    newStrat_price=pd.DataFrame(columns=['newStrat'])
    newStrat_price.loc[0]=100.0
    rebalance_index=vol_day-30
    rebalance_time=0
    weights_pd=pd.DataFrame(columns=["w" + str(i) for i in range(1,len(newStrat.columns)+1)])

    rebalance_type=0
    if Rebalancing_type=="monthly" or Rebalancing_type=="Monthly" :
        rebalance_type=3
    elif Rebalancing_type=="Quarterly" or Rebalancing_type=="quarterly" :
        rebalance_type=4
    elif Rebalancing_type=="Weekly" or Rebalancing_type=="weekly" :
        rebalance_type=2
    elif Rebalancing_type=="Daily" or Rebalancing_type=="daily" :
        rebalance_type=1 
        
    for i in range(vol_day+day_delay,len(newStrat)):
        if (i==vol_day+day_delay):
            month_return=np.array(newStrat.loc[newStrat.index[i-day_delay]]/newStrat.loc[newStrat.index[i-day_return-day_delay]])
            ticker_index=(np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1]))
            weight=MD_weight_calculation(i,newStrat,ticker_index,vol_day,decay,day_delay)
            rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight
        elif (rebalance_type==1 and newStrat.index[i].day!=newStrat.index[i-1].day) or (rebalance_type==2 and newStrat.index[i].week!=newStrat.index[i-1].week) or (rebalance_type==3 and newStrat.index[i].month!=newStrat.index[i-1].month) or (rebalance_type==4 and newStrat.index[i].month!=newStrat.index[i-1].month and newStrat.index[i].month%3==0) :
            month_return=np.array(newStrat.loc[newStrat.index[i-day_delay]]/newStrat.loc[newStrat.index[i-day_return-day_delay]])
            ticker_index=np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1])
            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-stop_loss) or (sum((newStrat.loc[newStrat.index[i-none_zero_day1]]/newStrat.loc[newStrat.index[i-none_zero_day2]])>1-none_zero_threshold))<none_zero_number:   
                weight_pot=np.zeros(len(newStrat.columns))
            else:
                weight_pot=MD_weight_calculation(i,newStrat,ticker_index,vol_day,decay,day_delay)
            if sum(abs(weight_pot-weight)>(threshold))>0:
                cost=0
                #cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                newStrat_price.loc[i-vol_day-day_delay+1]=(1-cost)*newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
                rebalance_index=i
                rebalance_time+=1
            else:
                newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))               
            weights_pd.loc[i-vol_day-day_delay+1]=weight
        else:
            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-0.9):
                weight_pot=np.zeros(len(newStrat.columns))
                cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight

    newStrat_price.index=newStrat.index[vol_day+day_delay-1:]
    weights_pd.index=newStrat.index[vol_day+day_delay:]
    Strat=Strategy(newStrat_price.index,newStrat_price['newStrat'])
    sharpe_temp=Strat.Sharpe()
    return (sharpe_temp,weights_pd,Strat,newStrat_price)


def Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec_Individual_momentum(newStrat,Rebalancing_type,MomentumType,Amount_Top,vol_day,day_delay,day_return,threshold,decay,stop_loss,none_zero_day1,none_zero_day2,none_zero_number,none_zero_threshold,Rebalancing_type2,iObservation_period):
    newStrat_price=pd.DataFrame(columns=['newStrat'])
    newStrat_price.loc[0]=100.0
    rebalance_index=vol_day-30
    rebalance_time=0
    weights_pd=pd.DataFrame(columns=["w" + str(i) for i in range(1,len(newStrat.columns)+1)])

    rebalance_type=0
    if Rebalancing_type=="monthly" or Rebalancing_type=="Monthly" :
        rebalance_type=3
    elif Rebalancing_type=="Quarterly" or Rebalancing_type=="quarterly" :
        rebalance_type=4
    elif Rebalancing_type=="Weekly" or Rebalancing_type=="weekly" :
        rebalance_type=2
    elif Rebalancing_type=="Daily" or Rebalancing_type=="daily" :
        rebalance_type=1 
        
    dfStrategy=Momentum_Allocation(newStrat,newStrat.columns[0:2],Rebalancing_type2,iObservation_period)
        
        
    for i in range(vol_day+day_delay,len(newStrat)):
        if (i==vol_day+day_delay):
            month_return=np.array(newStrat.loc[newStrat.index[i-day_delay]]/newStrat.loc[newStrat.index[i-day_return-day_delay]])
            ticker_index=(np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1]))
            weight=MD_weight_calculation(i,newStrat,ticker_index,vol_day,decay,day_delay)
            rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight
        elif (rebalance_type==1 and newStrat.index[i].day!=newStrat.index[i-1].day) or (rebalance_type==2 and newStrat.index[i].week!=newStrat.index[i-1].week) or (rebalance_type==3 and newStrat.index[i].month!=newStrat.index[i-1].month) or (rebalance_type==4 and newStrat.index[i].month!=newStrat.index[i-1].month and newStrat.index[i].month%3==0) :
            month_return=np.array(newStrat.loc[newStrat.index[i-day_delay]]/newStrat.loc[newStrat.index[i-day_return-day_delay]])
            index1=newStrat.columns.get_loc(dfStrategy.loc[newStrat.index[i-1],"Strategy1"])
            month_return=np.delete(month_return,1-index1)
            ticker_index=np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1])
            for m in range(0,len(ticker_index)):
                if ticker_index[m]>=(1-index1):
                    ticker_index[m]+=1
            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-stop_loss) or (sum((newStrat.loc[newStrat.index[i-none_zero_day1]]/newStrat.loc[newStrat.index[i-none_zero_day2]])>1-none_zero_threshold))<none_zero_number:   
                weight_pot=np.zeros(len(newStrat.columns))
            else:
                weight_pot=MD_weight_calculation(i,newStrat,ticker_index,vol_day,decay,day_delay)
            if sum(abs(weight_pot-weight)>(threshold))>0:
                cost=0
                #cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                newStrat_price.loc[i-vol_day-day_delay+1]=(1-cost)*newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
                rebalance_index=i
                rebalance_time+=1
            else:
                newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))               
            weights_pd.loc[i-vol_day-day_delay+1]=weight
        else:
            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-0.9):
                weight_pot=np.zeros(len(newStrat.columns))
                cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight

    newStrat_price.index=newStrat.index[vol_day+day_delay-1:]
    weights_pd.index=newStrat.index[vol_day+day_delay:]
    Strat=Strategy(newStrat_price.index,newStrat_price['newStrat'])
    sharpe_temp=Strat.Sharpe()
    return (sharpe_temp,weights_pd,Strat,newStrat_price)
    

def Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec_Individual_momentum_everyX(newStrat,Rebalancing_type,MomentumType,Amount_Top,vol_day,day_delay,day_return,threshold,decay,stop_loss,none_zero_day1,none_zero_day2,none_zero_number,none_zero_threshold,Rebalancing_type2,iObservation_period):
    newStrat_price=pd.DataFrame(columns=['newStrat'])
    newStrat_price.loc[0]=100.0
    rebalance_index=vol_day-30
    rebalance_time=0
    weights_pd=pd.DataFrame(columns=["w" + str(i) for i in range(1,len(newStrat.columns)+1)])

    rebalance_type=0
    if Rebalancing_type=="monthly" or Rebalancing_type=="Monthly" :
        rebalance_type=3
    elif Rebalancing_type=="Quarterly" or Rebalancing_type=="quarterly" :
        rebalance_type=4
    elif Rebalancing_type=="Weekly" or Rebalancing_type=="weekly" :
        rebalance_type=2
    elif Rebalancing_type=="Daily" or Rebalancing_type=="daily" :
        rebalance_type=1 
         
    rebalancing_count_down=0
    for i in range(vol_day+day_delay,len(newStrat)):
        if (i==vol_day+day_delay):
            month_return=np.array(newStrat.loc[newStrat.index[i-day_delay]]/newStrat.loc[newStrat.index[i-day_return-day_delay]])
            ticker_index=(np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1]))
            weight=MD_weight_calculation(i,newStrat,ticker_index,vol_day,decay,day_delay)
            rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight
        elif (rebalance_type==1 and newStrat.index[i].day!=newStrat.index[i-1].day) or (rebalance_type==2 and newStrat.index[i].week!=newStrat.index[i-1].week) or (rebalance_type==3 and newStrat.index[i].month!=newStrat.index[i-1].month) or (rebalance_type==4 and newStrat.index[i].month!=newStrat.index[i-1].month and newStrat.index[i].month%3==0) :
            if rebalancing_count_down<=0:
                month_return=np.array(newStrat.loc[newStrat.index[i-day_delay]]/newStrat.loc[newStrat.index[i-day_return-day_delay]])
                ticker_index=np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1])
                rebalancing_count_down=Rebalancing_type2
            
            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-stop_loss) or (sum((newStrat.loc[newStrat.index[i-none_zero_day1]]/newStrat.loc[newStrat.index[i-none_zero_day2]])>1-none_zero_threshold))<none_zero_number:   
                weight_pot=np.zeros(len(newStrat.columns))
            else:
                weight_pot=MD_weight_calculation(i,newStrat,ticker_index,vol_day,decay,day_delay)
            if sum(abs(weight_pot-weight)>(threshold))>0:
                cost=0
                #cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                newStrat_price.loc[i-vol_day-day_delay+1]=(1-cost)*newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
                rebalance_index=i
                rebalance_time+=1
            else:
                newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))               
            weights_pd.loc[i-vol_day-day_delay+1]=weight
            rebalancing_count_down-=1
        else:
            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-0.9):
                weight_pot=np.zeros(len(newStrat.columns))
                cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight

    newStrat_price.index=newStrat.index[vol_day+day_delay-1:]
    weights_pd.index=newStrat.index[vol_day+day_delay:]
    Strat=Strategy(newStrat_price.index,newStrat_price['newStrat'])
    sharpe_temp=Strat.Sharpe()
    return (sharpe_temp,weights_pd,Strat,newStrat_price)
    
#
#def Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec_Individual_momentum_everyX__stoploss_everyday(newStrat,Rebalancing_type,MomentumType,Amount_Top,vol_day,day_delay,day_return,threshold,decay,stop_loss,none_zero_day1,none_zero_day2,none_zero_number,none_zero_threshold,Rebalancing_type2,iObservation_period):
#    newStrat_price=pd.DataFrame(columns=['newStrat'])
#    newStrat_price.loc[0]=100.0
#    rebalance_index=vol_day-30
#    rebalance_time=0
#    weights_pd=pd.DataFrame(columns=["w" + str(i) for i in range(1,len(newStrat.columns)+1)])
#
#    rebalance_type=0
#    if Rebalancing_type=="monthly" or Rebalancing_type=="Monthly" :
#        rebalance_type=3
#    elif Rebalancing_type=="Quarterly" or Rebalancing_type=="quarterly" :
#        rebalance_type=4
#    elif Rebalancing_type=="Weekly" or Rebalancing_type=="weekly" :
#        rebalance_type=2
#    elif Rebalancing_type=="Daily" or Rebalancing_type=="daily" :
#        rebalance_type=1 
#         
#    rebalancing_count_down=0
#    for i in range(vol_day+day_delay,len(newStrat)):
#        if (i==vol_day+day_delay):
#            month_return=np.array(newStrat.loc[newStrat.index[i-day_delay]]/newStrat.loc[newStrat.index[i-day_return-day_delay]])
#            ticker_index=(np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1]))
#            weight=MD_weight_calculation(i,newStrat,ticker_index,vol_day,decay,day_delay)
#            rebalance_index=i
#            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
#            weights_pd.loc[i-vol_day-day_delay+1]=weight
#        elif (rebalance_type==1 and newStrat.index[i].day!=newStrat.index[i-1].day) or (rebalance_type==2 and newStrat.index[i].week!=newStrat.index[i-1].week) or (rebalance_type==3 and newStrat.index[i].month!=newStrat.index[i-1].month) or (rebalance_type==4 and newStrat.index[i].month!=newStrat.index[i-1].month and newStrat.index[i].month%3==0) :
#            if rebalancing_count_down<=0:
#                month_return=np.array(newStrat.loc[newStrat.index[i-day_delay]]/newStrat.loc[newStrat.index[i-day_return-day_delay]])
#                ticker_index=np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1])
#                rebalancing_count_down=Rebalancing_type2
#            
#            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-stop_loss) or (sum((newStrat.loc[newStrat.index[i-none_zero_day1]]/newStrat.loc[newStrat.index[i-none_zero_day2]])>1-none_zero_threshold))<none_zero_number:   
#                weight_pot=np.zeros(len(newStrat.columns))
#            else:
#                weight_pot=MD_weight_calculation(i,newStrat,ticker_index,vol_day,decay,day_delay)
#            if sum(abs(weight_pot-weight)>(threshold))>0:
#                cost=0
#                #cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
#                weight=weight_pot
#                newStrat_price.loc[i-vol_day-day_delay+1]=(1-cost)*newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
#                rebalance_index=i
#                rebalance_time+=1
#            else:
#                newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))               
#            weights_pd.loc[i-vol_day-day_delay+1]=weight
#            rebalancing_count_down-=1
#        else:
#            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-stop_loss):
#                weight_pot=np.zeros(len(newStrat.columns))
#                cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
#                weight=weight_pot
#                rebalance_index=i
#            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))
#            weights_pd.loc[i-vol_day-day_delay+1]=weight
#
#    newStrat_price.index=newStrat.index[vol_day+day_delay-1:]
#    weights_pd.index=newStrat.index[vol_day+day_delay:]
#    Strat=Strategy(newStrat_price.index,newStrat_price['newStrat'])
#    sharpe_temp=Strat.Sharpe()
#    return (sharpe_temp,weights_pd,Strat,newStrat_price)
    



#def Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec_Individual_momentum_everyX__stoploss_everyday_underindex(newStrat,Rebalancing_type,MomentumType,Amount_Top,vol_day,day_delay,day_return,threshold,decay,stop_loss,none_zero_day1,none_zero_day2,none_zero_number,none_zero_threshold,Rebalancing_type2,iObservation_period):
#    newStrat_price=pd.DataFrame(columns=['newStrat'])
#    newStrat_price.loc[0]=100.0
#    rebalance_index=vol_day-30
#    rebalance_time=0
#    weights_pd=pd.DataFrame(columns=["w" + str(i) for i in range(1,len(newStrat.columns)+1)])
#
#    rebalance_type=0
#    if Rebalancing_type=="monthly" or Rebalancing_type=="Monthly" :
#        rebalance_type=3
#    elif Rebalancing_type=="Quarterly" or Rebalancing_type=="quarterly" :
#        rebalance_type=4
#    elif Rebalancing_type=="Weekly" or Rebalancing_type=="weekly" :
#        rebalance_type=2
#    elif Rebalancing_type=="Daily" or Rebalancing_type=="daily" :
#        rebalance_type=1 
#         
#    rebalancing_count_down=0
#    for i in range(vol_day+day_delay,len(newStrat)):
#        if (i==vol_day+day_delay):
#            month_return=np.array(newStrat.loc[newStrat.index[i-day_delay]]/newStrat.loc[newStrat.index[i-day_return-day_delay]])
#            ticker_index=(np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1]))
#            weight=MD_weight_calculation(i,newStrat,ticker_index,vol_day,decay,day_delay)
#            rebalance_index=i
#            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
#            weights_pd.loc[i-vol_day-day_delay+1]=weight
#        elif (rebalance_type==1 and newStrat.index[i].day!=newStrat.index[i-1].day) or (rebalance_type==2 and newStrat.index[i].week!=newStrat.index[i-1].week) or (rebalance_type==3 and newStrat.index[i].month!=newStrat.index[i-1].month) or (rebalance_type==4 and newStrat.index[i].month!=newStrat.index[i-1].month and newStrat.index[i].month%3==0) :
#            if rebalancing_count_down<=0:
#                month_return=np.array(newStrat.loc[newStrat.index[i-day_delay]]/newStrat.loc[newStrat.index[i-day_return-day_delay]])
#                ticker_index=np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1])
#                rebalancing_count_down=Rebalancing_type2
#            
#            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-0.9) or (sum((newStrat.loc[newStrat.index[i-none_zero_day1]]/newStrat.loc[newStrat.index[i-none_zero_day2]])>1-none_zero_threshold))<0:   
#                weight_pot=np.zeros(len(newStrat.columns))
#            else:
#                weight_pot=MD_weight_calculation(i,newStrat,ticker_index,vol_day,decay,day_delay)
#            if sum(abs(weight_pot-weight)>(threshold))>0:
#                cost=0
#                #cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
#                weight=weight_pot
#                newStrat_price.loc[i-vol_day-day_delay+1]=(1-cost)*newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
#                rebalance_index=i
#                rebalance_time+=1
#            else:
#                newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))               
#            weights_pd.loc[i-vol_day-day_delay+1]=weight
#            rebalancing_count_down-=1
#        else:
#            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-0.9):
#                weight_pot=np.zeros(len(newStrat.columns))
#                cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
#                weight=weight_pot
#                rebalance_index=i
#            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))
#            weights_pd.loc[i-vol_day-day_delay+1]=weight
#
#    newStrat_price.index=newStrat.index[vol_day+day_delay-1:]
#    newStrat_price2=pd.DataFrame(columns=['newStrat'])
#    weights_pd['w']=0
#    newStrat_price2.loc[0]=100.0
#    rebalancing_index=0
#    w=1.0
#    for k in range(1,len(newStrat_price)):
#        if (newStrat_price.loc[newStrat_price.index[k-2]]/newStrat_price.loc[newStrat_price.index[rebalancing_index]])[0]<(1-stop_loss):
#            w=0.0
#        
#        if  (rebalance_type==1 and newStrat_price.index[k].day!=newStrat_price.index[k-1].day) or (rebalance_type==2 and newStrat_price.index[k].week!=newStrat_price.index[k-1].week) or (rebalance_type==3 and newStrat_price.index[k].month!=newStrat_price.index[k-1].month) or (rebalance_type==4 and newStrat_price.index[k].month!=newStrat_price.index[k-1].month and newStrat_price.index[k].month%3==0) :
#            if (newStrat_price.loc[newStrat_price.index[k-2]]/newStrat_price.loc[newStrat_price.index[rebalancing_index]])[0]>(1-stop_loss):
#                w=1.0
#            rebalancing_index=k-1
#        newStrat_price2.loc[k]=newStrat_price2.loc[k-1]*(1+w*(newStrat_price.loc[newStrat_price.index[k]]/newStrat_price.loc[newStrat_price.index[k-1]]-1))
#        weights_pd.loc[k,'w']=w
#    weights_pd.index=newStrat.index[vol_day+day_delay:]
#    Strat=Strategy(newStrat_price.index,newStrat_price2['newStrat'])
#    sharpe_temp=Strat.Sharpe()
#    return (sharpe_temp,weights_pd,Strat,newStrat_price)
        
    
    
def Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec_Individual_momentum_everyX_sharpe(newStrat,Rebalancing_type,MomentumType,Amount_Top,vol_day,day_delay,day_return,threshold,decay,stop_loss,none_zero_day1,none_zero_day2,none_zero_number,none_zero_threshold,Rebalancing_type2,iObservation_period):
    newStrat_price=pd.DataFrame(columns=['newStrat'])
    newStrat_price.loc[0]=100.0
    rebalance_index=vol_day-30
    rebalance_time=0
    weights_pd=pd.DataFrame(columns=["w" + str(i) for i in range(1,len(newStrat.columns)+1)])

    rebalance_type=0
    if Rebalancing_type=="monthly" or Rebalancing_type=="Monthly" :
        rebalance_type=3
    elif Rebalancing_type=="Quarterly" or Rebalancing_type=="quarterly" :
        rebalance_type=4
    elif Rebalancing_type=="Weekly" or Rebalancing_type=="weekly" :
        rebalance_type=2
    elif Rebalancing_type=="Daily" or Rebalancing_type=="daily" :
        rebalance_type=1 
         
    rebalancing_count_down=0
    for i in range(vol_day+day_delay,len(newStrat)):
        if (i==vol_day+day_delay):
            month_return=np.array([])
            for j in range(0,len(newStrat.columns)):
                month_return=np.insert(month_return,j,Sharpe_Calculator(newStrat.loc[newStrat.index[i-day_return-day_delay]:newStrat.index[i-day_delay],newStrat.columns[j]],np.array([1])))
            ticker_index=(np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1]))
            weight=MD_weight_calculation(i,newStrat,ticker_index,vol_day,decay,day_delay)
            rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight
        elif (rebalance_type==1 and newStrat.index[i].day!=newStrat.index[i-1].day) or (rebalance_type==2 and newStrat.index[i].week!=newStrat.index[i-1].week) or (rebalance_type==3 and newStrat.index[i].month!=newStrat.index[i-1].month) or (rebalance_type==4 and newStrat.index[i].month!=newStrat.index[i-1].month and newStrat.index[i].month%3==0) :
            if rebalancing_count_down<=0:
                month_return=np.array([])
                for j in range(0,len(newStrat.columns)):
                    month_return=np.insert(month_return,j,Sharpe_Calculator(newStrat.loc[newStrat.index[i-day_return-day_delay]:newStrat.index[i-day_delay],newStrat.columns[j]],np.array([1])))
                ticker_index=np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1])
                rebalancing_count_down=Rebalancing_type2
                
            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-stop_loss) or (sum((newStrat.loc[newStrat.index[i-none_zero_day1]]/newStrat.loc[newStrat.index[i-none_zero_day2]])>1-none_zero_threshold))<none_zero_number:   
                weight_pot=np.zeros(len(newStrat.columns))
            else:
                weight_pot=MD_weight_calculation(i,newStrat,ticker_index,vol_day,decay,day_delay)
            if sum(abs(weight_pot-weight)>(threshold))>0:
                cost=0
                #cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                newStrat_price.loc[i-vol_day-day_delay+1]=(1-cost)*newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
                rebalance_index=i
                rebalance_time+=1
            else:
                newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))               
            weights_pd.loc[i-vol_day-day_delay+1]=weight
            rebalancing_count_down-=1
        else:
            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-0.9):
                weight_pot=np.zeros(len(newStrat.columns))
                cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight

    newStrat_price.index=newStrat.index[vol_day+day_delay-1:]
    weights_pd.index=newStrat.index[vol_day+day_delay:]
    Strat=Strategy(newStrat_price.index,newStrat_price['newStrat'])
    sharpe_temp=Strat.Sharpe()
    return (sharpe_temp,weights_pd,Strat,newStrat_price)
    
    
def Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec_sharpe_momentum(newStrat,Rebalancing_type,MomentumType,Amount_Top,vol_day,day_delay,day_return,threshold,decay,stop_loss,none_zero_day1,none_zero_day2,none_zero_number,none_zero_threshold):
    newStrat_price=pd.DataFrame(columns=['newStrat'])
    newStrat_price.loc[0]=100.0
    rebalance_index=vol_day-30
    rebalance_time=0
    weights_pd=pd.DataFrame(columns=["w" + str(i) for i in range(1,len(newStrat.columns)+1)])

    rebalance_type=0
    if Rebalancing_type=="monthly" or Rebalancing_type=="Monthly" :
        rebalance_type=3
    elif Rebalancing_type=="Quarterly" or Rebalancing_type=="quarterly" :
        rebalance_type=4
    elif Rebalancing_type=="Weekly" or Rebalancing_type=="weekly" :
        rebalance_type=2
    elif Rebalancing_type=="Daily" or Rebalancing_type=="daily" :
        rebalance_type=1   
    
    for i in range(vol_day+day_delay,len(newStrat)):
        if (i==vol_day+day_delay):
            month_return=np.array([])
            for j in range(0,len(newStrat.columns)):
                month_return=np.insert(month_return,j,Sharpe_Calculator(newStrat.loc[newStrat.index[i-day_return-day_delay]:newStrat.index[i-day_delay],newStrat.columns[j]],np.array([1])))
            ticker_index=(np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1]))
            weight=MD_weight_calculation(i,newStrat,ticker_index,vol_day,decay,day_delay)
            rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight
        elif (rebalance_type==1 and newStrat.index[i].day!=newStrat.index[i-1].day) or (rebalance_type==2 and newStrat.index[i].week!=newStrat.index[i-1].week) or (rebalance_type==3 and newStrat.index[i].month!=newStrat.index[i-1].month) or (rebalance_type==4 and newStrat.index[i].month!=newStrat.index[i-1].month and newStrat.index[i].month%3==0) :
            month_return=np.array([])
            for j in range(0,len(newStrat.columns)):
                month_return=np.insert(month_return,j,Sharpe_Calculator(newStrat.loc[newStrat.index[i-day_return-day_delay]:newStrat.index[i-day_delay],newStrat.columns[j]],np.array([1])))
            ticker_index=np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1])
            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-stop_loss) or (sum((newStrat.loc[newStrat.index[i-none_zero_day1]]/newStrat.loc[newStrat.index[i-none_zero_day2]])>1-none_zero_threshold))<none_zero_number:   
                weight_pot=np.zeros(len(newStrat.columns))
            else:
                weight_pot=MD_weight_calculation(i,newStrat,ticker_index,vol_day,decay,day_delay)
            if sum(abs(weight_pot-weight)>(threshold))>0:
                cost=0
                #cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                newStrat_price.loc[i-vol_day-day_delay+1]=(1-cost)*newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
                rebalance_index=i
                rebalance_time+=1
            else:
                newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))               
            weights_pd.loc[i-vol_day-day_delay+1]=weight
#            print(weight)
        else:
            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-0.9):
                weight_pot=np.zeros(len(newStrat.columns))
                cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight

    newStrat_price.index=newStrat.index[vol_day+day_delay-1:]
    weights_pd.index=newStrat.index[vol_day+day_delay:]
    Strat=Strategy(newStrat_price.index,newStrat_price['newStrat'])
    sharpe_temp=Strat.Sharpe()
    return (sharpe_temp,weights_pd,Strat,newStrat_price)
    
    
    
def Maximum_Diverisification_Strategy_Return_NoneZero_Daily(newStrat,Rebalancing_type,MomentumType,Amount_Top,vol_day,day_delay,day_return,threshold,decay,stop_loss,none_zero_day1,none_zero_day2,none_zero_number,none_zero_threshold):
    newStrat_price=pd.DataFrame(columns=['newStrat'])
    newStrat_price.loc[0]=100.0
    rebalance_index=vol_day-30
    rebalance_time=0
    weights_pd=pd.DataFrame(columns=["w" + str(i) for i in range(1,len(newStrat.columns)+1)])

    rebalance_type=0
    if Rebalancing_type=="monthly" or Rebalancing_type=="Monthly" :
        rebalance_type=3
    elif Rebalancing_type=="Quarterly" or Rebalancing_type=="quarterly" :
        rebalance_type=4
    elif Rebalancing_type=="Weekly" or Rebalancing_type=="weekly" :
        rebalance_type=2
    elif Rebalancing_type=="Daily" or Rebalancing_type=="daily" :
        rebalance_type=1   
    
    for i in range(vol_day+day_delay,len(newStrat)):
        if (i==vol_day+day_delay):
            month_return=np.array(newStrat.loc[newStrat.index[i-day_delay]]/newStrat.loc[newStrat.index[i-day_return-day_delay]])
            ticker_index=(np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[0]))
            weight=MD_weight_calculation_Rules_book(i,newStrat,ticker_index,vol_day,decay,day_delay)
            rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight
        elif (rebalance_type==1 and newStrat.index[i].day!=newStrat.index[i-1].day) or (rebalance_type==2 and newStrat.index[i].week!=newStrat.index[i-1].week) or (rebalance_type==3 and newStrat.index[i].month!=newStrat.index[i-1].month) or (rebalance_type==4 and newStrat.index[i].month!=newStrat.index[i-1].month and newStrat.index[i].month%3==0) :
            month_return=np.array(newStrat.loc[newStrat.index[i-day_delay]]/newStrat.loc[newStrat.index[i-day_return-day_delay]])
            ticker_index=np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[0])
            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-stop_loss) or (sum((newStrat.loc[newStrat.index[i-none_zero_day1]]/newStrat.loc[newStrat.index[i-none_zero_day2]])>1-none_zero_threshold))<none_zero_number:   
                weight_pot=np.zeros(len(newStrat.columns))
            else:
                weight_pot=MD_weight_calculation_Rules_book(i,newStrat,ticker_index,vol_day,decay,day_delay)
            if sum(abs(weight_pot-weight)>(threshold))>0:
                cost=0
                #cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                newStrat_price.loc[i-vol_day-day_delay+1]=(1-cost)*newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
                rebalance_index=i
                rebalance_time+=1
            else:
                newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))               
            weights_pd.loc[i-vol_day-day_delay+1]=weight
        else:
            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-stop_loss):
                weight_pot=np.zeros(len(newStrat.columns))
                cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight

    newStrat_price.index=newStrat.index[vol_day+day_delay-1:]
    weights_pd.index=newStrat.index[vol_day+day_delay:]
    Strat=Strategy(newStrat_price.index,newStrat_price['newStrat'])
    sharpe_temp=Strat.Sharpe()
    return (sharpe_temp,weights_pd,Strat,newStrat_price)
    
def Maximum_Diverisification_Strategy_Return_NoneZero_AGG(newStrat,Rebalancing_type,MomentumType,Amount_Top,vol_day,day_delay,day_return,threshold,decay,stop_loss,none_zero_day1,none_zero_day2,none_zero_number,none_zero_threshold):
#    AGG=newStrat['AGG']
#    newStrat=newStrat[list(set(newStrat1.columns)-set(['AGG']))]    
    newStrat_price=pd.DataFrame(columns=['newStrat'])
    newStrat_price.loc[0]=100.0
    rebalance_index=vol_day-30
    rebalance_time=0
    weights_pd=pd.DataFrame(columns=["w" + str(i) for i in range(1,len(newStrat.columns)+1)])

    rebalance_type=0
    if Rebalancing_type=="monthly" or Rebalancing_type=="Monthly" :
        rebalance_type=3
    elif Rebalancing_type=="Quarterly" or Rebalancing_type=="quarterly" :
        rebalance_type=4
    elif Rebalancing_type=="Weekly" or Rebalancing_type=="weekly" :
        rebalance_type=2
    elif Rebalancing_type=="Daily" or Rebalancing_type=="daily" :
        rebalance_type=1   
    
    for i in range(vol_day+day_delay,len(newStrat)):
        if (i==vol_day+day_delay):
            month_return=np.array(newStrat.loc[newStrat.index[i-day_delay]]/newStrat.loc[newStrat.index[i-day_return-day_delay]])[0:(len(newStrat.columns)-1)]
            ticker_index=np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1])
            weight=MD_weight_calculation(i,newStrat,ticker_index,vol_day,decay,day_delay)
            rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight
        elif (rebalance_type==1 and newStrat.index[i].day!=newStrat.index[i-1].day) or (rebalance_type==2 and newStrat.index[i].week!=newStrat.index[i-1].week) or (rebalance_type==3 and newStrat.index[i].month!=newStrat.index[i-1].month) or (rebalance_type==4 and newStrat.index[i].month!=newStrat.index[i-1].month and newStrat.index[i].month%3==0) :
            month_return=np.array(newStrat.loc[newStrat.index[i-day_delay]]/newStrat.loc[newStrat.index[i-day_return-day_delay]])[0:(len(newStrat.columns)-1)]
            ticker_index=np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1])
            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-stop_loss) :   
                weight_pot=np.zeros(len(newStrat.columns))
            elif (sum((newStrat.loc[newStrat.index[i-none_zero_day1]]/newStrat.loc[newStrat.index[i-none_zero_day2]])[0:(len(newStrat.columns)-1)]>1-none_zero_threshold))<none_zero_number:
                weight_pot=np.zeros(len(newStrat.columns))
                weight_pot[len(newStrat.columns)-1]=1.0
            else:
                weight_pot=MD_weight_calculation(i,newStrat,ticker_index,vol_day,decay,day_delay)
            if sum(abs(weight_pot-weight)>(threshold))>0:
                cost=0
                #cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                newStrat_price.loc[i-vol_day-day_delay+1]=(1-cost)*newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
                rebalance_index=i
                rebalance_time+=1
            else:
                newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))               
            weights_pd.loc[i-vol_day-day_delay+1]=weight
        else:
            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-0.9):
                weight_pot=np.zeros(len(newStrat.columns))
                cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight

    newStrat_price.index=newStrat.index[vol_day+day_delay-1:]
    weights_pd.index=newStrat.index[vol_day+day_delay:]
    Strat=Strategy(newStrat_price.index,newStrat_price['newStrat'])
    sharpe_temp=Strat.Sharpe()
    return (sharpe_temp,weights_pd,Strat,newStrat_price)




def Maximum_Diverisification_Strategy_Return_OnlyPositive(newStrat,Rebalancing_type,MomentumType,Amount_Top,vol_day,day_delay,day_return,threshold,decay,stop_loss,none_zero_day1,none_zero_day2,none_zero_number,none_zero_threshold):
    newStrat_price=pd.DataFrame(columns=['newStrat'])
    newStrat_price.loc[0]=100.0
    rebalance_index=vol_day-30
    rebalance_time=0
    weights_pd=pd.DataFrame(columns=["w" + str(i) for i in range(1,len(newStrat.columns)+1)])

    rebalance_type=0
    if Rebalancing_type=="monthly" or Rebalancing_type=="Monthly" :
        rebalance_type=3
    elif Rebalancing_type=="Quarterly" or Rebalancing_type=="quarterly" :
        rebalance_type=4
    elif Rebalancing_type=="Weekly" or Rebalancing_type=="weekly" :
        rebalance_type=2
    elif Rebalancing_type=="Daily" or Rebalancing_type=="daily" :
        rebalance_type=1   
    
    for i in range(vol_day+day_delay,len(newStrat)):
        if (i==vol_day+day_delay):
            month_return=np.array(newStrat.loc[newStrat.index[i-day_delay]]/newStrat.loc[newStrat.index[i-day_return-day_delay]])
            ticker_index=np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1])
            weight=MD_weight_calculation(i,newStrat,ticker_index,vol_day,decay,day_delay)
            rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight
        elif (rebalance_type==1 and newStrat.index[i].day!=newStrat.index[i-1].day) or (rebalance_type==2 and newStrat.index[i].week!=newStrat.index[i-1].week) or (rebalance_type==3 and newStrat.index[i].month!=newStrat.index[i-1].month) or (rebalance_type==4 and newStrat.index[i].month!=newStrat.index[i-1].month and newStrat.index[i].month%3==0) :
            month_return=np.array(newStrat.loc[newStrat.index[i-none_zero_day1]]/newStrat.loc[newStrat.index[i-none_zero_day2]])
            ticker_index=np.array(range(0,len(month_return)))[month_return>1-none_zero_threshold]
            if len(ticker_index)==len(newStrat.columns):
                ticker_index=np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1])
            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-stop_loss) or len(ticker_index)==0:
                weight_pot=np.zeros(len(newStrat.columns))
            else:
                weight_pot=MD_weight_calculation(i,newStrat,ticker_index,vol_day,decay,day_delay)
            if sum(abs(weight_pot-weight)>(threshold))>0:
                cost=0
                #cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                newStrat_price.loc[i-vol_day-day_delay+1]=(1-cost)*newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
                rebalance_index=i
                rebalance_time+=1
            else:
                newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))               
            weights_pd.loc[i-vol_day-day_delay+1]=weight
        else:
            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-0.9):
                weight_pot=np.zeros(len(newStrat.columns))
                cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight

    newStrat_price.index=newStrat.index[vol_day+day_delay-1:]
    weights_pd.index=newStrat.index[vol_day+day_delay:]
    Strat=Strategy(newStrat_price.index,newStrat_price['newStrat'])
    sharpe_temp=Strat.Sharpe()
    return (sharpe_temp,weights_pd,Strat,newStrat_price)

def Maximum_Diverisification_Strategy_Sharpe_Momentum(newStrat,Rebalancing_type,MomentumType,Amount_Top,vol_day,day_delay,day_return,threshold,decay,stop_loss):
    newStrat_price=pd.DataFrame(columns=['newStrat'])
    newStrat_price.loc[0]=100.0
    rebalance_index=vol_day-30
    rebalance_time=0
    weights_pd=pd.DataFrame(columns=["w" + str(i) for i in range(1,len(newStrat.columns)+1)])

    rebalance_type=0
    if Rebalancing_type=="monthly" or Rebalancing_type=="Monthly" :
        rebalance_type=3
    elif Rebalancing_type=="Quarterly" or Rebalancing_type=="quarterly" :
        rebalance_type=4
    elif Rebalancing_type=="Weekly" or Rebalancing_type=="weekly" :
        rebalance_type=2
    elif Rebalancing_type=="Daily" or Rebalancing_type=="daily" :
        rebalance_type=1   
    
    for i in range(vol_day+day_delay,len(newStrat)):
        if (i==vol_day+day_delay):
            month_return=np.array([])
            for j in range(0,len(newStrat.columns)):
                month_return=np.insert(month_return,j,Sharpe_Calculator(newStrat.loc[newStrat.index[i-day_return-day_delay]:newStrat.index[i-day_delay],newStrat.columns[j]],np.array([1])))
#            month_return=np.array(newStrat.loc[newStrat.index[i-day_delay]]/newStrat.loc[newStrat.index[i-day_return-day_delay]])
            ticker_index=np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[0])
            weight=MD_weight_calculation_Rules_book(i,newStrat,ticker_index,vol_day,decay,day_delay)
            rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight
        elif (rebalance_type==1 and newStrat.index[i].day!=newStrat.index[i-1].day) or (rebalance_type==2 and newStrat.index[i].week!=newStrat.index[i-1].week) or (rebalance_type==3 and newStrat.index[i].month!=newStrat.index[i-1].month) or (rebalance_type==4 and newStrat.index[i].month!=newStrat.index[i-1].month and newStrat.index[i].month%3==0) :
            month_return=np.array([])
            for j in range(0,len(newStrat.columns)):
                month_return=np.insert(month_return,j,Sharpe_Calculator(newStrat.loc[newStrat.index[i-day_return-day_delay]:newStrat.index[i-day_delay],newStrat.columns[j]],np.array([1])))
            ticker_index=np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[0])
            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-stop_loss):
                weight_pot=np.zeros(len(newStrat.columns))
            else:
                weight_pot=MD_weight_calculation_Rules_book(i,newStrat,ticker_index,vol_day,decay,day_delay)
            if sum(abs(weight_pot-weight)>(threshold))>0:
                cost=0
                #cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                newStrat_price.loc[i-vol_day-day_delay+1]=(1-cost)*newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
                rebalance_index=i
                rebalance_time+=1
            else:
                newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))               
            weights_pd.loc[i-vol_day-day_delay+1]=weight
        else:
            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-0.9):
                weight_pot=np.zeros(len(newStrat.columns))
                cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight

    newStrat_price.index=newStrat.index[vol_day+day_delay-1:]
    weights_pd.index=newStrat.index[vol_day+day_delay:]
    Strat=Strategy(newStrat_price.index,newStrat_price['newStrat'])
    sharpe_temp=Strat.Sharpe()
    return (sharpe_temp,weights_pd,Strat,newStrat_price)

def Maximum_Diverisification_Strategy(newStrat,Rebalancing_type,MomentumType,Amount_Top,vol_day,day_delay,day_return,threshold,decay,stop_loss):
    newStrat_price=pd.DataFrame(columns=['newStrat'])
    newStrat_price.loc[0]=100.0
    rebalance_index=vol_day-30
    rebalance_time=0
    weights_pd=pd.DataFrame(columns=["w" + str(i) for i in range(1,len(newStrat.columns)+1)])

    rebalance_type=0
    if Rebalancing_type=="monthly" or Rebalancing_type=="Monthly" :
        rebalance_type=3
    elif Rebalancing_type=="Quarterly" or Rebalancing_type=="quarterly" :
        rebalance_type=4
    elif Rebalancing_type=="Weekly" or Rebalancing_type=="weekly" :
        rebalance_type=2
    elif Rebalancing_type=="Daily" or Rebalancing_type=="daily" :
        rebalance_type=1   
    
    for i in range(vol_day+day_delay,len(newStrat)):
        if (i==vol_day+day_delay):
            month_return=np.array(newStrat.loc[newStrat.index[i-day_delay]]/newStrat.loc[newStrat.index[i-day_return-day_delay]])
            ticker_index=np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1])
            weight=c(i,newStrat,ticker_index,vol_day,decay,day_delay)
            rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight
        elif (rebalance_type==1 and newStrat.index[i].day!=newStrat.index[i-1].day) or (rebalance_type==2 and newStrat.index[i].week!=newStrat.index[i-1].week) or (rebalance_type==3 and newStrat.index[i].month!=newStrat.index[i-1].month) or (rebalance_type==4 and newStrat.index[i].month!=newStrat.index[i-1].month and newStrat.index[i].month%3==0) :
            month_return=np.array(newStrat.loc[newStrat.index[i-day_delay]]/newStrat.loc[newStrat.index[i-day_return-day_delay]])
            ticker_index=np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1])
            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-stop_loss):
                weight_pot=np.zeros(len(newStrat.columns))
            else:
                weight_pot=MD_weight_calculation(i,newStrat,ticker_index,vol_day,decay,day_delay)
            if sum(abs(weight_pot-weight)>(threshold))>0:
                cost=0
                #cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                newStrat_price.loc[i-vol_day-day_delay+1]=(1-cost)*newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
                rebalance_index=i
                rebalance_time+=1
            else:
                newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))               
            weights_pd.loc[i-vol_day-day_delay+1]=weight
        else:
            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-0.9):
                weight_pot=np.zeros(len(newStrat.columns))
                #cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight

    newStrat_price.index=newStrat.index[vol_day+day_delay-1:]
    weights_pd.index=newStrat.index[vol_day+day_delay:]
    Strat=Strategy(newStrat_price.index,newStrat_price['newStrat'])
    sharpe_temp=Strat.Sharpe()
    return (sharpe_temp,weights_pd,Strat,newStrat_price)


def CVaR_Strategy(newStrat,Rebalancing_type,MomentumType,Amount_Top,vol_day,day_delay,day_return,threshold,stop_loss,target,none_zero_day1,none_zero_day2,none_zero_number,none_zero_threshold):
    newStrat_price=pd.DataFrame(columns=['newStrat'])
    newStrat_price.loc[0]=100.0
    rebalance_index=vol_day-30
    rebalance_time=0
    weights_pd=pd.DataFrame(columns=["w" + str(i) for i in range(1,len(newStrat.columns)+1)])

    rebalance_type=0
    if Rebalancing_type=="monthly" or Rebalancing_type=="Monthly" :
        rebalance_type=3
    elif Rebalancing_type=="Quarterly" or Rebalancing_type=="quarterly" :
        rebalance_type=4
    elif Rebalancing_type=="Weekly" or Rebalancing_type=="weekly" :
        rebalance_type=2
    elif Rebalancing_type=="Daily" or Rebalancing_type=="daily" :
        rebalance_type=1   
    
    for i in range(vol_day+day_delay,len(newStrat)):
        if (i==vol_day+day_delay):
            month_return=np.array(newStrat.loc[newStrat.index[i-2]]/newStrat.loc[newStrat.index[i-day_return]])
            ticker_index=np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1])
            weight=CVaR_weight(i,newStrat,ticker_index,vol_day,target)
            rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight
        elif (rebalance_type==1 and newStrat.index[i].day!=newStrat.index[i-1].day) or (rebalance_type==2 and newStrat.index[i].week!=newStrat.index[i-1].week) or (rebalance_type==3 and newStrat.index[i].month!=newStrat.index[i-1].month) or (rebalance_type==4 and newStrat.index[i].month!=newStrat.index[i-1].month and newStrat.index[i].month%3==0) :
            month_return=np.array(newStrat.loc[newStrat.index[i-2]]/newStrat.loc[newStrat.index[i-day_return]])
            ticker_index=np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1])
            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-stop_loss) or (sum((newStrat.loc[newStrat.index[i-none_zero_day1]]/newStrat.loc[newStrat.index[i-none_zero_day2]])>1-none_zero_threshold))<none_zero_number:
                weight_pot=np.zeros(len(newStrat.columns))
            else:
                weight_pot=CVaR_weight(i,newStrat,ticker_index,vol_day,target)
            if sum(abs(weight_pot-weight)>(threshold))>0 and sum(weight_pot<0)<1:
                cost=0
                #cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                newStrat_price.loc[i-vol_day-day_delay+1]=(1-cost)*newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
                rebalance_index=i
                rebalance_time+=1
            else:
                newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))               
            weights_pd.loc[i-vol_day-day_delay+1]=weight
        else:
            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-stop_loss):
                weight_pot=np.zeros(len(newStrat.columns))
                #cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight

    newStrat_price.index=newStrat.index[vol_day+day_delay-1:]
    weights_pd.index=newStrat.index[vol_day+day_delay:]
    Strat=Strategy(newStrat_price.index,newStrat_price['newStrat'])
    sharpe_temp=Strat.Sharpe()
    return (sharpe_temp,weights_pd,Strat,newStrat_price)


def Sharpe_Strategy(newStrat,Rebalancing_type,MomentumType,Amount_Top,vol_day,day_delay,day_return,threshold,stop_loss,LB):
    newStrat_price=pd.DataFrame(columns=['newStrat'])
    newStrat_price.loc[0]=100.0
    rebalance_index=vol_day-30
    rebalance_time=0
    weights_pd=pd.DataFrame(columns=["w" + str(i) for i in range(1,len(newStrat.columns)+1)])

    rebalance_type=0
    if Rebalancing_type=="monthly" or Rebalancing_type=="Monthly" :
        rebalance_type=3
    elif Rebalancing_type=="Quarterly" or Rebalancing_type=="quarterly" :
        rebalance_type=4
    elif Rebalancing_type=="Weekly" or Rebalancing_type=="weekly" :
        rebalance_type=2
    elif Rebalancing_type=="Daily" or Rebalancing_type=="daily" :
        rebalance_type=1   
    
    for i in range(vol_day+day_delay,len(newStrat)):
        if (i==vol_day+day_delay):
            month_return=np.array(newStrat.loc[newStrat.index[i-2]]/newStrat.loc[newStrat.index[i-day_return]])
            ticker_index=np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1])
            weight=Sharpe_weight_Calculation(i,newStrat,ticker_index,vol_day,LB)
            rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight
        elif (rebalance_type==1 and newStrat.index[i].day!=newStrat.index[i-1].day) or (rebalance_type==2 and newStrat.index[i].week!=newStrat.index[i-1].week) or (rebalance_type==3 and newStrat.index[i].month!=newStrat.index[i-1].month) or (rebalance_type==4 and newStrat.index[i].month!=newStrat.index[i-1].month and newStrat.index[i].month%3==0) :
            month_return=np.array(newStrat.loc[newStrat.index[i-2]]/newStrat.loc[newStrat.index[i-day_return]])
            ticker_index=np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1])
            if (newStrat_price.loc[i-vol_day-day_delay]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-stop_loss):
                weight_pot=np.zeros(len(newStrat.columns))
            else:
                weight_pot=Sharpe_weight_Calculation(i,newStrat,ticker_index,vol_day,LB)
                print(weight_pot)
            if sum(abs(weight_pot-weight)>(threshold))>0:
                cost=0
                #cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                newStrat_price.loc[i-vol_day-day_delay+1]=(1-cost)*newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
                rebalance_index=i
                rebalance_time+=1
            else:
                newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))               
            weights_pd.loc[i-vol_day-day_delay+1]=weight
        else:
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight

    newStrat_price.index=newStrat.index[vol_day+day_delay-1:]
    weights_pd.index=newStrat.index[vol_day+day_delay:]
    Strat=Strategy(newStrat_price.index,newStrat_price['newStrat'])
    sharpe_temp=Strat.Sharpe()
    return (sharpe_temp,weights_pd,Strat,newStrat_price)


def Vol_Over_Sharpe_Strategy(newStrat,Rebalancing_type,MomentumType,Amount_Top,vol_day,day_delay,day_return,threshold,stop_loss,LB,decay):
    newStrat_price=pd.DataFrame(columns=['newStrat'])
    newStrat_price.loc[0]=100.0
    rebalance_index=vol_day-30
    rebalance_time=0
    weights_pd=pd.DataFrame(columns=["w" + str(i) for i in range(1,len(newStrat.columns)+1)])

    rebalance_type=0
    if Rebalancing_type=="monthly" or Rebalancing_type=="Monthly" :
        rebalance_type=3
    elif Rebalancing_type=="Quarterly" or Rebalancing_type=="quarterly" :
        rebalance_type=4
    elif Rebalancing_type=="Weekly" or Rebalancing_type=="weekly" :
        rebalance_type=2
    elif Rebalancing_type=="Daily" or Rebalancing_type=="daily" :
        rebalance_type=1   
    
    for i in range(vol_day+day_delay,len(newStrat)):
        if (i==vol_day+day_delay):
            month_return=np.array(newStrat.loc[newStrat.index[i-2]]/newStrat.loc[newStrat.index[i-day_return]])
            ticker_index=np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1])
            weight=Vol_Over_Sharpe_weight_Calculation(i,newStrat,ticker_index,vol_day,LB,decay)
            rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight
        elif (rebalance_type==1 and newStrat.index[i].day!=newStrat.index[i-1].day) or (rebalance_type==2 and newStrat.index[i].week!=newStrat.index[i-1].week) or (rebalance_type==3 and newStrat.index[i].month!=newStrat.index[i-1].month) or (rebalance_type==4 and newStrat.index[i].month!=newStrat.index[i-1].month and newStrat.index[i].month%3==0) :
            month_return=np.array(newStrat.loc[newStrat.index[i-2]]/newStrat.loc[newStrat.index[i-day_return]])
            ticker_index=np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1])
            if (newStrat_price.loc[i-vol_day-day_delay]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-stop_loss):
                weight_pot=np.zeros(len(newStrat.columns))
            else:
                weight_pot=Vol_Over_Sharpe_weight_Calculation(i,newStrat,ticker_index,vol_day,LB,decay)
#                print weight_pot
            if sum(abs(weight_pot-weight)>(threshold))>0 and sum(weight_pot<0)<1:
                cost=0
                #cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                newStrat_price.loc[i-vol_day-day_delay+1]=(1-cost)*newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
                rebalance_index=i
                rebalance_time+=1
            else:
                newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))               
            weights_pd.loc[i-vol_day-day_delay+1]=weight
        else:
            if (newStrat_price.loc[i-vol_day-day_delay]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-stop_loss):
                weight=np.zeros(len(newStrat.columns))
                rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight

    newStrat_price.index=newStrat.index[vol_day+day_delay-1:]
    weights_pd.index=newStrat.index[vol_day+day_delay:]
    Strat=Strategy(newStrat_price.index,newStrat_price['newStrat'])
    sharpe_temp=Strat.Sharpe()
    return (sharpe_temp,weights_pd,Strat,newStrat_price)


def Vol_Over_Sharpe(price,weight,decay_var):
    cov_matrix=Covariance_matrix2(price.index[-1],price,len(price)-2,decay_var)
    vol=(np.transpose(weight)).dot(np.reshape(np.diagonal(cov_matrix)**0.5,(len(weight),1)))
    sharpe=Sharpe_Calculator(price,np.transpose(weight))
#    print -sharpe/vol
    return -sharpe/vol

def Vol_Over_Sharpe_Optimization(newStrat,LB,decay_var):
    cons=({'type': 'eq', 'fun': lambda x:  1 - sum(x)})
   
    #equality constraint 
    bds=[]
    for i in range(0,len(newStrat.columns)):
        bds.append((0.00001,0.9999999))

    #set initial value
    initial_value= np.ones((len(newStrat.columns),1),dtype=np.float)*1.0/(len(newStrat.columns)+1)
    result =scipy.optimize.minimize(lambda x:Vol_Over_Sharpe(newStrat,x,decay_var)/100.0,x0=initial_value,constraints=cons,bounds=bds,options={'disp': True }).x
    #,options={'disp': True }
    optimal_weights=[]
    for i in range(0,len(result)):
        optimal_weights.append(float(result[i]))
#    (price,weight,decay_var)=(newStrat,optimal_weights,decay_var)
#    cov_matrix=Covariance_matrix2(price.index[-1],price,len(price)-2,decay_var)
#    vol=(np.transpose(weight)).dot(np.reshape(np.diagonal(cov_matrix)**0.5,(len(weight),1)))
#    sharpe=Sharpe_Calculator(price,np.transpose(weight))
    print(optimal_weights)
    return optimal_weights
    
def Vol_Over_Sharpe_weight_Calculation(i,newStrat,ticker_index,period_day,LB,decay_var):
    weight=np.zeros(len(newStrat.columns))
    ticker_list=newStrat.columns[ticker_index]      
    underlying_price=pd.DataFrame()
    for j in ticker_list:
        underlying_price[ticker_list]=newStrat[ticker_list]
    underlying_price.index=newStrat.index
    weight[ticker_index]=Vol_Over_Sharpe_Optimization(underlying_price.loc[underlying_price.index[range(i-period_day,i)]],LB,decay_var)    
    return weight


def CVaR_weight(i,newStrat,ticker_index,period_day,target):
    weight=np.zeros(len(newStrat.columns))
    ticker_list=newStrat.columns[ticker_index]      
    underlying_price=pd.DataFrame()
    for j in ticker_list:
        underlying_price[ticker_list]=newStrat[ticker_list]
    underlying_price.index=newStrat.index
    ret=underlying_price.loc[underlying_price.index[range(i-period_day,i)]].pct_change()
    ret.dropna()
    weight[ticker_index]=CVaROptimization(ret, target/250)
    return weight
    
def return_calculator(price,weight):
    newStrat=pd.DataFrame(columns=['newStrat'])
    newStrat.loc[0]=100.0
    for i in range(1,len(price.index)):
        newStrat.loc[i]=newStrat.loc[0]*weight.dot(price.loc[price.index[i]]/price.loc[price.index[0]])
    newStrat.index=price.index
    return newStrat
    
def Sharpe_Calculator(price,weight):
#    print weight
    newStrat=return_calculator(price,weight)
    dfin=newStrat.index[len(newStrat)-1]
    ddebut= newStrat.index[0]
    days = (dfin-ddebut).days
    Irr= (newStrat.loc[newStrat.index[len(newStrat)-1],'newStrat']/newStrat.loc[newStrat.index[0],'newStrat'])**(365.0/days)-1        
    vol=np.log(newStrat['newStrat'].pct_change()+1).std()*np.sqrt(252)
    return Irr/(vol)

def SharpeOptimization(newStrat,LB):
    cons=({'type': 'eq', 'fun': lambda x:  1 - sum(x)})
   
    #equality constraint 
    bds=[]
    for i in range(0,len(newStrat.columns)):
        bds.append((LB,None))
        
    #set initial value
    initial_value= np.ones((len(newStrat.columns),1),dtype=np.float)*1.0/len(newStrat.columns)
    
    result =scipy.optimize.minimize(lambda x:-Sharpe_Calculator(newStrat,x), x0=initial_value,constraints=cons,bounds=bds).x
    optimal_weights=[]
    for i in range(0,len(result)):
        optimal_weights.append(float(result[i]))
    return optimal_weights

def Sharpe_weight_Calculation(i,newStrat,ticker_index,period_day,LB):
    weight=np.zeros(len(newStrat.columns))
    ticker_list=newStrat.columns[ticker_index]      
    underlying_price=pd.DataFrame()
    for j in ticker_list:
        underlying_price[ticker_list]=newStrat[ticker_list]
    underlying_price.index=newStrat.index
    weight[ticker_index]=SharpeOptimization(underlying_price.loc[underlying_price.index[range(i-period_day,i)]],LB)    
    return weight

def MD_weight_calculation(i,newStrat,ticker_index,vol_day,decay,day_delay):
    weight=np.zeros(len(newStrat.columns))
    ticker_list=newStrat.columns[ticker_index]    
    underlying_price=pd.DataFrame()
    for j in ticker_list:
        underlying_price[j]=newStrat[j]
    underlying_price.index=newStrat.index
    cov_matrix=Covariance_matrix2(underlying_price.index[i-day_delay],underlying_price,vol_day,decay)
    weight[ticker_index]=np.array(Min_Def_Ratio(cov_matrix))
    return weight
    
def MD_weight_calculation_momentum(i,newStrat,ticker_index,vol_day,decay,day_delay,num):
    weight=np.zeros(len(newStrat.columns))
    ticker_list=newStrat.columns[ticker_index]    
    underlying_price=pd.DataFrame()
    for j in ticker_list:
        underlying_price[j]=newStrat[j]
    underlying_price.index=newStrat.index
    cov_matrix=Covariance_matrix2(underlying_price.index[i-day_delay],underlying_price,vol_day,decay)
    weight[ticker_index]=np.array(Min_Def_Ratio(cov_matrix))
    num=np.array(num)
    num=np.insert(num,0,0)
    top_list=[]
    for k in range(0,len(num)-1):
        index=np.argmax(weight[num[k]:num[k+1]])+num[k]
        top_list.append(index)
        weight[index]=np.sum(weight[num[k]:num[k+1]])
    low_list=list(set(range(0,len(weight)))-set(top_list))
    weight[low_list]=0.0
    return weight
    
def Performance_MD_weight_calculation(i,newStrat,ticker_index,vol_day,decay,day_delay):
    weight=np.zeros(len(newStrat.columns))
    ticker_list=newStrat.columns[ticker_index]    
    underlying_price=pd.DataFrame()
    for j in ticker_list:
        underlying_price[j]=newStrat[j]
    underlying_price.index=newStrat.index
    cov_matrix=Covariance_matrix2(underlying_price.index[i-day_delay],underlying_price,vol_day,decay)
    weight[ticker_index]=np.array(Perf_Min_Def_Ratio(cov_matrix,underlying_price.loc[underlying_price.index[i-day_delay]]/underlying_price.loc[underlying_price.index[i-day_delay-vol_day]]))
    return weight

def Performance_MD_weight_calculation_ver2(i,newStrat,ticker_index,vol_day,decay,day_delay):
    weight=np.zeros(len(newStrat.columns))
    ticker_list=newStrat.columns[ticker_index]    
    underlying_price=pd.DataFrame()
    for j in ticker_list:
        underlying_price[j]=newStrat[j]
    underlying_price.index=newStrat.index
    cov_matrix=Covariance_matrix2(underlying_price.index[i-day_delay],underlying_price,vol_day,decay)
    weight[ticker_index]=np.array(Perf_Min_Def_Ratio(cov_matrix,underlying_price.loc[underlying_price.index[i-day_delay]]/underlying_price.loc[underlying_price.index[i-day_delay-vol_day]]-1))
    return weight 
    
def Perf_Min_Def_Ratio(Covariance_matrix,ret):
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

    result =scipy.optimize.minimize(lambda x:-np.dot(ret,x)*np.dot(np.transpose(x),(np.diagonal(Covariance_matrix)**0.5))/(np.dot(np.transpose(x),np.dot(Covariance_matrix,x))**0.5), x0=initial_value,constraints=cons,bounds=bds).x
    optimal_weights=[] 
    for i in range(0,len(result)):
        optimal_weights.append(float(result[i]))
    return optimal_weights
    
    
def MD_weight_calculation_Rules_book(i,newStrat,ticker_index,vol_day,decay,day_delay):
    weight=np.zeros(len(newStrat.columns))
    cov_matrix=Covariance_matrix2(newStrat.index[i-day_delay],newStrat,vol_day,decay)
    weight=np.array(Min_Def_Ratio_Rules_book(cov_matrix,ticker_index))
    return weight


    
def Min_Def_Ratio_Rules_book(Covariance_matrix,ticker_index):
    #inequalitys constraints
    cons=({'type': 'eq', 'fun': lambda x:  1 - sum(x)},{'type': 'eq', 'fun': lambda x:  sum(x[ticker_index]**2)})
   
    #equality constraint 
    bds=[]
    for i in range(0,len(Covariance_matrix)):
        if i not in ticker_index:
            bds.append((0.000000000000001,None))
        else:
            bds.append((-0.0000000000001,0.00000000001))

    #set initial value
    initial_value= np.zeros((len(Covariance_matrix),1),dtype=np.float)
    for i in range(0,len(Covariance_matrix)):
        initial_value[i,0]=1.0/len(Covariance_matrix)
    initial_value[ticker_index]=0.0
    result =scipy.optimize.minimize(lambda x:-np.dot(np.transpose(x),(np.diagonal(Covariance_matrix)**0.5))/(np.dot(np.transpose(x),np.dot(Covariance_matrix,x))**0.5), x0=initial_value,constraints=cons,bounds=bds).x
    optimal_weights=[]
    for i in range(0,len(result)):
        optimal_weights.append(float(result[i]))
    return optimal_weights
    
def data_reading(path):
    newStrat1=pd.read_csv(path)
    newStrat1.index=newStrat1['Dates'].map(lambda x: datetime.datetime.strptime(x,"%m/%d/%Y"))
    del newStrat1['Dates']
    return newStrat1

def excess_return_data(path):
    newStrat1=data_reading(path)
    for i in newStrat1.columns:  
        pd_etf=pd.DataFrame({i:newStrat1[i]})
        newStrat1[i]=beta_adjusted(pd_etf)
    return newStrat1
        
def stardard_strategy_test(newStrat1,path):    
#    for i in [7,20,120,250]:
#        Horizon=i        
#        name='Weekly'
#        m=Sharpe_Momentum_execute(newStrat1,newStrat1.columns,name,Horizon)
#        Strat=Strategy(m.index,m['Index'])
#        Strat.Save(path+"\Sharpe_Momentum"+name+str(Horizon)+".csv")
##        
#        name='Monthly'
#        m=Sharpe_Momentum_execute(newStrat1,newStrat1.columns,name,Horizon)
#        Strat=Strategy(m.index,m['Index'])
#        Strat.Save(path+"\Sharpe_Momentum"+name+str(Horizon)+".csv")
#        
#        name='Quarterly'
#        m=Sharpe_Momentum_execute(newStrat1,newStrat1.columns,name,Horizon)
#        Strat=Strategy(m.index,m['Index'])
#        Strat.Save(path+"\Sharpe_Momentum"+name+str(Horizon)+".csv")
        
    for i in [7,20,120,250]:
        Horizon=i
        name='Weekly'
        m=Momentum_execute(newStrat1,newStrat1.columns,name,Horizon)
        Strat=Strategy(m.index,m['Index'])
        Strat.Save(path+"\Momentum"+name+str(Horizon)+".csv")
        
        name='Monthly'
        m=Momentum_execute(newStrat1,newStrat1.columns,name,Horizon)
        Strat=Strategy(m.index,m['Index'])
        Strat.Save(path+"\Momentum"+name+str(Horizon)+".csv")
        
        name='Quarterly'
        m=Momentum_execute(newStrat1,newStrat1.columns,name,Horizon)
        Strat=Strategy(m.index,m['Index'])
        Strat.Save(path+"\Momentum"+name+str(Horizon)+".csv")
        
        name='Daily'
        m=Momentum_execute(newStrat1,newStrat1.columns,name,Horizon)
        Strat=Strategy(m.index,m['Index'])
        Strat.Save(path+"\Momentum"+name+str(Horizon)+".csv")

#        name='Daily'
#        m=Sharpe_Momentum_execute(newStrat1,newStrat1.columns,name,Horizon)
#        Strat=Strategy(m.index,m['Index'])
#        Strat.Save(path+"\Sharpe_Momentum"+name+str(Horizon)+".csv")

    EW=Equal_Exposure_daily(newStrat1,newStrat1.columns)
    Strat=Strategy(EW.index,EW['Index'])  
    Strat.Save(path+"\EW_daily.csv")
    
    EW=Equal_Exposure_weekly(newStrat1,newStrat1.columns)
    Strat=Strategy(EW.index,EW['Index'])  
    Strat.Save(path+"\EW_weekly.csv")
    
    EW=Equal_Exposure_monthly(newStrat1,newStrat1.columns)
    Strat=Strategy(EW.index,EW['Index'])  
    Strat.Save(path+"\EW_month.csv")
    
    EW=Equal_Exposure_quarterly(newStrat1,newStrat1.columns)
    Strat=Strategy(EW.index,EW['Index'])  
    Strat.Save(path+"\EW_quarterly.csv")
    
    for i in [7,20,120,250]:
        Horizon=i
        name='Weekly'
        m=Rainbow_execute(newStrat1,newStrat1.columns,name,Horizon,[0.75,0.25])
        Strat=Strategy(m.index,m['Index'])
        Strat.Save(path+"\Rainbow7525"+name+str(Horizon)+".csv")
        
        name='Monthly'
        m=Rainbow_execute(newStrat1,newStrat1.columns,name,Horizon,[0.75,0.25])
        Strat=Strategy(m.index,m['Index'])
        Strat.Save(path+"\Rainbow7525"+name+str(Horizon)+".csv")
        
        name='Quarterly'
        m=Rainbow_execute(newStrat1,newStrat1.columns,name,Horizon,[0.75,0.25])
        Strat=Strategy(m.index,m['Index'])
        Strat.Save(path+"\Rainbow7525"+name+str(Horizon)+".csv")
        
        name='Daily'
        m=Rainbow_execute(newStrat1,newStrat1.columns,name,Horizon,[0.75,0.25])
        Strat=Strategy(m.index,m['Index'])
        Strat.Save(path+"\Rainbow7525"+name+str(Horizon)+".csv")

    for i in [7,20,120,250]:
        Horizon=i
        name='Weekly'
        m=Rainbow_execute(newStrat1,newStrat1.columns,name,Horizon,[0.50,0.30,0.20])
        Strat=Strategy(m.index,m['Index'])
        Strat.Save(path+"\Rainbow503020"+name+str(Horizon)+".csv")
        
        name='Monthly'
        m=Rainbow_execute(newStrat1,newStrat1.columns,name,Horizon,[0.50,0.30,0.20])
        Strat=Strategy(m.index,m['Index'])
        Strat.Save(path+"\Rainbow503020"+name+str(Horizon)+".csv")
        
        name='Quarterly'
        m=Rainbow_execute(newStrat1,newStrat1.columns,name,Horizon,[0.50,0.30,0.20])
        Strat=Strategy(m.index,m['Index'])
        Strat.Save(path+"\Rainbow503020"+name+str(Horizon)+".csv")
        
        name='Daily'
        m=Rainbow_execute(newStrat1,newStrat1.columns,name,Horizon,[0.50,0.30,0.20])
        Strat=Strategy(m.index,m['Index'])
        Strat.Save(path+"\Rainbow503020"+name+str(Horizon)+".csv")
        
    for i in [7,20,120,250]:
        Horizon=i
        name='Weekly'
        (sharpe,weights_pd,Strat,newStrat_price)=ERC_Execute_Return_NoneZero(newStrat1,name,"low",0,Horizon,2,Horizon,0.1,0.95,0.99,2,5,2,0.99)
        Strat.Save(path+"\ERC"+name+str(Horizon)+".csv")
        
        name='Monthly'
        (sharpe,weights_pd,Strat,newStrat_price)=ERC_Execute_Return_NoneZero(newStrat1,name,"low",0,Horizon,2,Horizon,0.0,0.95,0.99,2,5,2,0.99)
        Strat.Save(path+"\ERC"+name+str(Horizon)+".csv")
        
        name='Quarterly'
        (sharpe,weights_pd,Strat,newStrat_price)=ERC_Execute_Return_NoneZero(newStrat1,name,"low",0,Horizon,2,Horizon,0.1,0.95,0.99,2,5,2,0.99)
        Strat.Save(path+"\ERC"+name+str(Horizon)+".csv")
        
        name='Daily'
        (sharpe,weights_pd,Strat,newStrat_price)=ERC_Execute_Return_NoneZero(newStrat1,name,"low",0,Horizon,2,Horizon,0.1,0.95,0.99,2,5,2,0.99)
        Strat.Save(path+"\ERC"+name+str(Horizon)+".csv")    
        

        
def Parameter_Optimization(Function_name,standard,dictionary,variable_number):
#    (sharpe_standard,weights_pd,Strat,newStrat_price)=Function_name(*standard)
    for i in range(0,variable_number):
        (sharpe_standard,weights_pd,Strat,newStrat_price)=Function_name(*standard)
        mymax=sharpe_standard
        mymax_id=standard[i]
        for j in dictionary[i]:
            standard_list=list(standard)
            standard_list[i]=j
            standard1=tuple(standard_list)
            (sharpe,weights_pd,Strat,newStrat_price)=Function_name(*standard1)
            print(sharpe)
            print(standard1[1:16])
            if sharpe>mymax:
                mymax_id=j
        standard_list=list(standard)
        standard_list[i]=mymax_id
        standard=tuple(standard_list)
    return standard
        
        



def Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec_Individual_momentum_everyX__stoploss_everyday_underindex(newStrat,Rebalancing_type,MomentumType,Amount_Top,vol_day,day_delay,day_return,threshold,decay,stop_loss,none_zero_day1,none_zero_day2,none_zero_number,none_zero_threshold,Rebalancing_type2,iObservation_period):
    newStrat_price=pd.DataFrame(columns=['newStrat'])
    newStrat_price.loc[0]=100.0
    rebalance_index=vol_day-30
    rebalance_time=0
    weights_pd=pd.DataFrame(columns=["w" + str(i) for i in range(1,len(newStrat.columns)+1)])

    rebalance_type=0
    if Rebalancing_type=="monthly" or Rebalancing_type=="Monthly" :
        rebalance_type=3
    elif Rebalancing_type=="Quarterly" or Rebalancing_type=="quarterly" :
        rebalance_type=4
    elif Rebalancing_type=="Weekly" or Rebalancing_type=="weekly" :
        rebalance_type=2
    elif Rebalancing_type=="Daily" or Rebalancing_type=="daily" :
        rebalance_type=1 
#    inter_var=pd.DataFrame(columns=["ret" + str(i) for i in range(1,len(newStrat.columns)+1)])

    rebalancing_count_down=0
    for i in range(vol_day+day_delay,len(newStrat)):
        if (i==vol_day+day_delay):
            month_return=np.array(newStrat.loc[newStrat.index[i-day_delay]]/newStrat.loc[newStrat.index[i-day_return-day_delay]])
            ticker_index=(np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1]))
            weight=MD_weight_calculation(i,newStrat,ticker_index,vol_day,decay,day_delay)
            rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight
        elif (rebalance_type==1 and newStrat.index[i].day!=newStrat.index[i-1].day) or (rebalance_type==2 and newStrat.index[i].week!=newStrat.index[i-1].week) or (rebalance_type==3 and newStrat.index[i].month!=newStrat.index[i-1].month) or (rebalance_type==4 and newStrat.index[i].month!=newStrat.index[i-1].month and newStrat.index[i].month%3==0) :
            if rebalancing_count_down<=0:
                month_return=np.array(newStrat.loc[newStrat.index[i-day_delay]]/newStrat.loc[newStrat.index[i-day_return-day_delay]])
                ticker_index=np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1])
                rebalancing_count_down=Rebalancing_type2
#                inter_var.loc[i-vol_day-day_delay+1]=np.array(newStrat.loc[newStrat.index[i-none_zero_day1]]/newStrat.loc[newStrat.index[i-none_zero_day2]])
            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-0.9) or (sum((newStrat.loc[newStrat.index[i-none_zero_day1]]/newStrat.loc[newStrat.index[i-none_zero_day2]])>1-none_zero_threshold))<1:   
                weight_pot=np.zeros(len(newStrat.columns))
            else:
                weight_pot=MD_weight_calculation(i,newStrat,ticker_index,vol_day,decay,day_delay)
            if sum(abs(weight_pot-weight)>(threshold))>0:
                cost=0
                #cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                newStrat_price.loc[i-vol_day-day_delay+1]=(1-cost)*newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
                rebalance_index=i
                rebalance_time+=1
            else:
                newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))               
            weights_pd.loc[i-vol_day-day_delay+1]=weight
            rebalancing_count_down-=1
        else:
            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-0.9):
                weight_pot=np.zeros(len(newStrat.columns))
                cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight

    newStrat_price.index=newStrat.index[vol_day+day_delay-1:]
    newStrat_price2=pd.DataFrame(columns=['newStrat'])
#    weights_pd['w']=0
    newStrat_price2.loc[0]=100.0
    rebalancing_index=0
    w=1.0
    for k in range(1,len(newStrat_price)):
#        weights_pd.loc[k,'w']=(newStrat_price.loc[newStrat_price.index[k-2]]/newStrat_price.loc[newStrat_price.index[rebalancing_index]])[0]
        if (newStrat_price.loc[newStrat_price.index[k-2]]/newStrat_price.loc[newStrat_price.index[rebalancing_index]])[0]<(1-stop_loss):
            w=0.0
#            rebalancing_index=k-1
        if  (rebalance_type==1 and newStrat_price.index[k].day!=newStrat_price.index[k-1].day) or (rebalance_type==2 and newStrat_price.index[k].week!=newStrat_price.index[k-1].week) or (rebalance_type==3 and newStrat_price.index[k].month!=newStrat_price.index[k-1].month) or (rebalance_type==4 and newStrat_price.index[k].month!=newStrat_price.index[k-1].month and newStrat_price.index[k].month%3==0) :
            if (newStrat_price.loc[newStrat_price.index[k-2]]/newStrat_price.loc[newStrat_price.index[rebalancing_index]])[0]>=(1-stop_loss):
                w=1.0
            rebalancing_index=k-1
        newStrat_price2.loc[k]=newStrat_price2.loc[k-1]*(1+w*(newStrat_price.loc[newStrat_price.index[k]]/newStrat_price.loc[newStrat_price.index[k-1]]-1))
        
    weights_pd.index=newStrat.index[vol_day+day_delay:]
    Strat=Strategy(newStrat_price.index,newStrat_price2['newStrat'])
    sharpe_temp=Strat.Sharpe()
    return (sharpe_temp,weights_pd,Strat,newStrat_price2)
    
    

def Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec_Individual_momentum_everyX__stoploss_everyday(newStrat,Rebalancing_type,MomentumType,Amount_Top,vol_day,day_delay,day_return,threshold,decay,stop_loss,none_zero_day1,none_zero_day2,none_zero_number,none_zero_threshold,Rebalancing_type2,iObservation_period):
    newStrat_price=pd.DataFrame(columns=['newStrat'])
    newStrat_price.loc[0]=100.0
    rebalance_index=vol_day-30
    rebalance_time=0
    weights_pd=pd.DataFrame(columns=["w" + str(i) for i in range(1,len(newStrat.columns)+1)])

    rebalance_type=0
    if Rebalancing_type=="monthly" or Rebalancing_type=="Monthly" :
        rebalance_type=3
    elif Rebalancing_type=="Quarterly" or Rebalancing_type=="quarterly" :
        rebalance_type=4
    elif Rebalancing_type=="Weekly" or Rebalancing_type=="weekly" :
        rebalance_type=2
    elif Rebalancing_type=="Daily" or Rebalancing_type=="daily" :
        rebalance_type=1 
#    inter_var=pd.DataFrame(columns=["ret" + str(i) for i in range(1,len(newStrat.columns)+1)])
#    return_check=pd.DataFrame(columns=['ret','denomi','nomi'])
    rebalancing_count_down=0
    for i in range(vol_day+day_delay,len(newStrat)):
        if (i==vol_day+day_delay):
            month_return=np.array(newStrat.loc[newStrat.index[i-day_delay]]/newStrat.loc[newStrat.index[i-day_return-day_delay]])
            ticker_index=(np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1]))
            weight=MD_weight_calculation(i,newStrat,ticker_index,vol_day,decay,day_delay)
            rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight
        elif (rebalance_type==1 and newStrat.index[i].day!=newStrat.index[i-1].day) or (rebalance_type==2 and newStrat.index[i].week!=newStrat.index[i-1].week) or (rebalance_type==3 and newStrat.index[i].month!=newStrat.index[i-1].month) or (rebalance_type==4 and newStrat.index[i].month!=newStrat.index[i-1].month and newStrat.index[i].month%3==0) :
            if rebalancing_count_down<=0:
                month_return=np.array(newStrat.loc[newStrat.index[i-day_delay]]/newStrat.loc[newStrat.index[i-day_return-day_delay]])
                ticker_index=np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1])
                rebalancing_count_down=Rebalancing_type2
#                inter_var.loc[i-vol_day-day_delay+1]=np.array(newStrat.loc[newStrat.index[i-none_zero_day1]]/newStrat.loc[newStrat.index[i-none_zero_day2]])
#                return_check.loc[i-vol_day-day_delay+1,'ret']=(newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]
#                return_check.loc[i-vol_day-day_delay+1,'denomi']=(newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]
#                return_check.loc[i-vol_day-day_delay+1,'nomi']=(newStrat_price.loc[i-vol_day-day_delay-1])[0]
            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-stop_loss) or (sum((newStrat.loc[newStrat.index[i-none_zero_day1]]/newStrat.loc[newStrat.index[i-none_zero_day2]])>1-none_zero_threshold))<none_zero_number:   
                weight_pot=np.zeros(len(newStrat.columns))
            else:
                weight_pot=MD_weight_calculation(i,newStrat,ticker_index,vol_day,decay,day_delay)
            if sum(abs(weight_pot-weight)>(threshold))>0:
                cost=0
                #cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                newStrat_price.loc[i-vol_day-day_delay+1]=(1-cost)*newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
                rebalance_index=i
                rebalance_time+=1
            else:
                newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))               
            weights_pd.loc[i-vol_day-day_delay+1]=weight
            rebalancing_count_down-=1
        else:
#            return_check.loc[i-vol_day-day_delay+1]=(newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]
#            return_check.loc[i-vol_day-day_delay+1,'denomi']=(newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]
#            return_check.loc[i-vol_day-day_delay+1,'nomi']=(newStrat_price.loc[i-vol_day-day_delay-1])[0]
            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-stop_loss):
                weight_pot=np.zeros(len(newStrat.columns))
#                cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight

    newStrat_price.index=newStrat.index[vol_day+day_delay-1:]
    weights_pd.index=newStrat.index[vol_day+day_delay:]
    Strat=Strategy(newStrat_price.index,newStrat_price['newStrat'])
    sharpe_temp=Strat.Sharpe()
    return (sharpe_temp,weights_pd,Strat,newStrat_price)
    
    


def Performance_Maximum_Diverisification_Strategy_Return_NoneZero_everyX_stoploss_everyday(newStrat,Rebalancing_type,MomentumType,Amount_Top,vol_day,day_delay,day_return,threshold,decay,stop_loss,none_zero_day1,none_zero_day2,none_zero_number,none_zero_threshold,Rebalancing_type2,iObservation_period):
    newStrat_price=pd.DataFrame(columns=['newStrat'])
    newStrat_price.loc[0]=100.0
    rebalance_index=vol_day-30
    rebalance_time=0
    weights_pd=pd.DataFrame(columns=["w" + str(i) for i in range(1,len(newStrat.columns)+1)])

    rebalance_type=0
    if Rebalancing_type=="monthly" or Rebalancing_type=="Monthly" :
        rebalance_type=3
    elif Rebalancing_type=="Quarterly" or Rebalancing_type=="quarterly" :
        rebalance_type=4
    elif Rebalancing_type=="Weekly" or Rebalancing_type=="weekly" :
        rebalance_type=2
    elif Rebalancing_type=="Daily" or Rebalancing_type=="daily" :
        rebalance_type=1 
#    inter_var=pd.DataFrame(columns=["ret" + str(i) for i in range(1,len(newStrat.columns)+1)])
#    return_check=pd.DataFrame(columns=['ret','denomi','nomi'])
    rebalancing_count_down=0
    for i in range(vol_day+day_delay,len(newStrat)):
        if (i==vol_day+day_delay):
            month_return=np.array(newStrat.loc[newStrat.index[i-day_delay]]/newStrat.loc[newStrat.index[i-day_return-day_delay]])
            ticker_index=(np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1]))
            weight=Performance_MD_weight_calculation(i,newStrat,ticker_index,vol_day,decay,day_delay)
            rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight
        elif (rebalance_type==1 and newStrat.index[i].day!=newStrat.index[i-1].day) or (rebalance_type==2 and newStrat.index[i].week!=newStrat.index[i-1].week) or (rebalance_type==3 and newStrat.index[i].month!=newStrat.index[i-1].month) or (rebalance_type==4 and newStrat.index[i].month!=newStrat.index[i-1].month and newStrat.index[i].month%3==0) :
            if rebalancing_count_down<=0:
                month_return=np.array(newStrat.loc[newStrat.index[i-day_delay]]/newStrat.loc[newStrat.index[i-day_return-day_delay]])
                ticker_index=np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1])
                rebalancing_count_down=Rebalancing_type2
#                inter_var.loc[i-vol_day-day_delay+1]=np.array(newStrat.loc[newStrat.index[i-none_zero_day1]]/newStrat.loc[newStrat.index[i-none_zero_day2]])
#                return_check.loc[i-vol_day-day_delay+1,'ret']=(newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]
#                return_check.loc[i-vol_day-day_delay+1,'denomi']=(newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]
#                return_check.loc[i-vol_day-day_delay+1,'nomi']=(newStrat_price.loc[i-vol_day-day_delay-1])[0]
            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-stop_loss) or (sum((newStrat.loc[newStrat.index[i-none_zero_day1]]/newStrat.loc[newStrat.index[i-none_zero_day2]])>1-none_zero_threshold))<none_zero_number:   
                weight_pot=np.zeros(len(newStrat.columns))
            else:
                weight_pot=Performance_MD_weight_calculation(i,newStrat,ticker_index,vol_day,decay,day_delay)
            if sum(abs(weight_pot-weight)>(threshold))>0:
                cost=0
                #cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                newStrat_price.loc[i-vol_day-day_delay+1]=(1-cost)*newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
                rebalance_index=i
                rebalance_time+=1
            else:
                newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))               
            weights_pd.loc[i-vol_day-day_delay+1]=weight
            rebalancing_count_down-=1
        else:
#            return_check.loc[i-vol_day-day_delay+1]=(newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]
#            return_check.loc[i-vol_day-day_delay+1,'denomi']=(newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]
#            return_check.loc[i-vol_day-day_delay+1,'nomi']=(newStrat_price.loc[i-vol_day-day_delay-1])[0]
            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-stop_loss):
                weight_pot=np.zeros(len(newStrat.columns))
#                cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight

    newStrat_price.index=newStrat.index[vol_day+day_delay-1:]
    weights_pd.index=newStrat.index[vol_day+day_delay:]
    Strat=Strategy(newStrat_price.index,newStrat_price['newStrat'])
    sharpe_temp=Strat.Sharpe()
    return (sharpe_temp,weights_pd,Strat,newStrat_price)
    

def Performance_Maximum_Diverisification_Strategy_Return_NoneZero_everyX_stoploss_everyday_ver2(newStrat,Rebalancing_type,MomentumType,Amount_Top,vol_day,day_delay,day_return,threshold,decay,stop_loss,none_zero_day1,none_zero_day2,none_zero_number,none_zero_threshold,Rebalancing_type2,iObservation_period):
    newStrat_price=pd.DataFrame(columns=['newStrat'])
    newStrat_price.loc[0]=100.0
    rebalance_index=vol_day-30
    rebalance_time=0
    weights_pd=pd.DataFrame(columns=["w" + str(i) for i in range(1,len(newStrat.columns)+1)])

    rebalance_type=0
    if Rebalancing_type=="monthly" or Rebalancing_type=="Monthly" :
        rebalance_type=3
    elif Rebalancing_type=="Quarterly" or Rebalancing_type=="quarterly" :
        rebalance_type=4
    elif Rebalancing_type=="Weekly" or Rebalancing_type=="weekly" :
        rebalance_type=2
    elif Rebalancing_type=="Daily" or Rebalancing_type=="daily" :
        rebalance_type=1 
#    inter_var=pd.DataFrame(columns=["ret" + str(i) for i in range(1,len(newStrat.columns)+1)])
#    return_check=pd.DataFrame(columns=['ret','denomi','nomi'])
    rebalancing_count_down=0
    for i in range(vol_day+day_delay,len(newStrat)):
        if (i==vol_day+day_delay):
            month_return=np.array(newStrat.loc[newStrat.index[i-day_delay]]/newStrat.loc[newStrat.index[i-day_return-day_delay]])
            ticker_index=(np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1]))
            weight=Performance_MD_weight_calculation_ver2(i,newStrat,ticker_index,vol_day,decay,day_delay)
            rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight
        elif (rebalance_type==1 and newStrat.index[i].day!=newStrat.index[i-1].day) or (rebalance_type==2 and newStrat.index[i].week!=newStrat.index[i-1].week) or (rebalance_type==3 and newStrat.index[i].month!=newStrat.index[i-1].month) or (rebalance_type==4 and newStrat.index[i].month!=newStrat.index[i-1].month and newStrat.index[i].month%3==0) :
            if rebalancing_count_down<=0:
                month_return=np.array(newStrat.loc[newStrat.index[i-day_delay]]/newStrat.loc[newStrat.index[i-day_return-day_delay]])
                ticker_index=np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1])
                rebalancing_count_down=Rebalancing_type2
#                inter_var.loc[i-vol_day-day_delay+1]=np.array(newStrat.loc[newStrat.index[i-none_zero_day1]]/newStrat.loc[newStrat.index[i-none_zero_day2]])
#                return_check.loc[i-vol_day-day_delay+1,'ret']=(newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]
#                return_check.loc[i-vol_day-day_delay+1,'denomi']=(newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]
#                return_check.loc[i-vol_day-day_delay+1,'nomi']=(newStrat_price.loc[i-vol_day-day_delay-1])[0]
            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-stop_loss) or (sum((newStrat.loc[newStrat.index[i-none_zero_day1]]/newStrat.loc[newStrat.index[i-none_zero_day2]])>1-none_zero_threshold))<none_zero_number:   
                weight_pot=np.zeros(len(newStrat.columns))
            else:
                weight_pot=Performance_MD_weight_calculation_ver2(i,newStrat,ticker_index,vol_day,decay,day_delay)
            if sum(abs(weight_pot-weight)>(threshold))>0:
                cost=0
                #cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                newStrat_price.loc[i-vol_day-day_delay+1]=(1-cost)*newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
                rebalance_index=i
                rebalance_time+=1
            else:
                newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))               
            weights_pd.loc[i-vol_day-day_delay+1]=weight
            rebalancing_count_down-=1
        else:
#            return_check.loc[i-vol_day-day_delay+1]=(newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]
#            return_check.loc[i-vol_day-day_delay+1,'denomi']=(newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]
#            return_check.loc[i-vol_day-day_delay+1,'nomi']=(newStrat_price.loc[i-vol_day-day_delay-1])[0]
            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-stop_loss):
                weight_pot=np.zeros(len(newStrat.columns))
#                cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight

    newStrat_price.index=newStrat.index[vol_day+day_delay-1:]
    weights_pd.index=newStrat.index[vol_day+day_delay:]
    Strat=Strategy(newStrat_price.index,newStrat_price['newStrat'])
    sharpe_temp=Strat.Sharpe()
    return (sharpe_temp,weights_pd,Strat,newStrat_price)
    
    
    

def Maximum_Diverisification_Strategy_Return_NoneZero_Weight_momentum_everyX__stoploss_everyday(newStrat,Rebalancing_type,MomentumType,Amount_Top,vol_day,day_delay,day_return,threshold,decay,stop_loss,none_zero_day1,none_zero_day2,none_zero_number,none_zero_threshold,Rebalancing_type2,iObservation_period):
    newStrat_price=pd.DataFrame(columns=['newStrat'])
    newStrat_price.loc[0]=100.0
    rebalance_index=vol_day-30
    rebalance_time=0
    weights_pd=pd.DataFrame(columns=["w" + str(i) for i in range(1,len(newStrat.columns)+1)])

    rebalance_type=0
    if Rebalancing_type=="monthly" or Rebalancing_type=="Monthly" :
        rebalance_type=3
    elif Rebalancing_type=="Quarterly" or Rebalancing_type=="quarterly" :
        rebalance_type=4
    elif Rebalancing_type=="Weekly" or Rebalancing_type=="weekly" :
        rebalance_type=2
    elif Rebalancing_type=="Daily" or Rebalancing_type=="daily" :
        rebalance_type=1 
#    inter_var=pd.DataFrame(columns=["ret" + str(i) for i in range(1,len(newStrat.columns)+1)])
#    return_check=pd.DataFrame(columns=['ret','denomi','nomi'])
    rebalancing_count_down=0
    for i in range(vol_day+day_delay,len(newStrat)):
        if (i==vol_day+day_delay):
            month_return=np.array(newStrat.loc[newStrat.index[i-day_delay]]/newStrat.loc[newStrat.index[i-day_return-day_delay]])
            ticker_index=(np.sort(two_side_filter(month_return,0,MomentumType)[1]))
            weight=MD_weight_calculation_momentum(i,newStrat,ticker_index,vol_day,decay,day_delay,Amount_Top)
            rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight
        elif (rebalance_type==1 and newStrat.index[i].day!=newStrat.index[i-1].day) or (rebalance_type==2 and newStrat.index[i].week!=newStrat.index[i-1].week) or (rebalance_type==3 and newStrat.index[i].month!=newStrat.index[i-1].month) or (rebalance_type==4 and newStrat.index[i].month!=newStrat.index[i-1].month and newStrat.index[i].month%3==0) :
            if rebalancing_count_down<=0:
                month_return=np.array(newStrat.loc[newStrat.index[i-day_delay]]/newStrat.loc[newStrat.index[i-day_return-day_delay]])
                ticker_index=np.sort(two_side_filter(month_return,0,MomentumType)[1])
                rebalancing_count_down=Rebalancing_type2
#                inter_var.loc[i-vol_day-day_delay+1]=np.array(newStrat.loc[newStrat.index[i-none_zero_day1]]/newStrat.loc[newStrat.index[i-none_zero_day2]])
#                return_check.loc[i-vol_day-day_delay+1,'ret']=(newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]
#                return_check.loc[i-vol_day-day_delay+1,'denomi']=(newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]
#                return_check.loc[i-vol_day-day_delay+1,'nomi']=(newStrat_price.loc[i-vol_day-day_delay-1])[0]
            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-stop_loss) or (sum((newStrat.loc[newStrat.index[i-none_zero_day1]]/newStrat.loc[newStrat.index[i-none_zero_day2]])>1-none_zero_threshold))<none_zero_number:   
                weight_pot=np.zeros(len(newStrat.columns))
            else:
                weight_pot=MD_weight_calculation_momentum(i,newStrat,ticker_index,vol_day,decay,day_delay,Amount_Top)
            if sum(abs(weight_pot-weight)>(threshold))>0:
                cost=0
                #cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                newStrat_price.loc[i-vol_day-day_delay+1]=(1-cost)*newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
                rebalance_index=i
                rebalance_time+=1
            else:
                newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))               
            weights_pd.loc[i-vol_day-day_delay+1]=weight
            rebalancing_count_down-=1
        else:
#            return_check.loc[i-vol_day-day_delay+1]=(newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]
#            return_check.loc[i-vol_day-day_delay+1,'denomi']=(newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]
#            return_check.loc[i-vol_day-day_delay+1,'nomi']=(newStrat_price.loc[i-vol_day-day_delay-1])[0]
            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-stop_loss):
                weight_pot=np.zeros(len(newStrat.columns))
#                cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight

    newStrat_price.index=newStrat.index[vol_day+day_delay-1:]
    weights_pd.index=newStrat.index[vol_day+day_delay:]
    Strat=Strategy(newStrat_price.index,newStrat_price['newStrat'])
    sharpe_temp=Strat.Sharpe()
    return (sharpe_temp,weights_pd,Strat,newStrat_price)
    
    
    

def Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec_last30_stop_loss(newStrat,Rebalancing_type,MomentumType,Amount_Top,vol_day,day_delay,day_return,threshold,decay,stop_loss,none_zero_day1,none_zero_day2,none_zero_number,none_zero_threshold):
    newStrat_price=pd.DataFrame(columns=['newStrat'])
    newStrat_price.loc[0]=100.0
    rebalance_index=vol_day-30
    rebalance_time=0
    weights_pd=pd.DataFrame(columns=["w" + str(i) for i in range(1,len(newStrat.columns)+1)])

    rebalance_type=0
    if Rebalancing_type=="monthly" or Rebalancing_type=="Monthly" :
        rebalance_type=3
    elif Rebalancing_type=="Quarterly" or Rebalancing_type=="quarterly" :
        rebalance_type=4
    elif Rebalancing_type=="Weekly" or Rebalancing_type=="weekly" :
        rebalance_type=2
    elif Rebalancing_type=="Daily" or Rebalancing_type=="daily" :
        rebalance_type=1 
        
    for i in range(vol_day+day_delay,len(newStrat)):
        if (i==vol_day+day_delay):
            month_return=np.array(newStrat.loc[newStrat.index[i-day_delay]]/newStrat.loc[newStrat.index[i-day_return-day_delay]])
            ticker_index=(np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1]))
            weight=MD_weight_calculation(i,newStrat,ticker_index,vol_day,decay,day_delay)
            rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight
        elif (rebalance_type==1 and newStrat.index[i].day!=newStrat.index[i-1].day) or (rebalance_type==2 and newStrat.index[i].week!=newStrat.index[i-1].week) or (rebalance_type==3 and newStrat.index[i].month!=newStrat.index[i-1].month) or (rebalance_type==4 and newStrat.index[i].month!=newStrat.index[i-1].month and newStrat.index[i].month%3==0) :
            month_return=np.array(newStrat.loc[newStrat.index[i-day_delay]]/newStrat.loc[newStrat.index[i-day_return-day_delay]])
            ticker_index=np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1])
            if i-vol_day-day_delay-60>0 and (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[i-vol_day-day_delay-60])[0]<(1-stop_loss) or (sum((newStrat.loc[newStrat.index[i-none_zero_day1]]/newStrat.loc[newStrat.index[i-none_zero_day2]])>1-none_zero_threshold))<none_zero_number:   
                weight_pot=np.zeros(len(newStrat.columns))
            else:
                weight_pot=MD_weight_calculation(i,newStrat,ticker_index,vol_day,decay,day_delay)
            if sum(abs(weight_pot-weight)>(threshold))>0:
                cost=0
                #cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                newStrat_price.loc[i-vol_day-day_delay+1]=(1-cost)*newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
                rebalance_index=i
                rebalance_time+=1
            else:
                newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))               
            weights_pd.loc[i-vol_day-day_delay+1]=weight
        else:
            if i-vol_day-day_delay-60>0 and (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[i-vol_day-day_delay-60])[0]<(1-stop_loss):
                weight_pot=np.zeros(len(newStrat.columns))
                cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight

    newStrat_price.index=newStrat.index[vol_day+day_delay-1:]
    weights_pd.index=newStrat.index[vol_day+day_delay:]
    Strat=Strategy(newStrat_price.index,newStrat_price['newStrat'])
    sharpe_temp=Strat.Sharpe()
    return (sharpe_temp,weights_pd,Strat,newStrat_price)

    
    


def Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec_Individual_momentum_everyX__stoploss_everyday_underindex_last30(newStrat,Rebalancing_type,MomentumType,Amount_Top,vol_day,day_delay,day_return,threshold,decay,stop_loss,none_zero_day1,none_zero_day2,none_zero_number,none_zero_threshold,Rebalancing_type2,iObservation_period):
    newStrat_price=pd.DataFrame(columns=['newStrat'])
    newStrat_price.loc[0]=100.0
    rebalance_index=vol_day-30
    rebalance_time=0
    weights_pd=pd.DataFrame(columns=["w" + str(i) for i in range(1,len(newStrat.columns)+1)])

    rebalance_type=0
    if Rebalancing_type=="monthly" or Rebalancing_type=="Monthly" :
        rebalance_type=3
    elif Rebalancing_type=="Quarterly" or Rebalancing_type=="quarterly" :
        rebalance_type=4
    elif Rebalancing_type=="Weekly" or Rebalancing_type=="weekly" :
        rebalance_type=2
    elif Rebalancing_type=="Daily" or Rebalancing_type=="daily" :
        rebalance_type=1 
#    inter_var=pd.DataFrame(columns=["ret" + str(i) for i in range(1,len(newStrat.columns)+1)])

    rebalancing_count_down=0
    for i in range(vol_day+day_delay,len(newStrat)):
        if (i==vol_day+day_delay):
            month_return=np.array(newStrat.loc[newStrat.index[i-day_delay]]/newStrat.loc[newStrat.index[i-day_return-day_delay]])
            ticker_index=(np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1]))
            weight=MD_weight_calculation(i,newStrat,ticker_index,vol_day,decay,day_delay)
            rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight
        elif (rebalance_type==1 and newStrat.index[i].day!=newStrat.index[i-1].day) or (rebalance_type==2 and newStrat.index[i].week!=newStrat.index[i-1].week) or (rebalance_type==3 and newStrat.index[i].month!=newStrat.index[i-1].month) or (rebalance_type==4 and newStrat.index[i].month!=newStrat.index[i-1].month and newStrat.index[i].month%3==0) :
            if rebalancing_count_down<=0:
                month_return=np.array(newStrat.loc[newStrat.index[i-day_delay]]/newStrat.loc[newStrat.index[i-day_return-day_delay]])
                ticker_index=np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1])
                rebalancing_count_down=Rebalancing_type2
#                inter_var.loc[i-vol_day-day_delay+1]=np.array(newStrat.loc[newStrat.index[i-none_zero_day1]]/newStrat.loc[newStrat.index[i-none_zero_day2]])
            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-0.9) or (sum((newStrat.loc[newStrat.index[i-none_zero_day1]]/newStrat.loc[newStrat.index[i-none_zero_day2]])>1-none_zero_threshold))<1:   
                weight_pot=np.zeros(len(newStrat.columns))
            else:
                weight_pot=MD_weight_calculation(i,newStrat,ticker_index,vol_day,decay,day_delay)
            if sum(abs(weight_pot-weight)>(threshold))>0:
                cost=0
                #cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                newStrat_price.loc[i-vol_day-day_delay+1]=(1-cost)*newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
                rebalance_index=i
                rebalance_time+=1
            else:
                newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))               
            weights_pd.loc[i-vol_day-day_delay+1]=weight
            rebalancing_count_down-=1
        else:
            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-0.9):
                weight_pot=np.zeros(len(newStrat.columns))
                cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight

    newStrat_price.index=newStrat.index[vol_day+day_delay-1:]
    newStrat_price2=pd.DataFrame(columns=['newStrat'])
#    weights_pd['w']=0
    newStrat_price2.loc[0]=100.0
    rebalancing_index=0
    w=1.0
    for k in range(1,len(newStrat_price)):
#        weights_pd.loc[k,'w']=(newStrat_price.loc[newStrat_price.index[k-2]]/newStrat_price.loc[newStrat_price.index[rebalancing_index]])[0]
        if  k-iObservation_period>0 and (newStrat_price.loc[newStrat_price.index[k-2]]/newStrat_price.loc[newStrat_price.index[k-iObservation_period]])[0]<(1-stop_loss):
            w=0.0
#            rebalancing_index=k-1
        if  (rebalance_type==1 and newStrat_price.index[k].day!=newStrat_price.index[k-1].day) or (rebalance_type==2 and newStrat_price.index[k].week!=newStrat_price.index[k-1].week) or (rebalance_type==3 and newStrat_price.index[k].month!=newStrat_price.index[k-1].month) or (rebalance_type==4 and newStrat_price.index[k].month!=newStrat_price.index[k-1].month and newStrat_price.index[k].month%3==0) :
            if k-iObservation_period>0 and(newStrat_price.loc[newStrat_price.index[k-2]]/newStrat_price.loc[newStrat_price.index[k-iObservation_period]])[0]>=(1-stop_loss):
                w=1.0
            rebalancing_index=k-1
        newStrat_price2.loc[k]=newStrat_price2.loc[k-1]*(1+w*(newStrat_price.loc[newStrat_price.index[k]]/newStrat_price.loc[newStrat_price.index[k-1]]-1))
        
    weights_pd.index=newStrat.index[vol_day+day_delay:]
    Strat=Strategy(newStrat_price.index,newStrat_price2['newStrat'])
    sharpe_temp=Strat.Sharpe()
    return (sharpe_temp,weights_pd,Strat,newStrat_price2)
    
    
def Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec_Individual_momentum_everyX__stoploss_everyday_underindex_MA(newStrat,Rebalancing_type,MomentumType,Amount_Top,vol_day,day_delay,day_return,threshold,decay,stop_loss,none_zero_day1,none_zero_day2,none_zero_number,none_zero_threshold,Rebalancing_type2,iObservation_period):
    newStrat_price=pd.DataFrame(columns=['newStrat'])
    newStrat_price.loc[0]=100.0
    rebalance_index=vol_day-30
    rebalance_time=0
    weights_pd=pd.DataFrame(columns=["w" + str(i) for i in range(1,len(newStrat.columns)+1)])

    rebalance_type=0
    if Rebalancing_type=="monthly" or Rebalancing_type=="Monthly" :
        rebalance_type=3
    elif Rebalancing_type=="Quarterly" or Rebalancing_type=="quarterly" :
        rebalance_type=4
    elif Rebalancing_type=="Weekly" or Rebalancing_type=="weekly" :
        rebalance_type=2
    elif Rebalancing_type=="Daily" or Rebalancing_type=="daily" :
        rebalance_type=1 
#    inter_var=pd.DataFrame(columns=["ret" + str(i) for i in range(1,len(newStrat.columns)+1)])

    rebalancing_count_down=0
    for i in range(vol_day+day_delay,len(newStrat)):
        if (i==vol_day+day_delay):
            month_return=np.array(newStrat.loc[newStrat.index[i-day_delay]]/newStrat.loc[newStrat.index[i-day_return-day_delay]])
            ticker_index=(np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1]))
            weight=MD_weight_calculation(i,newStrat,ticker_index,vol_day,decay,day_delay)
            rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight
        elif (rebalance_type==1 and newStrat.index[i].day!=newStrat.index[i-1].day) or (rebalance_type==2 and newStrat.index[i].week!=newStrat.index[i-1].week) or (rebalance_type==3 and newStrat.index[i].month!=newStrat.index[i-1].month) or (rebalance_type==4 and newStrat.index[i].month!=newStrat.index[i-1].month and newStrat.index[i].month%3==0) :
            if rebalancing_count_down<=0:
                month_return=np.array(newStrat.loc[newStrat.index[i-day_delay]]/newStrat.loc[newStrat.index[i-day_return-day_delay]])
                ticker_index=np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1])
                rebalancing_count_down=Rebalancing_type2
#                inter_var.loc[i-vol_day-day_delay+1]=np.array(newStrat.loc[newStrat.index[i-none_zero_day1]]/newStrat.loc[newStrat.index[i-none_zero_day2]])
            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-0.9) or (sum((newStrat.loc[newStrat.index[i-none_zero_day1]]/newStrat.loc[newStrat.index[i-none_zero_day2]])>1-none_zero_threshold))<1:   
                weight_pot=np.zeros(len(newStrat.columns))
            else:
                weight_pot=MD_weight_calculation(i,newStrat,ticker_index,vol_day,decay,day_delay)
            if sum(abs(weight_pot-weight)>(threshold))>0:
                cost=0
                #cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                newStrat_price.loc[i-vol_day-day_delay+1]=(1-cost)*newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
                rebalance_index=i
                rebalance_time+=1
            else:
                newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))               
            weights_pd.loc[i-vol_day-day_delay+1]=weight
            rebalancing_count_down-=1
        else:
            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-0.9):
                weight_pot=np.zeros(len(newStrat.columns))
                cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight

    newStrat_price.index=newStrat.index[vol_day+day_delay-1:]
    newStrat_price2=pd.DataFrame(columns=['newStrat'])
#    weights_pd['w']=0
    newStrat_price2.loc[0]=100.0
    rebalancing_index=0
    w=1.0
    for k in range(1,len(newStrat_price)):
#        weights_pd.loc[k,'w']=(newStrat_price.loc[newStrat_price.index[k-2]]/newStrat_price.loc[newStrat_price.index[rebalancing_index]])[0]
        if  k-iObservation_period>0 and np.average(newStrat_price.loc[newStrat_price.index[(k-7):(k-2)]])<np.average(newStrat_price.loc[newStrat_price.index[(k-22):(k-2)]]) and np.average(newStrat_price.loc[newStrat_price.index[(k-22):(k-2)]])<np.average(newStrat_price.loc[newStrat_price.index[(k-iObservation_period):(k-2)]]):
            w=0.0
            print(1)
#            rebalancing_index=k-1
        if  (rebalance_type==1 and newStrat_price.index[k].day!=newStrat_price.index[k-1].day) or (rebalance_type==2 and newStrat_price.index[k].week!=newStrat_price.index[k-1].week) or (rebalance_type==3 and newStrat_price.index[k].month!=newStrat_price.index[k-1].month) or (rebalance_type==4 and newStrat_price.index[k].month!=newStrat_price.index[k-1].month and newStrat_price.index[k].month%3==0) :
            if (k-iObservation_period>0) and (not (np.average(newStrat_price.loc[newStrat_price.index[(k-7):(k-2)]])<np.average(newStrat_price.loc[newStrat_price.index[(k-22):(k-2)]]) and np.average(newStrat_price.loc[newStrat_price.index[(k-22):(k-2)]])<np.average(newStrat_price.loc[newStrat_price.index[(k-iObservation_period):(k-2)]]))):
                w=1.0
            rebalancing_index=k-1
        newStrat_price2.loc[k]=newStrat_price2.loc[k-1]*(1+w*(newStrat_price.loc[newStrat_price.index[k]]/newStrat_price.loc[newStrat_price.index[k-1]]-1))
        
    weights_pd.index=newStrat.index[vol_day+day_delay:]
    Strat=Strategy(newStrat_price.index,newStrat_price2['newStrat'])
    sharpe_temp=Strat.Sharpe()
    return (sharpe_temp,weights_pd,Strat,newStrat_price2)
    

    
def Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec_Individual_momentum_everyX__stoploss_everyday_underindex_MA(newStrat,Rebalancing_type,MomentumType,Amount_Top,vol_day,day_delay,day_return,threshold,decay,stop_loss,none_zero_day1,none_zero_day2,none_zero_number,none_zero_threshold,Rebalancing_type2,iObservation_period):
    newStrat_price=pd.DataFrame(columns=['newStrat'])
    newStrat_price.loc[0]=100.0
    rebalance_index=vol_day-30
    rebalance_time=0
    weights_pd=pd.DataFrame(columns=["w" + str(i) for i in range(1,len(newStrat.columns)+1)])

    rebalance_type=0
    if Rebalancing_type=="monthly" or Rebalancing_type=="Monthly" :
        rebalance_type=3
    elif Rebalancing_type=="Quarterly" or Rebalancing_type=="quarterly" :
        rebalance_type=4
    elif Rebalancing_type=="Weekly" or Rebalancing_type=="weekly" :
        rebalance_type=2
    elif Rebalancing_type=="Daily" or Rebalancing_type=="daily" :
        rebalance_type=1 
#    inter_var=pd.DataFrame(columns=["ret" + str(i) for i in range(1,len(newStrat.columns)+1)])

    rebalancing_count_down=0
    for i in range(vol_day+day_delay,len(newStrat)):
        if (i==vol_day+day_delay):
            month_return=np.array(newStrat.loc[newStrat.index[i-day_delay]]/newStrat.loc[newStrat.index[i-day_return-day_delay]])
            ticker_index=(np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1]))
            weight=MD_weight_calculation(i,newStrat,ticker_index,vol_day,decay,day_delay)
            rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight
        elif (rebalance_type==1 and newStrat.index[i].day!=newStrat.index[i-1].day) or (rebalance_type==2 and newStrat.index[i].week!=newStrat.index[i-1].week) or (rebalance_type==3 and newStrat.index[i].month!=newStrat.index[i-1].month) or (rebalance_type==4 and newStrat.index[i].month!=newStrat.index[i-1].month and newStrat.index[i].month%3==0) :
            if rebalancing_count_down<=0:
                month_return=np.array(newStrat.loc[newStrat.index[i-day_delay]]/newStrat.loc[newStrat.index[i-day_return-day_delay]])
                ticker_index=np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1])
                rebalancing_count_down=Rebalancing_type2
#                inter_var.loc[i-vol_day-day_delay+1]=np.array(newStrat.loc[newStrat.index[i-none_zero_day1]]/newStrat.loc[newStrat.index[i-none_zero_day2]])
            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-0.9) or (sum((newStrat.loc[newStrat.index[i-none_zero_day1]]/newStrat.loc[newStrat.index[i-none_zero_day2]])>1-none_zero_threshold))<none_zero_number:   
                weight_pot=np.zeros(len(newStrat.columns))
            else:
                weight_pot=MD_weight_calculation(i,newStrat,ticker_index,vol_day,decay,day_delay)
            if sum(abs(weight_pot-weight)>(threshold))>0:
                cost=0
                #cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                newStrat_price.loc[i-vol_day-day_delay+1]=(1-cost)*newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
                rebalance_index=i
                rebalance_time+=1
            else:
                newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))               
            weights_pd.loc[i-vol_day-day_delay+1]=weight
            rebalancing_count_down-=1
        else:
            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-0.9):
                weight_pot=np.zeros(len(newStrat.columns))
                cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight

    newStrat_price.index=newStrat.index[vol_day+day_delay-1:]
    newStrat_price2=pd.DataFrame(columns=['newStrat'])
#    weights_pd['w']=0
    newStrat_price2.loc[0]=100.0
    rebalancing_index=0
    w=1.0
    print(rebalance_time)
    for k in range(1,len(newStrat_price)):
#        weights_pd.loc[k,'w']=(newStrat_price.loc[newStrat_price.index[k-2]]/newStrat_price.loc[newStrat_price.index[rebalancing_index]])[0]
        if  k-iObservation_period>0 and np.average(newStrat_price.loc[newStrat_price.index[(k-7):(k-2)]])<np.average(newStrat_price.loc[newStrat_price.index[(k-iObservation_period):(k-2)]]) and np.average(newStrat_price.loc[newStrat_price.index[(k-iObservation_period):(k-2)]])<np.average(newStrat_price.loc[newStrat_price.index[(k-250):(k-2)]]):
            w=0.0
            print(1)
#            rebalancing_index=k-1
        if  (rebalance_type==1 and newStrat_price.index[k].day!=newStrat_price.index[k-1].day) or (rebalance_type==2 and newStrat_price.index[k].week!=newStrat_price.index[k-1].week) or (rebalance_type==3 and newStrat_price.index[k].month!=newStrat_price.index[k-1].month) or (rebalance_type==4 and newStrat_price.index[k].month!=newStrat_price.index[k-1].month and newStrat_price.index[k].month%3==0) :
            if (k-iObservation_period>0) and (not (np.average(newStrat_price.loc[newStrat_price.index[(k-7):(k-2)]])<np.average(newStrat_price.loc[newStrat_price.index[(k-iObservation_period):(k-2)]]) and np.average(newStrat_price.loc[newStrat_price.index[(k-iObservation_period):(k-2)]])<np.average(newStrat_price.loc[newStrat_price.index[(k-250):(k-2)]]))):
                w=1.0
            rebalancing_index=k-1
        newStrat_price2.loc[k]=newStrat_price2.loc[k-1]*(1+w*(newStrat_price.loc[newStrat_price.index[k]]/newStrat_price.loc[newStrat_price.index[k-1]]-1))
        
    weights_pd.index=newStrat.index[vol_day+day_delay:]
    Strat=Strategy(newStrat_price.index,newStrat_price2['newStrat'])
    sharpe_temp=Strat.Sharpe()
    return (sharpe_temp,weights_pd,Strat,newStrat_price2)
    
def Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec_Individual_momentum_everyX__stoploss_everyday_underindex_MA_no_stop_loss(newStrat,Rebalancing_type,MomentumType,Amount_Top,vol_day,day_delay,day_return,threshold,decay,stop_loss,none_zero_day1,none_zero_day2,none_zero_number,none_zero_threshold,Rebalancing_type2,iObservation_period):
    newStrat_price=pd.DataFrame(columns=['newStrat'])
    newStrat_price.loc[0]=100.0
    rebalance_index=vol_day-30
    rebalance_time=0
    weights_pd=pd.DataFrame(columns=["w" + str(i) for i in range(1,len(newStrat.columns)+1)])

    rebalance_type=0
    if Rebalancing_type=="monthly" or Rebalancing_type=="Monthly" :
        rebalance_type=3
    elif Rebalancing_type=="Quarterly" or Rebalancing_type=="quarterly" :
        rebalance_type=4
    elif Rebalancing_type=="Weekly" or Rebalancing_type=="weekly" :
        rebalance_type=2
    elif Rebalancing_type=="Daily" or Rebalancing_type=="daily" :
        rebalance_type=1 
#    inter_var=pd.DataFrame(columns=["ret" + str(i) for i in range(1,len(newStrat.columns)+1)])

    rebalancing_count_down=0
    for i in range(vol_day+day_delay,len(newStrat)):
        if (i==vol_day+day_delay):
            month_return=np.array(newStrat.loc[newStrat.index[i-day_delay]]/newStrat.loc[newStrat.index[i-day_return-day_delay]])
            ticker_index=(np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1]))
            weight=MD_weight_calculation(i,newStrat,ticker_index,vol_day,decay,day_delay)
            rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight
        elif (rebalance_type==1 and newStrat.index[i].day!=newStrat.index[i-1].day) or (rebalance_type==2 and newStrat.index[i].week!=newStrat.index[i-1].week) or (rebalance_type==3 and newStrat.index[i].month!=newStrat.index[i-1].month) or (rebalance_type==4 and newStrat.index[i].month!=newStrat.index[i-1].month and newStrat.index[i].month%3==0) :
            if rebalancing_count_down<=0:
                month_return=np.array(newStrat.loc[newStrat.index[i-day_delay]]/newStrat.loc[newStrat.index[i-day_return-day_delay]])
                ticker_index=np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1])
                rebalancing_count_down=Rebalancing_type2
#                inter_var.loc[i-vol_day-day_delay+1]=np.array(newStrat.loc[newStrat.index[i-none_zero_day1]]/newStrat.loc[newStrat.index[i-none_zero_day2]])
            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-0.9) or (sum((newStrat.loc[newStrat.index[i-none_zero_day1]]/newStrat.loc[newStrat.index[i-none_zero_day2]])>1-none_zero_threshold))<none_zero_number:   
                weight_pot=np.zeros(len(newStrat.columns))
            else:
                weight_pot=MD_weight_calculation(i,newStrat,ticker_index,vol_day,decay,day_delay)
            if sum(abs(weight_pot-weight)>(threshold))>0:
                cost=0
                #cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                newStrat_price.loc[i-vol_day-day_delay+1]=(1-cost)*newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
                rebalance_index=i
                rebalance_time+=1
            else:
                newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))               
            weights_pd.loc[i-vol_day-day_delay+1]=weight
            rebalancing_count_down-=1
        else:
            if (newStrat_price.loc[i-vol_day-day_delay-1]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-0.9):
                weight_pot=np.zeros(len(newStrat.columns))
                cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
                weight=weight_pot
                rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight

    newStrat_price.index=newStrat.index[vol_day+day_delay-1:]

#    weights_pd['w']=0
    print(rebalance_time)

    weights_pd.index=newStrat.index[vol_day+day_delay:]
    Strat=Strategy(newStrat_price.index,newStrat_price['newStrat'])
    sharpe_temp=Strat.Sharpe()
    return (sharpe_temp,weights_pd,Strat,newStrat_price)
    
    
    
def Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec_Individual_momentum_everyX__stoploss_everyday_underindex_MA_simple(newStrat,Rebalancing_type,MomentumType,Amount_Top,vol_day,day_delay,day_return,threshold,decay,stop_loss,none_zero_day1,none_zero_day2,none_zero_number,none_zero_threshold,Rebalancing_type2,iObservation_period):
    newStrat_price=pd.DataFrame(columns=['newStrat'])
    newStrat_price.loc[0]=100.0
    rebalance_index=vol_day-30
    rebalance_time=0
    weights_pd=pd.DataFrame(columns=["w" + str(i) for i in range(1,len(newStrat.columns)+1)])

    rebalance_type=0
    if Rebalancing_type=="monthly" or Rebalancing_type=="Monthly" :
        rebalance_type=3
    elif Rebalancing_type=="Quarterly" or Rebalancing_type=="quarterly" :
        rebalance_type=4
    elif Rebalancing_type=="Weekly" or Rebalancing_type=="weekly" :
        rebalance_type=2
    elif Rebalancing_type=="Daily" or Rebalancing_type=="daily" :
        rebalance_type=1 
#    inter_var=pd.DataFrame(columns=["ret" + str(i) for i in range(1,len(newStrat.columns)+1)])

    rebalancing_count_down=0
    for i in range(vol_day+day_delay,len(newStrat)):
        if (i==vol_day+day_delay):
            month_return=np.array(newStrat.loc[newStrat.index[i-day_delay]]/newStrat.loc[newStrat.index[i-day_return-day_delay]])
            ticker_index=(np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1]))
            weight=MD_weight_calculation(i,newStrat,ticker_index,vol_day,decay,day_delay)
            rebalance_index=i
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight
        elif (rebalance_type==1 and newStrat.index[i].day!=newStrat.index[i-1].day) or (rebalance_type==2 and newStrat.index[i].week!=newStrat.index[i-1].week) or (rebalance_type==3 and newStrat.index[i].month!=newStrat.index[i-1].month) or (rebalance_type==4 and newStrat.index[i].month!=newStrat.index[i-1].month and newStrat.index[i].month%3==0) :

            month_return=np.array(newStrat.loc[newStrat.index[i-day_delay]]/newStrat.loc[newStrat.index[i-day_return-day_delay]])
            ticker_index=np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1])
            rebalancing_count_down=Rebalancing_type2
            weight_pot=MD_weight_calculation(i,newStrat,ticker_index,vol_day,decay,day_delay)
            cost=0
            weight=weight_pot
            newStrat_price.loc[i-vol_day-day_delay+1]=(1-cost)*newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
            rebalance_index=i
            rebalance_time+=1         
            weights_pd.loc[i-vol_day-day_delay+1]=weight
        else:
            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))
            weights_pd.loc[i-vol_day-day_delay+1]=weight

    newStrat_price.index=newStrat.index[vol_day+day_delay-1:]
    newStrat_price2=pd.DataFrame(columns=['newStrat'])
#    weights_pd['w']=0
    newStrat_price2.loc[0]=100.0
    rebalancing_index=0
    w=1.0
    print(rebalance_time)
    for k in range(1,len(newStrat_price)):
#        weights_pd.loc[k,'w']=(newStrat_price.loc[newStrat_price.index[k-2]]/newStrat_price.loc[newStrat_price.index[rebalancing_index]])[0]
        if  k-250>0 and np.average(newStrat_price.loc[newStrat_price.index[(k-7):(k-2)]])<np.average(newStrat_price.loc[newStrat_price.index[(k-iObservation_period):(k-2)]]) and np.average(newStrat_price.loc[newStrat_price.index[(k-iObservation_period):(k-2)]])<np.average(newStrat_price.loc[newStrat_price.index[(k-250):(k-2)]]):
            w=0.0
            print(1)
#            rebalancing_index=k-1
        if  (rebalance_type==1 and newStrat_price.index[k].day!=newStrat_price.index[k-1].day) or (rebalance_type==2 and newStrat_price.index[k].week!=newStrat_price.index[k-1].week) or (rebalance_type==3 and newStrat_price.index[k].month!=newStrat_price.index[k-1].month) or (rebalance_type==4 and newStrat_price.index[k].month!=newStrat_price.index[k-1].month and newStrat_price.index[k].month%3==0) :
            if (k-250>0) and (not (np.average(newStrat_price.loc[newStrat_price.index[(k-7):(k-2)]])<np.average(newStrat_price.loc[newStrat_price.index[(k-iObservation_period):(k-2)]]) and np.average(newStrat_price.loc[newStrat_price.index[(k-iObservation_period):(k-2)]])<np.average(newStrat_price.loc[newStrat_price.index[(k-250):(k-2)]]))):
                w=1.0
            rebalancing_index=k-1
        newStrat_price2.loc[k]=newStrat_price2.loc[k-1]*(1+w*(newStrat_price.loc[newStrat_price.index[k]]/newStrat_price.loc[newStrat_price.index[k-1]]-1))
        
    weights_pd.index=newStrat.index[vol_day+day_delay:]
    Strat=Strategy(newStrat_price.index,newStrat_price2['newStrat'])
    sharpe_temp=Strat.Sharpe()
    return (sharpe_temp,weights_pd,Strat,newStrat_price2)
    
    
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

    
#Rainbow_execute(dfPrice,ContractList,Rebalancing_type,iObservation_period,RainbowCoef)


#def Maximum_Diverisification_Strategy(newStrat,Rebalancing_type,MomentumType,Amount_Top,vol_day,day_delay,day_return,threshold,decay,stop_loss):
#    newStrat_price=pd.DataFrame(columns=['newStrat'])
#    newStrat_price.loc[0]=100.0
#    rebalance_index=vol_day-30
#    rebalance_time=0
#    weights_pd=pd.DataFrame(columns=["w" + str(i) for i in range(1,len(newStrat.columns)+1)])
#
#    rebalance_type=0
#    if Rebalancing_type=="monthly" or Rebalancing_type=="Monthly" :
#        rebalance_type=3
#    elif Rebalancing_type=="Quarterly" or Rebalancing_type=="quarterly" :
#        rebalance_type=4
#    elif Rebalancing_type=="Weekly" or Rebalancing_type=="weekly" :
#        rebalance_type=2
#    elif Rebalancing_type=="Daily" or Rebalancing_type=="daily" :
#        rebalance_type=1   
#    
#    for i in range(vol_day+day_delay,len(newStrat)):
#        if (i==vol_day+day_delay):
#            month_return=np.array(newStrat.loc[newStrat.index[i-2]]/newStrat.loc[newStrat.index[i-day_return]])
#            ticker_index=np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1])
#            weight=MD_weight_calculation(i,newStrat,ticker_index,vol_day,decay,day_delay)
#            rebalance_index=i
#            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
#            weights_pd.loc[i-vol_day-day_delay+1]=weight
#        elif (rebalance_type==1 and newStrat.index[i].day!=newStrat.index[i-1].day) or (rebalance_type==2 and newStrat.index[i].week!=newStrat.index[i-1].week) or (rebalance_type==3 and newStrat.index[i].month!=newStrat.index[i-1].month) or (rebalance_type==4 and newStrat.index[i].month!=newStrat.index[i-1].month and newStrat.index[i].month%3==0) :
#            month_return=np.array(newStrat.loc[newStrat.index[i-2]]/newStrat.loc[newStrat.index[i-day_return]])
#            ticker_index=np.sort(two_side_filter(month_return,Amount_Top,MomentumType)[1])
#            if (newStrat_price.loc[i-vol_day-day_delay]/newStrat_price.loc[rebalance_index-vol_day-day_delay])[0]<(1-stop_loss):
#                weight_pot=np.zeros(len(newStrat.columns))
#            else:
#                weight_pot=MD_weight_calculation(i,newStrat,ticker_index,vol_day,decay,day_delay)
#            if sum(abs(weight_pot-weight)>(threshold))>0:
#                cost=0
#                #cost=sum(abs(weight-weight_pot)*newStrat.loc[newStrat.index[i-1]]/newStrat.loc[newStrat.index[rebalance_index-1]])*0.0002*newStrat_price.loc[rebalance_index-vol_day-day_delay]/newStrat_price.loc[i-vol_day-day_delay]
#                weight=weight_pot
#                newStrat_price.loc[i-vol_day-day_delay+1]=(1-cost)*newStrat_price.loc[i-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[i-1]]-1)))
#                rebalance_index=i
#                rebalance_time+=1
#            else:
#                newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))               
#            weights_pd.loc[i-vol_day-day_delay+1]=weight
#        else:
#            newStrat_price.loc[i-vol_day-day_delay+1]=newStrat_price.loc[rebalance_index-vol_day-day_delay]*(1+np.dot(weight,(newStrat.loc[newStrat.index[i]]/newStrat.loc[newStrat.index[rebalance_index-1]]-1)))
#            weights_pd.loc[i-vol_day-day_delay+1]=weight
#
#    newStrat_price.index=newStrat.index[vol_day+day_delay-1:]
#    weights_pd.index=newStrat.index[vol_day+day_delay:]
#    Strat=Strategy(newStrat_price.index,newStrat_price['newStrat'])
#    sharpe_temp=Strat.Sharpe()
#    return (sharpe_temp,weights_pd,Strat,newStrat_price)
#