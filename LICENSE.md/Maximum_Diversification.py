# -*- coding: utf-8 -*-
"""
Created on Wed Oct 05 13:01:42 2016

@author: sjin
"""

import pandas as pd
import sqlite3
import datetime
import numpy as np
import os
from os import chdir
import sqlite3
import matplotlib.pyplot as plt

#**********************************
# IMPORT US STRUCTURING LIBRAIRIES
#**********************************
DataBase_path = r'H:\Local_Code\Database'
Prism_path = r'H:\Local_Code\Database\PrismRequest'
os.chdir(r'H:\Local_Code')

import class_CalendarUS
#from DB_functions import *
from class_Strategy import *
from PortfolioAllocation import *


(sharpe,weights_pd,Strat,newStrat_price)=CVaR_Strategy(newStrat1[newStrat1.columns[[0,1,4,10,14]]],"monthly","low",1,40,6,23,0.1,0.01*i)


#######################################################################
newStrat1=excess_return_data(r"H:\Desktop\diver_data48_largeuniverse.csv")

newStrat1=data_reading(r"H:\Desktop\data_multi_asset_without_shortVIX.csv")
newStrat1=data_reading(r"H:\Desktop\data_multi_asset2.csv")
newStrat1=data_reading(r"H:\Desktop\data_multi_asset2_withAGG.csv")
newStrat1=data_reading(r"C:\Users\sjin\Desktop\1018\withoutVIX.csv")

newStrat1=data_reading(r"H:\Desktop\1021\Mutual_fund_data.csv")
newStrat1=data_reading(r"H:\Desktop\1024\Mutual_fund_data2.csv")

newStrat1=data_reading(r"H:\Desktop\1026\Emerging_market_etf.csv")
newStrat1=data_reading(r"H:\Desktop\1026\Emerging_market_4etf.csv")


newStrat1=data_reading(r"H:\Desktop\1104\data_multi_asset2.csv")

stardard_strategy_test(newStrat1,r"H:\Desktop\1101\stardard_strategy_etf5")
stardard_strategy_test(newStrat1,r"H:\Desktop\1101\standard_strategy_etf4")
stardard_strategy_test(newStrat1,r"H:\Desktop\1101\etf4_momentum_comparison")
stardard_strategy_test(newStrat1,r"H:\Desktop\1101\etf5_momentum_comparison")



newStrat1=data_reading(r"H:\Desktop\data_multi_asset2.csv")
stardard_strategy_test(newStrat1,r"H:\Desktop\1103\Standard_test\1")
newStrat1=data_reading(r"H:\Desktop\data_multi_asset_without_shortVIX.csv")
stardard_strategy_test(newStrat1,r"H:\Desktop\1103\Standard_test\2")
newStrat1=data_reading(r"H:\Desktop\1026\Emerging_market_etf.csv")
stardard_strategy_test(newStrat1,r"H:\Desktop\1103\Standard_test\3")
newStrat1=data_reading(r"H:\Desktop\1026\Emerging_market_4etf.csv")
stardard_strategy_test(newStrat1,r"H:\Desktop\1103\Standard_test\4")
newStrat1=data_reading(r"H:\Desktop\1021\Mutual_fund_data.csv")
stardard_strategy_test(newStrat1,r"H:\Desktop\1103\Standard_test\5")
newStrat1=data_reading(r"H:\Desktop\1021\Mutual_fund_data2.csv")
stardard_strategy_test(newStrat1,r"H:\Desktop\1103\Standard_test\6")


newStrat1=data_reading(r"H:\Desktop\1107\Multi_asset.csv")

newStrat1=data_reading(r"H:\Desktop\1107\Multi_asset_SPY.csv")
newStrat1=data_reading(r"H:\Desktop\1107\Multi_asset_SPY_QQQ.csv")

path=r"H:\Desktop\1104\rainbow_4etf_2"
for i in range(1,11):
    Horizon=20
    name='Quarterly'
    m=Rainbow_execute(newStrat1,newStrat1.columns,name,Horizon,[0,i*0.1,1-i*0.1])
    Strat=Strategy(m.index,m['Index'])
    Strat.Save(path+"\Rainbow"+name+str(Horizon)+str(i)+".csv")

    Horizon=120
    name='monthly'
    m=Rainbow_execute(newStrat1,newStrat1.columns,name,Horizon,[0,i*0.1,1-i*0.1])
    Strat=Strategy(m.index,m['Index'])
    Strat.Save(path+"\Rainbow"+name+str(Horizon)+str(i)+".csv")
    
#stardard_strategy_test(newStrat1,path)=(newStrat1,r"H:\Desktop\1101\etf4_momentum_comparison")
#def Sharpe_Momentum_execute(dfPrice,ContractList,Rebalancing_type,iObservation_period)=(newStrat1,newStrat1.columns,name,Horizon)
#def Sharpe_Momentum_Allocation(dfPrice,ContractList,Rebalancing_type,iObservation_period)=(newStrat1,newStrat1.columns,name,Horizon)


for i in newStrat1.columns:  
    pd_etf=pd.DataFrame({i:newStrat1[i]})   
    newStrat1[i]=beta_adjusted(pd_etf)
newStrat1=newStrat1.dropna()    

result=pd.DataFrame(columns=["i","j","m","k","n","sharpe"])
num=0
for i in range(0,3):
    for j in range(9,10):
        for m in range(6,7):
            for k in range(9,10):
                for n in range(8,9):
#    (sharpe,weights_pd,Strat,newStrat_price)=MDS(newStrat1[newStrat1.columns[[0,1,4,10,14]]],"monthly","low",1,40,6,23,0.1,0.92,0.01*i)
#                    (sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy_Return_NoneZero(newStrat1,"weekly","low",1,120,4,120,0.1,0.95,0.02,2+2*i,5*j,m,0.001*k)
#                    (sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy_Return_OnlyPositive(newStrat1,"weekly","low",1,120,4,120,0.1,0.95,0.02,2+2*i,5*j,m,0.001*k)
#                    (sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy_Return_NoneZero_Daily(newStrat1,"monthly","low",1,10*j,1,10*k,0.1,0.95,0.02,2,5*7,2,0.001*0)
                    (sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy_Return_NoneZero_Daily(newStrat1,"monthly","low",1,60,4,10,0.1,0.95,0.01,2,18,2,0.001*0)
                    result.loc[num]=np.array([i,j,m,k,n,sharpe])
                    num=num+1                    
                    
for i in range(12,34):
    for j in range(9,10):
#        ticker_list=[1,2,11]
#        t=list(set(ticker_list)-set([ticker_list[i]]))
        (sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy_Return_NoneZero(newStrat1[newStrat1.columns[[0,1,2,33]]],"weekly","low",1,120,4,120,0.1,0.95,0.02,2,5,1,0.001*0)
        Strat.Save(r'H:\Desktop\1024\More_Mutual_Fund\ETF_'+newStrat1.columns[33]+'.csv')

(sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec(newStrat1,"weekly","low",0,120,4,120,0.1,0.96,0.02,2,5,1,0.001*0)

(sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy_Return_NoneZero(newStrat1,"weekly","low",1,120,4,120,0.1,0.95,0.02,2,5,1,0.001*0)


########
newStrat1=data_reading(r"H:\Desktop\1107\Multi_asset.csv")
(sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec(newStrat1,"weekly","every X",2,120,4,120,0.1,0.95,0.02,2,5,0,0.001*0) #1.19
print(sharpe)
(sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec_sharpe_momentum(newStrat1,"weekly","every X",2,120,4,120,0.1,0.95,0.02,2,5,0,0.001*0)  #1.05
print(sharpe)
########
newStrat1=data_reading(r"H:\Desktop\1107\Multi_asset.csv")
(sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec(newStrat1,"weekly","every XX",2,120,4,120,0.1,0.95,0.02,2,5,0,0.001*0) #1.12
print(sharpe)
(sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec_sharpe_momentum(newStrat1,"weekly","every XX",2,120,4,120,0.1,0.95,0.02,2,5,0,0.001*0)  #0.74
print(sharpe)

########
newStrat1=data_reading(r"H:\Desktop\1107\Multi_asset_SPY_QQQ.csv")
(sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec(newStrat1,"weekly","first X",2,120,4,120,0.1,0.95,0.02,2,5,0,0.001*0) #1.36
print(sharpe)
(sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec_sharpe_momentum(newStrat1,"weekly","first X",2,120,4,20,0.1,0.95,0.02,2,5,0,0.001*0)  #1.38
print(sharpe)
########
newStrat1=data_reading(r"H:\Desktop\1107\Multi_asset_SPY_QQQ.csv")
(sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec(newStrat1,"weekly","first XX",2,120,4,120,0.1,0.95,0.02,2,5,0,0.001*0) #1.54
print(sharpe)
(sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec_sharpe_momentum(newStrat1,"weekly","first XX",2,120,4,20,0.1,0.95,0.02,2,5,0,0.001*0)  #1.08
print(sharpe)

########
newStrat1=data_reading(r"H:\Desktop\data_multi_asset_without_shortVIX.csv")
(sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec(newStrat1,"weekly","low",0,120,4,120,0.1,0.95,0.02,2,5,0,0.001*0) #1.56
print(sharpe)
(sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec_sharpe_momentum(newStrat1,"weekly","low",0,120,4,20,0.1,0.95,0.02,2,5,0,0.001*0)  #1.56
print(sharpe)
########
newStrat1=data_reading(r"H:\Desktop\data_multi_asset_without_shortVIX.csv")
(sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec(newStrat1,"weekly","low",1,120,4,120,0.1,0.95,0.02,2,5,0,0.001*0) #1.73
print(sharpe)
(sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec_sharpe_momentum(newStrat1,"weekly","low",1,120,4,20,0.1,0.95,0.02,2,5,0,0.001*0)  #1.27
print(sharpe)

########
newStrat1=data_reading(r"H:\Desktop\1107\Multi_asset_SPY_QQQ.csv")
(sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec_Individual_momentum(newStrat1,"weekly","low",1,120,4,120,0.1,0.95,0.02,2,5,0,0.001*0,'monthly',50)
dfStrategy=Momentum_Allocation(newStrat1,newStrat1.columns[0:2],'monthly',50)
print(sharpe)

(sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec_Individual_momentum(newStrat1,"weekly","low",1,120,4,120,0.1,0.95,0.02,2,5,0,0.001*0,'quarterly',50)
dfStrategy=Momentum_Allocation(newStrat1,newStrat1.columns[0:2],'quarterly',50)
print(sharpe)

(sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec_Individual_momentum(newStrat1,"weekly","low",1,120,4,120,0.1,0.95,0.02,2,5,0,0.001*0,'weekly',50)
dfStrategy=Momentum_Allocation(newStrat1,newStrat1.columns[0:2],'weekly',50)
print(sharpe)
###########
(sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec_sharpe_momentum(newStrat1,"weekly","first X",2,120,4,20,0.1,0.95,0.02,2,5,0,0.001*0)

(newStrat,Rebalancing_type,MomentumType,Amount_Top,vol_day,day_delay,day_return,threshold,decay,stop_loss,none_zero_day1,none_zero_day2,none_zero_number,none_zero_threshold)=(newStrat1,"weekly","every X",2,20,4,20,0.1,0.95,0.02,2,5,0,0.001*0)
(newStrat,Rebalancing_type,MomentumType,Amount_Top,vol_day,day_delay,day_return,threshold,decay,stop_loss,none_zero_day1,none_zero_day2,none_zero_number,none_zero_threshold)=(newStrat1,"weekly","first X",2,120,4,20,0.1,0.95,0.02,2,5,0,0.001*0)

(newStrat,Rebalancing_type,MomentumType,Amount_Top,vol_day,day_delay,day_return,threshold,decay,stop_loss,none_zero_day1,none_zero_day2,none_zero_number,none_zero_threshold,Rebalancing_type2,iObservation_period)=(newStrat1,"weekly","low",1,120,4,120,0.1,0.95,0.02,2,5,0,0.001*0,'monthly',50)

(sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy_Return_NoneZero_Daily(newStrat1,"monthly","low",1,50,2,90,0.1,0.95,0.02,2,5*7,2,0.001*0)

(sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy_Return_NoneZero_Daily(newStrat1,"monthly","low",1,30,4,30,0.1,0.95,0.01,2,18,2,0.001*0)

(sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy_Return_NoneZero_Daily(newStrat1,"monthly","low",1,35,4,35,0.1,0.95,0.02,2,18,1,0.001*0)

(sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy_Return_NoneZero_Daily(newStrat1,"monthly","low",0,30,4,30,0.1,0.95,0.99,2,18,2,0.001*99)
newStrat_price.plot()
print(sharpe)


(sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy_Sharpe_Momentum(newStrat1,"weekly","low",1,120,4,120,0.1,0.96,0.02)

for i in range(1,11):
    (sharpe,weights_pd,Strat,newStrat_price)=ERC_Execute_Return_NoneZero(newStrat1,"monthly","low",2,10*i,3,10*i,0.1,0.95,0.99,2,5,1,0.01)
    newStrat_price.plot()
    print(sharpe)


Strat.Save(r"H:\Desktop\1104\withoutVIX.csv")



dictionary1={
            0:[newStrat1],
1:["monthly","weekly","quarterly"],
2:["low"],
3:np.arange(1,3,1),
4:np.arange(10,110,10),
5:np.arange(2,6,1),
6:np.arange(10,110,10),
7:np.arange(0.025,0.1,0.025),
8:np.arange(0.94,0.99,0.01),
9:np.arange(0.01,0.03,0.01),
10:np.arange(2,10),
11:np.arange(5,50,5),
12:np.arange(1,3),
13:np.arange(0.001,0.003,0.001)
            }
            
########################################################################################################################################################## 
dictionary2={
            0:[newStrat1],
1:["monthly","weekly","quarterly","daily"],
2:["every X"],
3:np.arange(2,3,1),
4:np.arange(20,250,10),
5:np.arange(2,30,2),
6:np.arange(20,40,1),
7:np.arange(0.00,0.1,0.01),
8:np.arange(0.91,0.99,0.01),
9:np.arange(0.01,0.04,0.01),
10:np.arange(2,10),
11:np.arange(5,150,5),
12:np.arange(0,3),
13:np.arange(0.000,0.004,0.001),
14:np.arange(2,16,2),
15:np.arange(10,20,10)
            }
            
            
            
par=Parameter_Optimization(Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec_Individual_momentum_everyX__stoploss_everyday,(newStrat1,"monthly","every X",2,240,16,39,0.09,0.98,0.03,9,90,1,0.001*0,1,20),dictionary2,16)
par_list=pd.DataFrame(columns=['para'])
for i in range(0,14):
    par_list.loc[i]=(par[1+i])
par_list.to_csv(r"H:\Desktop\1111\optimization_result.csv")
#(Function_name,standard,dictionary,variable_number)=(Maximum_Diverisification_Strategy_Return_NoneZero,(newStrat1,"weekly","low",1,120,4,120,0.1,0.95,0.02,2,5,1,0.001*0),dictionary,14)

(sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec_Individual_momentum_everyX__stoploss_everyday_underindex(newStrat1,"monthly","every X",2,240,16,40,0.10,0.98,0.03,9,90,1,0.001*0,1,20) #sharpe 1.6825
print(sharpe)

(sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec_Individual_momentum_everyX__stoploss_everyday_underindex(newStrat1,"monthly","every X",2,240,16,40,0.10,0.98,0.03,16,40,1,0.001*0,1,20) #sharpe 1.6491
print(sharpe)

(sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec_Individual_momentum_everyX__stoploss_everyday(newStrat1,"monthly","every X",2,240,16,40,0.10,0.98,0.03,16,40,1,0.001*0,1,20) #sharpe 1.6478
print(sharpe)

(sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec_Individual_momentum_everyX__stoploss_everyday(newStrat1,"monthly","every X",2,240,15,40,0.10,0.98,0.03,15,40,1,0.001*0,1,20) #sharpe 1.5381
print(sharpe)

(sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec_Individual_momentum_everyX__stoploss_everyday(newStrat1,"monthly","every X",2,240,16,40,0.10,0.98,0.03,9,90,1,0.001*0,1,20)#sharpe 1.6811
print(sharpe)

(sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec_Individual_momentum_everyX__stoploss_everyday(newStrat1,"monthly","every X",2,240,16,40,0.10,0.98,0.03,16,56,1,0.001*0,1,20) #sharpe 1.7069
print(sharpe)
Strat.Save(r'H:\Desktop\1114\withTIPS_result.csv')


(sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec_Individual_momentum_everyX__stoploss_everyday(newStrat1,"monthly","every X",2,250,2,40,0.10,0.98,0.03,2,42,1,0.001*0,1,20) #sharpe 1.30
print(sharpe)

(sharpe,weights_pd,Strat,newStrat_price)=Performance_Maximum_Diverisification_Strategy_Return_NoneZero_everyX_stoploss_everyday(newStrat1,"monthly","every X",2,60,5,40,0.10,0.98,0.03,16,56,1,0.001*0,1,20) #sharpe 1.30
print(sharpe)


dictionary2={
            0:[newStrat1],
1:["monthly","weekly","quarterly","daily"],
2:["every XX"],
3:np.arange(2,3,1),
4:np.arange(20,250,10),
5:np.arange(2,30,2),
6:np.arange(20,40,1),
7:np.arange(0.00,0.1,0.01),
8:np.arange(0.91,0.99,0.01),
9:np.arange(0.01,0.04,0.01),
10:np.arange(2,10),
11:np.arange(5,150,5),
12:np.arange(0,3),
13:np.arange(0.000,0.004,0.001),
14:np.arange(2,16,2),
15:np.arange(10,20,10)
            }
            

par2=Parameter_Optimization(Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec_Individual_momentum_everyX,(newStrat1,"monthly","every XX",2,190,10,38,0.09,0.98,0.03,2,145,1,0.001*0,1,20),dictionary2,16)
#(Function_name,standard,dictionary,variable_number)=(Maximum_Diverisification_Strategy_Return_NoneZero,(newStrat1,"weekly","low",1,120,4,120,0.1,0.95,0.02,2,5,1,0.001*0),dictionary,14)

dictionary2={
            0:[newStrat1],
1:["monthly","weekly","quarterly","daily"],
2:["every X"],
3:np.arange(2,3,1),
4:np.arange(20,250,10),
5:np.arange(2,30,2),
6:np.arange(20,40,1),
7:np.arange(0.00,0.1,0.01),
8:np.arange(0.91,0.99,0.01),
9:np.arange(0.01,0.04,0.01),
10:np.arange(2,10),
11:np.arange(5,150,5),
12:np.arange(0,3),
13:np.arange(0.000,0.004,0.001),
14:np.arange(2,16,2),
15:np.arange(10,20,10)
            }
            

par3=Parameter_Optimization(Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec_Individual_momentum_everyX_sharpe,(newStrat1,"weekly","every XX",2,120,4,120,0.1,0.95,0.02,2,5,1,0.001*0,1,20),dictionary2,16)
#(Function_name,standard,dictionary,variable_number)=(Maximum_Diverisification_Strategy_Return_NoneZero,(newStrat1,"weekly","low",1,120,4,120,0.1,0.95,0.02,2,5,1,0.001*0),dictionary,14)

dictionary2={
            0:[newStrat1],
1:["monthly","weekly","quarterly","daily"],
2:["every XX"],
3:np.arange(2,3,1),
4:np.arange(20,250,10),
5:np.arange(2,30,2),
6:np.arange(20,40,1),
7:np.arange(0.00,0.1,0.01),
8:np.arange(0.91,0.99,0.01),
9:np.arange(0.01,0.04,0.01),
10:np.arange(2,10),
11:np.arange(5,150,5),
12:np.arange(0,3),
13:np.arange(0.000,0.004,0.001),
14:np.arange(2,16,2),
15:np.arange(10,20,10)
            }
            

par4=Parameter_Optimization(Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec_Individual_momentum_everyX_sharpe,(newStrat1,"weekly","every XX",2,120,4,120,0.1,0.95,0.02,2,5,1,0.001*0,1,20),dictionary2,16)
#(Function_name,standard,dictionary,variable_number)=(Maximum_Diverisification_Strategy_Return_NoneZero,(newStrat1,"weekly","low",1,120,4,120,0.1,0.95,0.02,2,5,1,0.001*0),dictionary,14)

###################################################################################################################################################################################################
(sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec_Individual_momentum_everyX__stoploss_everyday(newStrat1,"monthly","every XX",2,200,10,40,0.09,0.98,0.03,2,40,0,0.001*0,1,20)
print(sharpe)
Strat.Save(r'H:\Desktop\1110\10ETF\4outof10_monthly_int_nonzero.csv')

(sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec_Individual_momentum_everyX__stoploss_everyday_debug(newStrat1,"monthly","every X",2,200,10,40,0.09,0.98,0.99,2,40,1,0.001*0,1,20)
print(sharpe)
Strat.Save(r'H:\Desktop\1110\10ETF\5outof10_monthly_int_nonzero.csv')

(newStrat,Rebalancing_type,MomentumType,Amount_Top,vol_day,day_delay,day_return,threshold,decay,stop_loss,none_zero_day1,none_zero_day2,none_zero_number,none_zero_threshold,Rebalancing_type2,iObservation_period)=(newStrat1,"monthly","every X",2,200,10,40,0.09,0.98,0.99,2,40,1,0.001*0,1,20)

(sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec_Individual_momentum_everyX__stoploss_everyday_underindex(newStrat1,"monthly","every X",2,200,10,40,0.09,0.98,0.01,2,40,1,0.001*0,1,20)
print(sharpe)
Strat.Save(r'H:\Desktop\1110\10ETF\continuous.csv')


(sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec_Individual_momentum_everyX__stoploss_everyday(newStrat1,"monthly","every X",2,200,10,40,0.09,0.98,0.01,2,40,1,0.001*0,1,20)
print(sharpe)
Strat.Save(r'H:\Desktop\1110\10ETF\5out10_nonzero_0.01Stoploss.csv')

(sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec_Individual_momentum_everyX(newStrat1,"monthly","every XX",2,200,10,40,0.09,0.98,0.03,2,40,0,0.001*0,0,20)
print(sharpe)
Strat.Save(r'H:\Desktop\1110\10ETF\4outof10_monthly_int.csv')

(sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec_Individual_momentum_everyX(newStrat1,"monthly","every X",2,200,10,40,0.09,0.98,0.03,2,40,0,0.001*0,0,20)
print(sharpe)
Strat.Save(r'H:\Desktop\1110\10ETF\5outof10_monthly_int.csv')


(sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec_Individual_momentum_everyX(newStrat1,"quarterly","every XX",2,190,10,38,0.09,0.98,0.03,2,145,1,0.001*0,1,20)
print(sharpe)
Strat.Save(r'H:\Desktop\1110\10ETF\4outof10_quarterly.csv')

(sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec_Individual_momentum_everyX(newStrat1,"quarterly","every X",2,190,10,38,0.09,0.98,0.03,2,145,1,0.001*0,1,20)
print(sharpe)
Strat.Save(r'H:\Desktop\1110\10ETF\5outof10_quarterly.csv')

(sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec_Individual_momentum_everyX(newStrat1,"quarterly","every X",2,190,10,38,0.09,0.98,0.03,2,145,1,0.001*0,4,20)
print(sharpe)
Strat.Save(r'H:\Desktop\1110\10ETF\5outof10_quarterly_yearly.csv')

(sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec_Individual_momentum(newStrat1,"weekly",'low', 1, 120, 4, 120, 0.1, 0.96, 0.02, 2, 5, 1, 0.000, 'weekly', 120)
print(sharpe)

(sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy_Return_NoneZero_Filter_Mec_Individual_momentum_everyX(newStrat1,"weekly",'every XX', 2, 120, 4, 120, 0.1, 0.95, 0.02, 2, 5, 1, 0.000, 1, 120)
print(sharpe)



('low', 1, 120, 4, 120, 0.1, 0.95999999999999996, 0.02, 2, 5, 1, 0.001, 'quarterly', )

for i in range(12,34):
    for j in range(9,10):
#        ticker_list=[1,2,11]
#        t=list(set(ticker_list)-set([ticker_list[i]]))
        i=7
        Strat=Strategy(newStrat1.index,newStrat1[newStrat1.columns[[0,1,3,7]]])
        Strat.Save(r'H:\Desktop\1024\Individual_ETF\ETF_'+newStrat1.columns[i]+'.csv')
        
(sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy_Return_NoneZero(newStrat1[newStrat1.columns[[0,1,3,7]]],"weekly","low",1,120,4,120,0.1,0.95,0.02,2,5,1,0.001*0)
#Maximum_Diverisification_Strategy_Return_NoneZero(newStrat,Rebalancing_type,MomentumType,Amount_Top,vol_day,day_delay,day_return,threshold,decay,stop_loss,none_zero_day1,none_zero_day2,none_zero_number,none_zero_threshold)                
#    Strat.Describe()
excess_return=beta_adjusted(newStrat_price)
Strat=Strategy(excess_return.index,excess_return)
(sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy(newStrat1,"weekly","low",1,120,4,120,0.1,0.95,0.02)
(sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy_Return_NoneZero(newStrat1,"weekly","low",1,120,4,120,0.1,0.95,0.02,2,5,1,0.001*999)

Strat.Save(r'H:\Desktop\1024\Individual_ETF\MD_fixedIncome.csv')
for i in range(1,8):
    for j in range(1,8):
        for m in range(2,3):
            for k in range(0,1):
                for n in range(0,1):
#                    (sharpe,weights_pd,Strat,newStrat_price)=CVaR_Strategy(newStrat1,"weekly","low",j,40,i,23,0.025*k,0.02*m,0.001*10**n)
                    (sharpe,weights_pd,Strat,newStrat_price)=CVaR_Strategy(newStrat1,"weekly","low",4,38+i,0,18+j,0.025*2,0.01,0.001)
                    result.loc[num]=np.array([i,j,m,k,n,sharpe])
                    print(sharpe)
                    num=num+1

i=4
j=1
m=1
k=1
n=0
(sharpe,weights_pd,Strat,newStrat_price)=CVaR_Strategy(newStrat1,"weekly","low",j,40,i,23,0.025*k,0.02*m,0.001*10**n,2,5,1,0.000)
newStrat_price.plot()

#CVaR_Strategy(newStrat,Rebalancing_type,MomentumType,Amount_Top,vol_day,day_delay,day_return,threshold,stop_loss,target)
for i in range(2,5):
    for j in range(0,2):
        for m in range(0,3):
            for k in range(1,3):
                for n in range(1,3):
#                    (sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy(newStrat1,"Weekly","low",j,120,i,120,0.025*m,0.99-0.01*k,0.01*n)
#                    (sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy(newStrat1,"weekly","low",1,120,4,120,0.1,0.97,0.02)
                    (sharpe,weights_pd,Strat,newStrat_price)=Vol_Over_Sharpe_Strategy(newStrat1,"weekly","low",j,120,0,120,0.05*m,0.02*k,0.00001,0.92+n*0.02)
                    result.loc[num]=np.array([i,j,m,k,n,sharpe])
                    num=num+1

(sharpe,weights_pd,Strat,newStrat_price)=Maximum_Diverisification_Strategy(newStrat1,"Weekly","low",1,35,3,120,0.05,0.99,0.01)
(sharpe,weights_pd,Strat,newStrat_price)=Vol_Over_Sharpe_Strategy(newStrat1,"weekly","low",0,120,2,120,0.0,0.02,0.00001,0.94)
(sharpe,weights_pd,Strat,newStrat_price)=Vol_Over_Sharpe_Strategy(newStrat1,"monthly","low",1,40,6,23,0.1,0.02,0.00001,0.94)
#result.loc[num]=np.array([i,j,k,m,n,a,sharpe])
#num=num+1
#CVaR_Strategy(newStrat,Rebalancing_type,MomentumType,Amount_Top,vol_day,day_delay,day_return,threshold,stop_loss,target)
a=Equal_Exposure_monthly(newStrat1,newStrat1.columns)
Strat=Strategy(a.index,a['Index'])
Strat.Save(r"H:\Desktop\1027\ETF5_output_EW.csv")


for i in range(0,4):
    (sharpe,weights_pd,Strat,newStrat_price)=Sharpe_Strategy(newStrat1,"weekly","low",1,40,6,23,0.1,0.01*i,0.1)
    Strat.Describe()

for i in range(0,4):
    (sharpe,weights_pd,Strat,newStrat_price)=Vol_Over_Sharpe_Strategy(newStrat1,"weekly","low",1,40,6,23,0.1,0.01*i,0.1,0.92)
    Strat.Describe()


Strat.Save(r"H:\Desktop\1021\Mutual_Fund_MD_Output_3ETF.csv")

for i in [7,20,120,250]:
    Horizon=i
    name='Weekly'
    m=Rainbow_execute(newStrat1,newStrat1.columns,name,Horizon,[0.65,0.35])
    Strat=Strategy(m.index,m['Index'])
    Strat.Save(path+"\Rainbow7525"+name+str(Horizon)+".csv")
    
    name='Monthly'
    m=Rainbow_execute(newStrat1,newStrat1.columns,name,Horizon,[0.65,0.35])
    Strat=Strategy(m.index,m['Index'])
    Strat.Save(path+"\Rainbow7525"+name+str(Horizon)+".csv")
    
    name='Quarterly'
    m=Rainbow_execute(newStrat1,newStrat1.columns,name,Horizon,[0.65,0.35])
    Strat=Strategy(m.index,m['Index'])
    Strat.Save(path+"\Rainbow7525"+name+str(Horizon)+".csv")
    
    name='Daily'
    m=Rainbow_execute(newStrat1,newStrat1.columns,name,Horizon,[0.65,0.35])
    Strat=Strategy(m.index,m['Index'])
    Strat.Save(path+"\Rainbow7525"+name+str(Horizon)+".csv")