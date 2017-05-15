# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 10:59:49 2015

@author: tzercher

This class enables to analyse a strategy.
It has several functions to compute Sharpe Ratio, Volatility...

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
from os import chdir
import os
import sqlite3

#**********************************
# IMPORT NYC ENGINEERING LIBRAIRIES
#**********************************
os.chdir(r'H:\Local_Code')
from class_CalendarUS import * 
from NTX_Stats import *
from DB_functions import *

class Strategy :
    
    """ special methods """
    def __repr__(self):
        """ method that displays the strategy index """
        print(self.newStrat)
    def __init__(self,newStrat_dates,newStrat_strat):
        newStrat=pd.DataFrame({'Dates':newStrat_dates,'strat':newStrat_strat})
        newStrat.set_index('Dates',inplace = True)
        self.newStrat=newStrat    
        
    """ class procedures""" 
    
    """Statistical ratios """
    def  IRR(self):
        """ returns the IRR """
        newStrat=self.newStrat
        dfin=newStrat.index[len(newStrat)-1]
        ddebut= newStrat.index[0]
        days = (dfin-ddebut).days
        if (newStrat.loc[newStrat.index[len(newStrat)-1],'strat']/newStrat.loc[newStrat.index[0],'strat'])>0:
            Irr= (newStrat.loc[newStrat.index[len(newStrat)-1],'strat']/newStrat.loc[newStrat.index[0],'strat'])**(365.0/days)-1
            return Irr
        else:
            return 0.0
        
        
    def  Daily_Vol(self):
        """ returns the volatility  """
        newStrat=self.newStrat
        vol=np.log(newStrat['strat'].pct_change()+1).std()
        return vol
        
    def  Annualized_Vol(self):
        """ returns the volatility  """
        newStrat=self.newStrat
        vol=np.log(newStrat['strat'].pct_change()+1).std()*np.sqrt(252)
        return vol
        
    def  Annualized_Vol_Down(self):
        """ returns the volatility of downward moves"""
        newStrat=self.newStrat
        vol=np.log(newStrat['strat'].pct_change()+1).map(lambda x: 0 if x >0 else x)
        vol=vol.std()*np.sqrt(252)
        return vol
    
    def Sharpe(self):
        """ returns the Sharpe Ratio  """
        vol= self.Annualized_Vol()
        Irr=self.IRR()
        return Irr/vol
    
    def Sharpe_Cleared(self):
        """ returns the Sharpe Ratio excluding the upward volatility """
        vol= self.Annualized_Vol_Down()
        Irr=self.IRR()
        return Irr/vol
    
    def Max_DrawDown(self):
        """ returns the Max DD  """
        newStrat=self.newStrat
        MaxDD=1.0
        for k in range(0,len(newStrat)-1):
            dMax=max(newStrat['strat'][0:k+1])
            dMin = min(newStrat['strat'][k+1:len(newStrat)+1])
            if dMin/dMax<MaxDD:
                MaxDD=dMin/dMax
        return MaxDD-1
        
    def Annual_Perf(self):
        newStrat = self.newStrat
        newStrat['Year']=newStrat.index.map(lambda x : x.year)
        years=newStrat['Year'].drop_duplicates().tolist()
        perf=[]
        for i in range(0,len(years)):
            temp = newStrat[newStrat['Year']==years[i]]
            begin= temp.index[0]
            position = (newStrat.index.tolist()).index(begin)
            if i>0:
                begin=newStrat.index[position-1]
            end= temp.index[len(temp)-1]
            perf.append(newStrat['strat'][end]/newStrat['strat'][begin]-1)
        del newStrat['Year']
        dfAnnualPerf= pd.DataFrame({'Year':years,'Perf':perf}) 
        dfAnnualPerf=dfAnnualPerf.sort_index(axis=1,ascending = False)
        return dfAnnualPerf 

    def Monthly_Perf(self):
        dfAnnualPerf=self.Annual_Perf()
        dfAnnualPerf.set_index('Year',inplace=True)
        newStrat = self.newStrat
        calendar= Calendar(newStrat.index[0],newStrat.index[len(newStrat)-1])
        newStrat['Year']=newStrat.index.map(lambda x : x.year)
        newStrat['Month']=newStrat.index.map(lambda x : x.month)
        
        years=sorted(newStrat['Year'].drop_duplicates().tolist(),reverse=True)
        performance= pd.DataFrame({'Year':years})
        month = ['January','February','March','April','May','June','July','August','September','October','November','December']
        monthEquiv={}
        for k in range(1,13):
            monthEquiv[k,'Month']=month[k-1]
            
        
        for months in month:
            performance[months]=np.nan
            
        performance.set_index('Year',inplace=True)
        performance['YTD']=0.0
        firstDate = newStrat.index[0]
        lastDate= newStrat.index[len(newStrat)-1]

        bfirstdone=False
        for year in sorted(years,reverse=False):  
            performance.loc[year,'YTD']=dfAnnualPerf.loc[year,'Perf']
            for month in range(1,13):
                if  bfirstdone==False: #first year
                    if firstDate.month == month:
      
                        MonthLastday=calendar.MonthLastBusinessDay(datetime.datetime(year,month,1))
                        endPeriodPerf= newStrat[newStrat.index<=MonthLastday].tail(1).index[0]
                        lastendperiod= endPeriodPerf
                        perf = newStrat.loc[endPeriodPerf,'strat']/newStrat.loc[firstDate,'strat']-1
                        performance.loc[year,monthEquiv[month,'Month']]=perf
                        bfirstdone=True
                    else: # no perf
                        performance.loc[year,monthEquiv[month,'Month']]=""      
                else:
                    MonthLastday=calendar.MonthLastBusinessDay(datetime.datetime(year,month,1))
                    if year == lastDate.year and lastDate.month<month:
                        performance.loc[year,monthEquiv[month,'Month']]=""
                    else:
                        endPeriodPerf= newStrat[newStrat.index<=MonthLastday].tail(1).index[0]
                        perf = newStrat.loc[endPeriodPerf,'strat']/newStrat.loc[lastendperiod,'strat']-1
                        lastendperiod=endPeriodPerf
                        performance.loc[year,monthEquiv[month,'Month']]=perf
        return performance

    def Weekly_perf(self,weekday):
        newStrat= self.newStrat
        """
        search for the first evaluation date
        """
        bWeekdayFound =False
        icount = 0
        while bWeekdayFound ==False:
            if calendar.isBusinessDay(newStrat.index[icount]):
                day = newStrat.index[icount].weekday()
                if day==weekday:
                    bWeekdayFound=True
                else:
                    icount+=1
            else:
                icount+=1
        Period_begin = newStrat.index[icount]
        newStrat=newStrat[newStrat.index>=newStrat.index[icount]]
       
        calculus_date = []
        perf = []

        for k in range(1,len(newStrat)):
            if newStrat.index[k].weekday()==day :
                perf.append(newStrat.loc[newStrat.index[k],'strat']/newStrat.loc[Period_begin,'strat']-1)
                calculus_date.append(newStrat.index[k])
                Period_begin=newStrat.index[k]
            else:
                if newStrat.index[k]>=Period_begin+datetime.timedelta(+7):
                    Period_begin=newStrat.index[k]+datetime.timedelta(-1)
            dfperf= pd.DataFrame({'Date':calculus_date,'Perf':perf})
            dfperf.set_index('Date',inplace=True)
            
        return dfperf

    """ Display Function"""
        
    def Describe(self):
        IRR=self.IRR()
        Vol=self.Annualized_Vol()
        Sharpe=self.Sharpe()
        Sharpe_Cleared=self.Sharpe_Cleared()
        MaxDD=self.Max_DrawDown()
        Daily_Std=self.Daily_Vol()
        dfAnnualPerf=self.Annual_Perf()
        self.newStrat['strat'].plot()
        chaine = 'IRR de  '+str(np.round(100*IRR,4))+"%" +"\n" +"Volatility  "
        chaine+= str(np.round(100*Vol,4))+"%"+ "\n"
        chaine += "Daily Volatility  "+str(np.round(100* Daily_Std,4))+"%"+ "\n"
        chaine+="Sharpe Ratio  " +str(np.round(Sharpe,4))+"\n"
        chaine+="Sortino  " +str(np.round(Sharpe_Cleared,4))+"\n"
        chaine+="Max Drawdown  " + str(np.round(100*MaxDD,4))+"%"+"\n"
        print(chaine)
        print("Annual Performance:")
        print(dfAnnualPerf)
    
    def Save(self,path):
        newStrat = self.newStrat
        irr=self.IRR()
        """ stats"""
        vol=self.Annualized_Vol()
        sharpe=self.Sharpe()
        sortino=self.Sharpe_Cleared()
        maxDD=self.Max_DrawDown()
        
        """ monthly perf"""
        dfmonthly_perf = self.Monthly_Perf()
        """ reindex """
        newStrat['Dates_saved']=newStrat.index
        new_index = range(0,len(newStrat))
        newStrat.reset_index(new_index, inplace = True)
        new_index = range(0,len(dfmonthly_perf ))
        dfmonthly_perf.reset_index(new_index, inplace = True)
        
        """ add statistics """
        newStrat["Blanck"]=""
        newStrat["Statistics"]=""
        newStrat.loc[1,"Statistics"]="IRR"
        newStrat.loc[2,"Statistics"]="Volatility"
        newStrat.loc[3,"Statistics"]="Sharpe"
        newStrat.loc[4,"Statistics"]="Sortino"
        newStrat.loc[5,"Statistics"]="Max Drawdown"
        newStrat["Values"]=""
        
        newStrat.loc[1,"Values"]=irr
        newStrat.loc[2,"Values"]=vol
        newStrat.loc[3,"Values"]=sharpe
        newStrat.loc[4,"Values"]=sortino
        newStrat.loc[5,"Values"]=maxDD
        
        """ add Monthly perf """        
        newStrat["Blanck2"]=""
                
        frames = [newStrat,dfmonthly_perf]
        
        newStrat=pd.concat(frames,axis = 1)
        
        newStrat.set_index('Dates_saved',inplace = True)        
        newStrat.to_csv(path)
        
        print("Saved.")
    
    """ Index transforms """    
    def Leverage(self,Leverage):
        """ returns the leveraged index  """
        newStrat = self.newStrat
        newStrat['strat']=Leverage*(newStrat['strat'].pct_change())
        newStrat.loc[newStrat.index[0],'strat']=0.0
        newStrat['strat']=newStrat['strat']+1
        newStrat['strat']=100*(newStrat['strat'].cumprod())
        return newStrat
    
    def FeesIndex(self,dFees):
        """ returns the index including fees """
        No_fees_index=self.newStrat
        fees_Index=[]
        fees_Index.append(100)
        perf_Index=[]
        perf_Index.append(0)
        fees_index=[]
        fees_index.append(0)
        for i in range(1,len(No_fees_index)):
            fees = dFees *(No_fees_index.index[i]-No_fees_index.index[i-1]).days/365
            perf_Index=No_fees_index['strat'][i]/No_fees_index['strat'][i-1]
            fees_Index.append(fees_Index[i-1]*(perf_Index-fees))
        fees_Index=pd.DataFrame({'Dates':No_fees_index.index,'strat':fees_Index})   
        fees_Index.set_index('Dates',inplace=True)
        return fees_Index
        
    def WithoutFeesIndex(self,dFees):
        """ returns the index including fees """
        No_fees_index=self.newStrat
        fees_Index=[]
        fees_Index.append(100)
        perf_Index=[]
        perf_Index.append(0)
        fees_index=[]
        fees_index.append(0)
        for i in range(1,len(No_fees_index)):
            fees = dFees *(No_fees_index.index[i]-No_fees_index.index[i-1]).days/365
            perf_Index=No_fees_index['strat'][i]/No_fees_index['strat'][i-1]
            fees_Index.append(fees_Index[i-1]*(perf_Index+fees))
        fees_Index=pd.DataFrame({'Dates':No_fees_index.index,'strat':fees_Index})   
        fees_Index.set_index('Dates',inplace=True)
        return fees_Index
        
    def FeesIndex_RiskPremia(self,dFees,calendar):
        """ returns the index including fees """
        No_fees_index=self.newStrat
        fees_Index=[]
        fees_Index.append(100)
        perf_Index=[]
        perf_Index.append(0)
        fees_index=[]
        fees_index.append(0)
        for i in range(1,len(No_fees_index)):
            days = 1#calendar.nbBusinessDaysBetweenTwoDates(No_fees_index.index[i-1],No_fees_index.index[i])
            fees = dFees *days/252
            perf_Index=No_fees_index['strat'][i]/No_fees_index['strat'][i-1]
            fees_Index.append(fees_Index[i-1]*(perf_Index-fees))
        fees_Index=pd.DataFrame({'Dates':No_fees_index.index,'strat':fees_Index})   
        fees_Index.set_index('Dates',inplace=True)
        return fees_Index
    
    def WithoutFeesIndex_RiskPremia(self,dFees,calendar):
        """ returns the index including fees """
        No_fees_index=self.newStrat
        fees_Index=[]
        fees_Index.append(No_fees_index['strat'][0])
        perf_Index=[]
        perf_Index.append(0)
        fees_index=[]
        fees_index.append(0)
        for i in range(1,len(No_fees_index)):
            days = 1#calendar.nbBusinessDaysBetweenTwoDates_exact(No_fees_index.index[i-1],No_fees_index.index[i])
            fees = dFees *days/252
            perf_Index=No_fees_index['strat'][i]/No_fees_index['strat'][i-1]
            fees_Index.append(fees_Index[i-1]*(perf_Index+fees))
            
        fees_Index=pd.DataFrame({'Dates':No_fees_index.index,'strat':fees_Index})   
        fees_Index.set_index('Dates',inplace=True)
        return fees_Index
        
        
    def RebasedIndex(self,datetime_BeginDate,Initial_Value):
        """ returns the index rebased on Begin Date """
        newStrat = self.newStrat
        newStrat= newStrat[newStrat.index>=datetime_BeginDate]
        newStrat['strat']=(newStrat['strat'].pct_change()).fillna(0)+1
        newStrat['strat']=Initial_Value*(newStrat['strat'].cumprod())
        return newStrat
        
    def VolTargetIndex(self,VolTargetLevel,WindowShort,WindowLong, ExpoCap,RatesTicker):
        newStrat = self.newStrat
                
        #********************************************
        # OPEN DATABASE, LOAD PRICES,CLOSE DB
        #********************************************

        """see Calendar Specification above"""
        
        calendar=Calendar(datetime.datetime(2000,1,1),newStrat.index[len(newStrat)-1])
        
        dtBegin=newStrat.index[0]
        dtAdd = calendar.addBusinessDays(-1,dtBegin)
        calendar = newStrat.index.tolist()
        
        calendar.insert(0,dtAdd)
        
        dftemp= Asset_Price_getPrices([RatesTicker],['Last_Price'],dtBegin,dtEnd,True,calendar)
        dftemp=dftemp.fillna(method='ffill')
        newStrat['Rates']=dftemp['Last_Price.'+RatesTicker]
        newStrat=newStrat.fillna(method='ffill')
        newStrat=newStrat.dropna()
 
        
        """ returns the index based on a vol target allocation """
        #compute the volatilities
        
        newStrat['VolWindow1']=ntx_rolling_std(newStrat,'strat',WindowShort,True)  
        newStrat['VolWindow2']=ntx_rolling_std(newStrat,'strat',WindowLong,True) 
        

        newStrat=newStrat.dropna()
        newStrat['Vol']=0.0
        #take the max
        for k in range(0,len(newStrat)):
            newStrat.loc[newStrat.index[k],'Vol']=VolTargetLevel/max(newStrat.loc[newStrat.index[k],'VolWindow2'],newStrat.loc[newStrat.index[k],'VolWindow1'])
        #divide by the target, cap the leverage
        newStrat['Vol']=newStrat['Vol'].map(lambda x : min(x,ExpoCap))
        
        #calculate the re-leveraged index
        newStrat['Vol Target']=100.0
        for k in range(3,len(newStrat)):
            perf = newStrat.loc[newStrat.index[k],'strat']/newStrat.loc[newStrat.index[k-1],'strat']-1
            perf_rate = (newStrat.index[k]-newStrat.index[k-1]).days* newStrat.loc[newStrat.index[k-1],'Rates']/36500
            newStrat.loc[newStrat.index[k],'Vol Target']=newStrat.loc[newStrat.index[k-1],'Vol Target']+newStrat.loc[newStrat.index[k-2],'Vol Target']*(newStrat.loc[newStrat.index[k-3],'Vol']*perf+(1-newStrat.loc[newStrat.index[k-3],'Vol'])*perf_rate)
        
        del newStrat['VolWindow1']
        del newStrat['VolWindow2']
        del newStrat['Vol']

        return newStrat



def Save(newStrat,Index,Benchmark,path):
    strat = Strategy(newStrat.index,newStrat[Index])    
    newStrat['_dates_']=newStrat.index
    
    newStrat["__"] = ""

    irr=strat.IRR()
    vol=strat.Annualized_Vol()
    sharpe=strat.Sharpe()
    sortino=strat.Sharpe_Cleared()
    maxDD=strat.Max_DrawDown()
    
    newStrat["Statistics"]=""
    newStrat["Statistics"][1]="IRR"
    newStrat["Statistics"][2]="Volatility"
    newStrat["Statistics"][3]="Sharpe"
    newStrat["Statistics"][4]="Sortino"
    newStrat["Statistics"][5]="Max Drawdown"

    newStrat["Values"]=""    
    newStrat["Values"][1]=irr
    newStrat["Values"][2]=vol
    newStrat["Values"][3]=sharpe
    newStrat["Values"][4]=sortino
    newStrat["Values"][5]=maxDD

    newStrat["___"] = ""
    newStrat["____"] = ""
    new_index = range(0,len(newStrat))
    
    
    dftemp = strat.Monthly_Perf()
    dftemp[Benchmark]=Strategy(newStrat.index,newStrat[Benchmark]).Monthly_Perf()['YTD']
    dftemp['Years']=dftemp.index
    
#    newStrat.set_index('Index',inplace=True)
    newStrat['index_excel']=new_index
    newStrat.set_index('index_excel',inplace=True)
    
    new_index = range(0,len(dftemp))
    dftemp['index_excel']=new_index
    dftemp.set_index('index_excel',inplace=True)
    
    frames = [newStrat,dftemp]
    newStrat=pd.concat(frames,axis = 1)
    del newStrat['Years']
    newStrat["____"]= dftemp['Years']
    
    newStrat.set_index('_dates_',inplace=True)
    newStrat.to_csv(path)
    
    print("Saved.")


