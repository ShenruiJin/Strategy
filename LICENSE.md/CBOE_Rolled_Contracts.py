
#********************************************************************
# This file contains the functions for Volatility Strategies
#********************************************************************
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

#**********************************
# IMPORT NYC ENGINEERING LIBRAIRIES
#**********************************
os.chdir(r'K:\ED_ExcelTools\Transfert\Structuring\Proprietary Indices\Python Script\US Structuring Libraries')
from class_CalendarUS import * 
from DB_functions import *
from class_Strategy import *

# this function transforms an excel date to a datetime
def minimalist_xldate_as_datetime(xldate, datemode):
    # datemode: 0 for 1900-based, 1 for 1904-based
    return (
        datetime.datetime(1899, 12, 30)
        + datetime.timedelta(days=xldate + 1462 * datemode)
        )
# this function transforms datetime to an excel date
def inv_minimalist_xldate_as_datetime(date):
    # datemode: 0 for 1900-based, 1 for 1904-based
    return ((date -datetime.datetime(1899, 12, 30)).days
        )
        
#**************************
#**************************
#** US VERSION
#**************************
#**************************       
        
#*****************************************************
#   US LAST TRADING DAY CALENDAR
# ****************************************************            
#CBOE_VIX_FUT_Calendar: generates the FirstDayTrading // LastDayTrading // SettlementDate between the two years
# The CBOE settlement date of a month is defined as follows:
# The Wednesday 30-days prior to the 3rd Friday of next month
# If the 3rd Friday of next month is not a BD: find the first BD-prior
# If the day 30-prior is not a BD: find the first BD-prior
            
def CBOE_VIX_FUT_Calendar(iYearBegin,iYearEnd):
    #//US calendar creation
    startDate = datetime.datetime(iYearBegin-1,1,1) # we want a bigger interval
    endDate= datetime.datetime(iYearEnd+1,12,31) # we want a bigger interval
    calendarUS = Calendar(startDate,endDate)
    ExistingDates=calendarUS.BusinessDaysFixingDates
    lLastTradingDay=[]
    lSettlementDate=[]
    for k in range(0,iYearEnd+1-iYearBegin+1+1):
        year = iYearBegin-1+k
        #finds the settlement date and last trading date for each month
        for l in range(1,13):
            if l<12:
                month=l+1
            else:
                month=1
                year+=1
            #finds the 3rd friday of next month
            FirstDay = datetime.datetime(year,month,1).weekday()
            if FirstDay==5:
                Delay = 6
            elif FirstDay==6:
                Delay =5
            else:
                Delay = 4-FirstDay
            NextMonth3rdFriday = datetime.datetime(year,month,1+Delay+14)
            #Check if it is a business date:
            if ExistingDates.count(NextMonth3rdFriday)==0:
                NextMonth3rdFriday = datetime.datetime(year,month,1+Delay+13)
            #Find the Wednesday 30 days prior
            ExcelDate = inv_minimalist_xldate_as_datetime(NextMonth3rdFriday)
            ExcelDate-=30
            Settlement_Date= minimalist_xldate_as_datetime(float(ExcelDate),0)
            if ExistingDates.count(Settlement_Date)==0:
                Settlement_Date = calendarUS.addBusinessDays(-1,Settlement_Date)
            #carefull, the rule changed on November 14
            if Settlement_Date>=datetime.datetime(2014,11,1):
                Last_Trading_Date=Settlement_Date
            else:   
                Last_Trading_Date= calendarUS.addBusinessDays(-1,Settlement_Date)
            #save
            lSettlementDate.append(Settlement_Date)
            lLastTradingDay.append(Last_Trading_Date)     
    dfCBOE_VIX_FUT_Calendar=pd.DataFrame({'FirstTradingDay':lLastTradingDay,'LastTradingDay':lLastTradingDay,'SettlementDate':lSettlementDate})
    dfCBOE_VIX_FUT_Calendar['LastTradingDay']=dfCBOE_VIX_FUT_Calendar['LastTradingDay'].shift(-1)
    dfCBOE_VIX_FUT_Calendar['SettlementDate']=dfCBOE_VIX_FUT_Calendar['SettlementDate'].shift(-1)
    dfCBOE_VIX_FUT_Calendar=dfCBOE_VIX_FUT_Calendar.dropna()
    dfCBOE_VIX_FUT_Calendar=dfCBOE_VIX_FUT_Calendar[dfCBOE_VIX_FUT_Calendar['FirstTradingDay']>=datetime.datetime(iYearBegin-1,12,1)]
    dfCBOE_VIX_FUT_Calendar=dfCBOE_VIX_FUT_Calendar[dfCBOE_VIX_FUT_Calendar['FirstTradingDay']<=datetime.datetime(iYearEnd+1,1,1)]
    dfCBOE_VIX_FUT_Calendar.reset_index(inplace=True)
    dfCBOE_VIX_FUT_Calendar=dfCBOE_VIX_FUT_Calendar.drop('index',1)
    dfCBOE_VIX_FUT_Calendar['LengthPeriod']=0.0
    for k in range(0,len(dfCBOE_VIX_FUT_Calendar)):
        FirstDay=dfCBOE_VIX_FUT_Calendar['FirstTradingDay'][k]
        #exlude the first trading day, include the last one
        FirstDay=calendarUS.addBusinessDays(1,FirstDay)
        LastDay=dfCBOE_VIX_FUT_Calendar['LastTradingDay'][k]
        dfCBOE_VIX_FUT_Calendar['LengthPeriod'][k] = calendarUS.nbBusinessDaysBetweenTwoDates(FirstDay,LastDay)
    return  dfCBOE_VIX_FUT_Calendar     
            
#*****************************************************
#   US SHORT TERM ROLL
# **************************************************** 
            
class US_ST_Rolled_Vol_Index_NTX():
    def __init__(self,Contract1,Contract2,Contract3,dateBegin,dateEnd):
        
        Contract_list=[Contract1,Contract2,Contract3]
        calendar=Calendar(dateBegin, dateEnd).BusinessDaysFixingDates
        newStrat= Asset_Price_getPrices(Contract_list,['Last_Price'],dateBegin,dateEnd,False,calendar)
        newStrat=newStrat.fillna(method='ffill')
        newStrat=newStrat.dropna()

        #read data
      
        startDate=newStrat.index[0]
        endDate=newStrat.index[len(newStrat)-1]
        iYearBegin=startDate.year
        iYearEnd=endDate.year
        
        #create CBOE Calendar
        dfCBOE_VIX_FUT_Calendar=CBOE_VIX_FUT_Calendar(iYearBegin,iYearEnd)
        lRollDates = dfCBOE_VIX_FUT_Calendar['LastTradingDay'].drop_duplicates().tolist()
        calendarUS = Calendar(startDate,endDate)
        #convert in datetime
        newStrat['Dates_Format']=newStrat.index
        #find the roll period for each date
        l=0
        while dfCBOE_VIX_FUT_Calendar['LastTradingDay'][l]<newStrat['Dates_Format'][0]:
            l+=1
        newStrat['FirstTradingDay']=datetime.datetime(1990,1,1)
        newStrat['LastTradingDay']=datetime.datetime(1990,1,1)
        newStrat['SettlementDate']=datetime.datetime(1990,1,1)
        newStrat['IsRoll']=0.0
        newStrat['t']=0.0
        for k in range(0,len(newStrat)):
            today=newStrat['Dates_Format'][k]
            if today >dfCBOE_VIX_FUT_Calendar['LastTradingDay'][l]:
                l+=1
            newStrat['FirstTradingDay'][k]=dfCBOE_VIX_FUT_Calendar['FirstTradingDay'][l]
            newStrat['LastTradingDay'][k]=dfCBOE_VIX_FUT_Calendar['LastTradingDay'][l]
            newStrat['SettlementDate'][k]=dfCBOE_VIX_FUT_Calendar['SettlementDate'][l]
            newStrat['t'][k]=dfCBOE_VIX_FUT_Calendar['LengthPeriod'][l]
            if lRollDates.count(today)==1:
                newStrat['IsRoll'][k]=1    
        newStrat['r']=0.0
        for k in range(0,len(newStrat)):
            FirstDay=newStrat['Dates_Format'][k]
            LastDay = newStrat['LastTradingDay'][k]
            if FirstDay==LastDay:
                newStrat['r'][k]=0
            else:    
                #from the next BD to the last trading day
                FirstDay=calendarUS.addBusinessDays(1,FirstDay)
                newStrat['r'][k]=calendarUS.nbBusinessDaysBetweenTwoDates(FirstDay,LastDay)
        #computes weights    
        newStrat['W1']=newStrat['r']/newStrat['t']
        newStrat['W2']=(newStrat['t']-newStrat['r'])/newStrat['t']

        newStrat['Index']=100.0
        for k in range(1,len(newStrat)):
            if newStrat['Dates_Format'][k-1]==newStrat['LastTradingDay'][k-1]:
                #Yesterday was the roll day: W2 in the second contract= W2 in the first contract
                w1=newStrat['W2'][k-1]
                w2=newStrat['W1'][k-1]
                U1_yesterday = newStrat["Last_Price."+str(Contract2)][k-1]
                U2_yesterday=newStrat["Last_Price."+str(Contract3)][k-1]
                ValueYesterday= w1*U1_yesterday
                ValueToday = w1*newStrat["Last_Price."+str(Contract1)][k]
            else:
                w1=newStrat['W1'][k-1]
                w2=newStrat['W2'][k-1]
                U1_yesterday = newStrat["Last_Price."+str(Contract1)][k-1]
                U2_yesterday=newStrat["Last_Price."+str(Contract2)][k-1]
                ValueYesterday= w1*U1_yesterday+w2*U2_yesterday
                ValueToday = w1*newStrat["Last_Price."+str(Contract1)][k]+w2*newStrat["Last_Price."+str(Contract2)][k]
            newStrat['Index'][k]=newStrat['Index'][k-1]*ValueToday/ValueYesterday
            
        self.Index=newStrat
        

class US_MT_Rolled_Vol_Index_NTX():
    def __init__(self,Contract1,Contract2,Contract3,Contract4,Contract5,dateBegin,dateEnd):
        Contract_list=[Contract1,Contract2,Contract3,Contract4,Contract5]
        calendar=Calendar(dateBegin, dateEnd).BusinessDaysFixingDates
        newStrat= Asset_Price_getPrices(Contract_list,['Last_Price'],dateBegin,dateEnd,False,calendar)
        newStrat=newStrat.fillna(method='ffill')
        newStrat=newStrat.dropna()
        
        #read data
      
        startDate=newStrat.index[0]
        endDate=newStrat.index[len(newStrat)-1]
        iYearBegin=startDate.year
        iYearEnd=endDate.year
        
        #create CBOE Calendar
        dfCBOE_VIX_FUT_Calendar=CBOE_VIX_FUT_Calendar(iYearBegin,iYearEnd)
        lRollDates = dfCBOE_VIX_FUT_Calendar['LastTradingDay'].drop_duplicates().tolist()
        calendarUS = Calendar(startDate,endDate)
        #convert in datetime
        newStrat['Dates_Format']=newStrat.index
        l=0
        while dfCBOE_VIX_FUT_Calendar['LastTradingDay'][l]<newStrat['Dates_Format'][0]:
            l+=1
        newStrat['FirstTradingDay']=datetime.datetime(1990,1,1)
        newStrat['LastTradingDay']=datetime.datetime(1990,1,1)
        newStrat['SettlementDate']=datetime.datetime(1990,1,1)
        newStrat['IsRoll']=0.0
        newStrat['t']=0.0
        for k in range(0,len(newStrat)):
            today=newStrat['Dates_Format'][k]
            if today >dfCBOE_VIX_FUT_Calendar['LastTradingDay'][l]:
                l+=1
            newStrat['FirstTradingDay'][k]=dfCBOE_VIX_FUT_Calendar['FirstTradingDay'][l]
            newStrat['LastTradingDay'][k]=dfCBOE_VIX_FUT_Calendar['LastTradingDay'][l]
            newStrat['SettlementDate'][k]=dfCBOE_VIX_FUT_Calendar['SettlementDate'][l]
            newStrat['t'][k]=dfCBOE_VIX_FUT_Calendar['LengthPeriod'][l]
            if lRollDates.count(today)==1:
                newStrat['IsRoll'][k]=1
            
        newStrat['r']=0.0
        for k in range(0,len(newStrat)):
            FirstDay=newStrat['Dates_Format'][k]
            LastDay = newStrat['LastTradingDay'][k]
            if FirstDay==LastDay:
                newStrat['r'][k]=0
            else:    
                #from the next BD to the last trading day
                FirstDay=calendarUS.addBusinessDays(1,FirstDay)
                newStrat['r'][k]=calendarUS.nbBusinessDaysBetweenTwoDates(FirstDay,LastDay)
            
        newStrat['W1']=newStrat['r']/newStrat['t']
        newStrat['W2']=1.0
        newStrat['W3']=1.0
        newStrat['W4']=(newStrat['t']-newStrat['r'])/newStrat['t']
        
        
        newStrat['Index']=100.0
        for k in range(1,len(newStrat)):
            if newStrat['Dates_Format'][k-1]==newStrat['LastTradingDay'][k-1]:
                #Yesterday was the roll day: W2 in the second contract= W2 in the first contract
                w1=1.0
                w2=1.0
                w3=1.0
                w4=0.0
                U1_yesterday = newStrat["Last_Price."+str(Contract2)][k-1]
                U2_yesterday=newStrat["Last_Price."+str(Contract3)][k-1]
                U3_yesterday=newStrat["Last_Price."+str(Contract4)][k-1]
                U4_yesterday=newStrat["Last_Price."+str(Contract5)][k-1]
                ValueYesterday= w1*U1_yesterday+w2*U2_yesterday+w3*U3_yesterday
                ValueToday= w1*newStrat["Last_Price."+str(Contract1)][k]+w2*newStrat["Last_Price."+str(Contract2)][k]+w3*newStrat["Last_Price."+str(Contract3)][k]
            else:
                w1=newStrat['W1'][k-1]
                w2=1.0
                w3=1.0
                w4=newStrat['W4'][k-1]
                U1_yesterday = newStrat["Last_Price."+str(Contract1)][k-1]
                U2_yesterday=newStrat["Last_Price."+str(Contract2)][k-1]
                U3_yesterday=newStrat["Last_Price."+str(Contract3)][k-1]
                U4_yesterday=newStrat["Last_Price."+str(Contract4)][k-1]
                ValueYesterday= w1*U1_yesterday+w2*U2_yesterday+w3*U3_yesterday+w4*U4_yesterday
                ValueToday = w1*newStrat["Last_Price."+str(Contract1)][k]+w2*newStrat["Last_Price."+str(Contract2)][k]+w3*newStrat["Last_Price."+str(Contract3)][k]+w4*newStrat["Last_Price."+str(Contract4)][k]
            newStrat['Index'][k]=newStrat['Index'][k-1]*ValueToday/ValueYesterday
            
        self.Index=newStrat
    
    







