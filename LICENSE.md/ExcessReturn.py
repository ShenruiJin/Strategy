"""
                           Excess Return FUNCTIONS

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
from CBOE_Rolled_Contracts import *
from DB_functions import *
from class_Strategy import *


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
    return newStrat
