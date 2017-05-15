# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 14:16:25 2016

@author: jlinot
"""



import pandas as pd
import numpy as np
import datetime
import os
#os.chdir(r'C:\Users\jlinot\Documents\Spyder_Workspace\Statistics')
#import Stats
#os.chdir(r'C:\Users\jlinot\Documents\Spyder_Workspace\AdjustmentFactor')
#import AdjustmentFactor


#r'C:\Users\jlinot\Desktop\Work\Structuring\Structuring NYC\Proprietary Indices\BUY WRITE\SPXLTBUP Buy Write\2016-03-02\data\OverlaySignal_CaesarIndex.csv'
data_overlay = pd.read_csv(r'C:\Users\sjin\Documents\python\MRP.csv')
data_engine = pd.read_csv(r'C:\Users\sjin\Documents\python\SPXLTBUP.csv')

data_engine.columns = ['dates','prices_engine']
data_overlay.columns = ['dates','prices_overlay']

data_engine['dates'] = data_engine['dates'].apply(lambda x: datetime.datetime.strptime(x,'%d/%m/%Y'))
data_overlay['dates'] = data_overlay['dates'].apply(lambda x: datetime.datetime.strptime(x,'%d/%m/%Y'))

data_intersect = pd.merge(data_engine,data_overlay,on=['dates'])
data_intersect['vol_60bd_engine'] = pd.rolling_std(np.log(data_intersect['prices_engine'].pct_change() +1),60)*np.sqrt(252)
data_intersect['vol_60bd_overlay'] = pd.rolling_std(np.log(data_intersect['prices_overlay'].pct_change()+1),60)*np.sqrt(252)
data_intersect['weights'] = (data_intersect['vol_60bd_engine'] /data_intersect['vol_60bd_overlay']).apply(lambda x: min(2.0,max(0.0,x)))
data_intersect = data_intersect[np.isfinite(data_intersect['vol_60bd_engine'])]
data_intersect.index = data_intersect['dates']

#data_intersect['weights'] = 1.0

data_intersect['strat'] = 100.0
data_intersect['real_weights'] = float('NaN')

strikeDate = data_intersect.index[0]
for date in data_intersect.index:
    data_intersect['strat'][date] = data_intersect['strat'][strikeDate] \
    * (1.0+ data_intersect['weights'][strikeDate]*(data_intersect['prices_overlay'][date]/data_intersect['prices_overlay'][strikeDate]-1.0) \
    + (data_intersect['prices_engine'][date]/data_intersect['prices_engine'][strikeDate]-1.0))
    data_intersect['real_weights'][date] = data_intersect['weights'][strikeDate]
    if date.month % 3 == 0 and date.month != strikeDate.month:
            strikeDate = date;
data_intersect['strat_fees'] = 100.0
#data_intersect['strat_fees'] = AdjustmentFactor.annual_fees(data_intersect['strat_fees'],data_intersect['strat'],data_intersect['dates'],0.01,0)

#PLOT CHARTS
data_intersect['strat'].plot()
data_intersect['strat_fees'].plot()

#EXPORT
#data_intersect.to_csv(r'C:\Users\jlinot\Desktop\export_overlay_SPXT.csv')

#STATISTIC_RATIOS
#Stats.statistic_ratios(data_intersect['strat_fees'],data_intersect['dates'])

