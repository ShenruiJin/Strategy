# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 17:21:08 2016

@author: jlinot
"""
import os
import pandas as pd
import datetime
import xml.etree.ElementTree

#**********************************
# IMPORT US STRUCTURING LIBRAIRIES
#**********************************
os.chdir(r'C:\Users\sjin\Documents\Local_Code')
import class_CalendarUS
from DB_functions import *


def MarketCapXML(tree):
    dct_Underlyings = {}
    
    """ do for each index """
    for elt in tree.iter('element'):
        attrib = elt.attrib
        ticker = attrib['code']
        
        if ticker not in dct_Underlyings.keys():
            dct_Underlyings[ticker]= {}
            dct_Underlyings[ticker]['Date']= []
            dct_Underlyings[ticker]['MarketCap']= []
        date = attrib['date'][0:10]
        date= datetime.datetime.strptime(date,'%Y-%m-%d')
        dct_Underlyings[ticker]['Date'].append(date)
        dct_Underlyings[ticker]['MarketCap'].append(float(attrib['cur_mkt_cap']))
       
    dct_MarketCap ={}
    for ticker in dct_Underlyings.keys():
        df=pd.DataFrame({'Date.'+ticker:dct_Underlyings[ticker]['Date'],'MarketCap.'+ticker:dct_Underlyings[ticker]['MarketCap']})
        dct_MarketCap[ticker]=df
        
    return dct_MarketCap
    

def VolumeXML(tree):
    dct_Underlyings = {}
    
    """ do for each index """
    for elt in tree.iter('element'):
        attrib = elt.attrib
        ticker = attrib['code']
        
        if ticker not in dct_Underlyings.keys():
            dct_Underlyings[ticker]= {}
            dct_Underlyings[ticker]['Date']= []
            dct_Underlyings[ticker]['Volume']= []
        date = attrib['date'][0:10]
        date= datetime.datetime.strptime(date,'%Y-%m-%d')
        dct_Underlyings[ticker]['Date'].append(date)
        dct_Underlyings[ticker]['Volume'].append(float(attrib['px_volume']))
       
    dct_Volume ={}
    for ticker in dct_Underlyings.keys():
        df=pd.DataFrame({'Date.'+ticker:dct_Underlyings[ticker]['Date'],'Volume.'+ticker:dct_Underlyings[ticker]['Volume']})
        dct_Volume[ticker]=df
        
    return dct_Volume
    
    
    
def BasketXML(tree):
    
    dct_Underlyings = {}
    
    """ do for each index """
    for index in tree.iter('index'):
        attrib = index.attrib
        Underlying = attrib['code']
        dct_Baskets = {}
        for composition in index.iter('composition'):
            date =composition.attrib['startDate']
            date = date[0:10]
            date= datetime.datetime.strptime(date,'%Y-%m-%d')
            ticker =[]
            weightCoefficient=[]
            weight = []
           
            for components in composition.iter('components'):
                for component in components.iter('component'):
                    attrib = component.attrib
                    try:
                        ticker.append(attrib['code'])
                        weightCoefficient.append(attrib['weightCoefficient'])
                        weight.append(attrib['weight'])
                    except KeyError:
                        pass
            
            Basket = pd.DataFrame({'Ticker':ticker,'WeightCoefficient':weightCoefficient,'Weight':weight})
            dct_Baskets[date]= Basket
        dct_Underlyings[Underlying] = dct_Baskets
        
    return dct_Underlyings
      

def priceProductXML(tree):
    dicoPrices = {}
    for price in tree.find('prices'):
        attrib = price.attrib
        ticker = attrib['code']
        date = datetime.datetime.strptime(attrib['date'],'%Y%m%d')
        opening = attrib['openingPrice']
        high = attrib['highPrice']
        low = attrib['lowPrice']
        closing = attrib['closingPrice']
        
        if (ticker in dicoPrices.keys()):
            dicoPrices[ticker].update({date : [opening, high, low, closing]})
        else:
            dicoPrices.update({ticker : {date : [opening, high, low, closing]}})            
    
    dico_dataFrame = {}
    for ticker in dicoPrices.keys():
        subDico = dicoPrices[ticker]
        df = pd.DataFrame(list(subDico.keys()),columns=['date.'+ticker])
        df['open.'+ticker] = float('NaN')
        df['high.'+ticker] = float('NaN')
        df['low.'+ticker] = float('NaN')
        df['last.'+ticker] = float('NaN')
        
        i = 0
        for date in df['date.'+ticker]:
            df['open.'+ticker][i] = float(subDico[date][0])
            df['high.'+ticker][i] = float(subDico[date][1])
            df['low.'+ticker][i] = float(subDico[date][2])
            df['last.'+ticker][i] = float(subDico[date][3])
            i += 1
        
        dico_dataFrame.update({ticker : df})
    
    return dico_dataFrame


def getFieldsName(tree):
    name_fields = []
    sub = tree.find('elements')[0] 
        
    for key in sub.keys():
            if (key not in ['code','date']):
                name_fields.append(key)
    
    return name_fields
    
    
def genericResponseXML(tree):
    dicoPrices = {}

    fields = getFieldsName(tree)
    
    for elem in tree.find('elements'):
        attrib = elem.attrib
        list_flds = []
        for key in attrib.keys():
            if (key=='code'):
                ticker = attrib['code']
            elif (key == 'date'):
                date = index_to_datetime(attrib['date'])[0]
            else:
                list_flds.append(attrib[key])
        
        if (ticker in dicoPrices.keys()):
            dicoPrices[ticker].update({date : list_flds})
        else:
            dicoPrices.update({ticker : {date : list_flds}})            
    
    dico_dataFrame = {}
    for ticker in dicoPrices.keys():
        subDico = dicoPrices[ticker]
        df = pd.DataFrame(list(subDico.keys()),columns=['date.'+str(ticker)])
        
        for flds in fields:
            df[flds] = float('NaN')
        
        i = 0
        for date in df['date.'+str(ticker)]:
            j=0
            for flds in fields:
                df[flds] = subDico[date][j]
                j += 1
            
            i += 1
        
        dico_dataFrame.update({ticker : df})
        
    return dico_dataFrame
    
        
#
#
#tree = xml.etree.ElementTree.parse(path).getroot()
#if (tree.tag== 'priceResponse'):
#    dico_df = priceProductXML(tree)
#else:
#    if tree.tag = 'indexCompositionResponse' :
#        a=1
#    else:
#        dico_df = genericResponseXML(tree)

#USpath =r'C:\Users\sjin\Documents\Local_Code\Database\PrismRequest\PrismResponse_Volume.xml'
#tree = xml.etree.ElementTree.parse(USpath).getroot()
#dct_Underlyings=BasketXML(tree)
#tickers=[]
#for underlying in dct_Underlyings.keys():
#    for date in dct_Underlyings[underlying ].keys():
#        for ticker in dct_Underlyings[underlying ][date]['Ticker']:
#            tickers.append(ticker)
#            
#UEpath = r'C:\Users\sjin\Documents\Local_Code\Database\PrismRequest\PrismResponse_Basket_SX5E.xml'
#tree = xml.etree.ElementTree.parse(UEpath ).getroot()
#dct_Underlyings=BasketXML(tree)
#for underlying in dct_Underlyings.keys():
#    for date in dct_Underlyings[underlying ].keys():
#        for ticker in dct_Underlyings[underlying ][date]['Ticker']:
#            tickers.append(ticker)            
#
#df = pd.DataFrame({'Tickers':tickers})
#df=df.dropna()
#df=df.drop_duplicates()
#
#df.to_csv(r'C:\Users\sjin\Documents\Local_Code\Database\AllEquityTickers.csv')

#
#dates = []
#for date in dct_Underlyings[underlying ].keys():
#    dates.append(date)
#
#df = pd.DataFrame({'Dates':dates})
#l=df['Dates'].tolist()
#l.sort()
#l[0]
#df.sort(ascending = False)