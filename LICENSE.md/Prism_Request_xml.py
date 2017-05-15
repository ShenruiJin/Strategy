# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 13:55:07 2016

@author: jlinot


This program enables to create Prism Requests
The files it generates have to be pasted in the Prism Request Folder (secured folder)



"""

import xml.etree.cElementTree as ET
import datetime as datetime
import pandas as pd




def writePrismXML(outPath,productType, fields_list, ticker_list, startDate, endDate):
    root = ET.Element('prismRequest')
    
    ET.SubElement(root, 'application').text = 'Caesar'
    ET.SubElement(root,'user').text = 'FE_US'
    ET.SubElement(root,'productType').text = productType
    #ET.SubElement(root,'service').text = field
    ET.SubElement(root,'service').text ="PARAMETRABLE"
    ET.SubElement(root,'requestTime').text = (datetime.datetime.now()).strftime('%Y-%m-%d')
    ET.SubElement(root,'codeType').text = 'BBG_CODE'

    #We write the tickers in the xml
    codes = ET.SubElement(root,'codes')    
    for ticker in ticker_list:
        ET.SubElement(codes,'code').text = ticker
    
    #We write the field
    fields = ET.SubElement(root,'fields')   
    for field in fields_list:
        ET.SubElement(fields,'field').text = field
        
    #We write the start and end dates for the request
    dateRange = ET.SubElement(root,'dateRange')
    ET.SubElement(dateRange,'startDate').text = startDate.strftime('%Y-%m-%d')
    ET.SubElement(dateRange,'endDate').text = endDate.strftime('%Y-%m-%d')
    
    tree = ET.ElementTree(root)
    tree.write(outPath)
    
    
start = datetime.datetime(2000,1,1)
end = datetime.datetime(2016,4,7)

""" open the all tickers file : this is all the equity ticker we use """
ticker_path =r'C:\Users\tzercher\Desktop\tickers.csv'
  
tickerlist = pd.read_csv(ticker_path)['Tickers'].dropna().tolist()
 
start = datetime.datetime(2000,1,1)
end= datetime.datetime(2016,4,1)   
 
"""Price"""
pathFile = 'K:\ED_ExcelTools\\Transfert\\Structuring\\Proprietary Indices\\Python Script\\US Structuring Libraries\Database\\Request_Price.xml'
tickerlist_volume = []
for ticker in tickerlist:
    tickerlist_volume.append(ticker +" Equity")
writePrismXML(pathFile,'EQUITY',['PX_LAST'],tickerlist_volume,start,end)
 
"""Volume"""
pathFile = 'K:\ED_ExcelTools\\Transfert\\Structuring\\Proprietary Indices\\Python Script\\US Structuring Libraries\Database\\Request_Volume.xml'
tickerlist_volume = []
for ticker in tickerlist:
    tickerlist_volume.append(ticker +" Equity")
writePrismXML(pathFile,'EQUITY',['PX_VOLUME'],tickerlist_volume,start,end)

"""MarketCap"""
pathFile = 'K:\ED_ExcelTools\\Transfert\\Structuring\\Proprietary Indices\\Python Script\\US Structuring Libraries\Database\\Request_MarketCap.xml'
writePrismXML(pathFile,'EQUITY',['CUR_MKT_CAP'],tickerlist,start,end)





#********************** request has to be modified as prism returns void *********************
"""EarningYield"""
pathFile = 'K:\ED_ExcelTools\\Transfert\\Structuring\\Proprietary Indices\\Python Script\\US Structuring Libraries\Database\\Request_EarningYield.xml'
writePrismXML(pathFile,'EQUITY',['EarningYield'],tickerlist,start,end)
"""Free Cash Flow """
pathFile = 'K:\ED_ExcelTools\\Transfert\\Structuring\\Proprietary Indices\\Python Script\\US Structuring Libraries\Database\\Request_FreeCashFlow.xml'
writePrismXML(pathFile,'EQUITY',['FreeCashFlowYield'],tickerlist,start,end)
"""BuyBacks"""
pathFile = 'K:\ED_ExcelTools\\Transfert\\Structuring\\Proprietary Indices\\Python Script\\US Structuring Libraries\Database\\Request_Buybacks.xml'
writePrismXML(pathFile,'EQUITY',['STOCK_BUYBACK_HISTORY'],tickerlist,start,end)
"""Dividends"""
pathFile = 'K:\ED_ExcelTools\\Transfert\\Structuring\\Proprietary Indices\\Python Script\\US Structuring Libraries\Database\\Request_Dividends.xml'
writePrismXML(pathFile,'EQUITY',['DVD_HIST_ALL'],tickerlist,start,end)
#********************** request has to be modified as prism returns void ********************* 


## ////  add other requests here ///
#
#FreeCashFlowYield
#EarningYield




