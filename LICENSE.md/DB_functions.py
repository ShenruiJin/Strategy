# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 18:08:35 2016

@author: tzercher
"""
#*****************************************************************************
#                           IMPORTS
#*****************************************************************************


""" GLOBAL VARIABLE: DICTIONNARY """

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
os.chdir(r'H:\Local_Code')
import class_CalendarUS
from readRequest import *


def to_datetime(xldate, datemode):
    # datemode: 0 for 1900-based, 1 for 1904-based
    return (
        datetime.datetime(1899, 12, 30)
        + datetime.timedelta(days=xldate + 1462 * datemode)
        )

def index_to_datetime(index):
    """
    delete all blanks
    """
    index = index.tolist()
    for k in range(0,len(index)):
        if type(index[k]) == str:
            if " " in index[k]:
                index[k]= index[k].split()[0]
            
    df= pd.DataFrame({'Index':index})
    index = df['Index']
    if type(index[0])==np.int64 or type(index[0])==float or type(index[0])==np.float64:
        index=index.map(lambda x: to_datetime(float(x),0))
    else:
        if type(index[0])== str:
            try:
                index=index.map(lambda x: datetime.datetime.strptime(x,"%m/%d/%Y"))
            except ValueError:
                try:                
                    index=index.map(lambda x: datetime.datetime.strptime(x,"%Y/%m/%d"))
                except ValueError:
                    try:
                        index=index.map(lambda x: datetime.datetime.strptime(x,"%m-%d-%Y"))
                    except ValueError:
                        try:
                            index=index.map(lambda x: datetime.datetime.strptime(x,"%Y-%m-%d"))
                        except ValueError:
                            try:
                                index=index.map(lambda x: datetime.datetime.strptime(x,"%m%d%Y"))
                            except ValueError:
                                try:
                                    index=index.map(lambda x: datetime.datetime.strptime(x,"%Y%m%d"))
                                except ValueError:
                                    try:
                                        index=index.map(lambda x: datetime.datetime.strptime(x,"%d%m%Y"))
                                    except ValueError:
                                        print('string format not implemented,'+ str(index))
                         
    return index   
    

# this function transforms datetime to an excel date
def inv_to_datetime(date):
    # datemode: 0 for 1900-based, 1 for 1904-based
    return ((date -datetime.datetime(1899, 12, 30)).days
        )


"""
The following functions returns the last created Database
"""
def Last_DB():
    lDB=[]
    lCreated_at = []
    path = r'C:\Users\sjin\Documents\Local_Code'
    os.chdir(path)
    for element in os.listdir(path):
        if element.endswith(".db"):
            if element != "Thumbs.db":
                lDB.append(element)
                lCreated_at.append(os.path.getmtime(element))  
    database = pd.DataFrame({'File':lDB,'Created_at':lCreated_at})
    database=database.sort('Created_at',ascending = False)
    dbName = database.loc[database.index[0],'File']
    return dbName



"""

This class helps the user to understand how to use request functions

"""
class Data_Request():
    def __init__(self):
        self.BDH_Fields = ["Last_Price","Bid","Ask","Volume","Free_Cash_Flow","Market_Cap","Earning_Yield"]
        self.BDS_Fields =['Dividends','BuyBacks']        
    def Help(self, function):
        if function == "DB_BDH_Call" :
            message=" " +"\n"
            message = "The DB_BDH_Call works as the BDH function in Bloomberg"+ "\n"
            message += " "+ "\n"
            message += "Provide a ticker_list,field_list" +"\n"
            message += " "+ "\n"
            message += "The complete list of fields is the following:" +"\n"
            message += " "+ "\n"
            for flds in self.BDH_Fields:
                message +=flds+"\n"
            return message

        if function == "DB_BDS_Call" :
            message=" " +"\n"
            message = "The DB_BDS_Call works as the BDS function in Bloomberg"+ "\n"
            message += " "+ "\n"
            message += "Provide a ticker_list, field_list, dtBegin, dtEnd, bIntersection, calendar" +"\n"
            message += " "+ "\n"
            message += "Calendar is a date list you provide to get data at those specific dates. By default, we take all available dates between dtBegin,dtEnd"+"\n"
            message += " "+ "\n"
            message +="bIntersection is a Boolean. True if you want shared dates only." +"\n"
            message += " "+ "\n"
            message += "The complete list of fields is the following:" +"\n"
            message += " "+ "\n"
            for flds in self.BDS_Fields:
                message +=flds+"\n"
            return message
            
      
#**********************************
#                                 Database Creation
#**********************************
      
""" the following function creates a new DB"""
def DB_creation(DBname = np.nan):
  if np.isnan(DBname):
      now = datetime.datetime.today()
      day = now.day
      month = now.month
      year =now.year
      hour = now.hour
      minute = now.minute
      DBname = "DB"+str(month) + str(day) + str(year) +"_" + str(hour)+"h"+str(minute)+"min.db"
  conn = sqlite3.connect(DBname)
  conn.commit()
  conn.close()
  return DBname +" created."
  
""" the following function creates a new database and the tables of the database"""      
def DB_creationScript(DBname = np.nan):  
    """ create the database """
    message= DB_creation(DBname)
    message += "\n"
    """ create the asset table """
    message+=Asset_TableCreation()
    message += "\n"
    """ create the Price table """
    message += Asset_Price_TableCreation()
    message +="\n"
    """ create Calendar Table"""
    message += Calendar_TableCreation()
    message +="\n"
    """ create the Basket table """
    message += Basket_TableCreation()    
    message += "\n"
    """ create the BasketComponents table """
    message += BasketComponents_TableCreation()    
    message += "\n"
    """ create the Dividends table """
    message += Dividends_TableCreation()
    message += "\n"
    """ create the Buyback table """
    message += Buybacks_TableCreation()
    message += "\n"
    """ create the FreeCashFlowtable """  
    message += FreeCashFlow_TableCreation()
    message += "\n"
    """MarketCap """    
    message +=MarketCap_TableCreation()
    message +="\n"
    """EarningYield """    
    message += EarningYield_TableCreation()
    message += "\n"
    """Trading volume """  
    message += TradingVolume_TableCreation()
    message += "\n"

    return message
    
    
#""" this function creates all existing tickers"""  
#
#def DB_FillAssetTable(DataBase_path):
#    dfTickers = pd.read_csv(DataBase_path+"\Tickers.csv")
#    """ Europe stocks """
#    EuropeCash_ticker_list =dfTickers['Europe Cash'].dropna().tolist()
#    message = "European assets: "+Asset_setTickers(EuropeCash_ticker_list)
#    message +="\n"
#    
#    """ US stocks """
#    USCash_ticker_list=dfTickers['US Cash'].dropna().tolist()
#    message += "US assets: "+ Asset_setTickers(USCash_ticker_list)
#    message +="\n"
#    
#    """Futures"""
#    Futures_ticker_list =dfTickers['Futures'].dropna().tolist()
#    message += "Futures: "+Asset_setTickers(Futures_ticker_list)
#    message +="\n"
#    
#    """NXS Strat"""
#    NXS_ticker_list =dfTickers["NXS Strategies"].dropna().tolist()
#    message += "Strategies: "+Asset_setTickers(NXS_ticker_list)
#    message +="\n"
#    
#    return message 
    
""" this function creates all existing price (bid last ask) hsty"""  

def DB_FillAssetPriceTable(Prism_path): 
    
#    xml_path=Prism_path  + '\\response_request_Price.xml'
#    
#    tree = xml.etree.ElementTree.parse(xml_path).getroot()
#    dico_df = priceProductXML(tree)
#    
#    tickers = list(dico_df.keys())
#    for ticker in tickers :
#        Asset_Price_setHistoricalPrice(ticker,dico_df[ticker])
#    
#    message = "European assets: "+Asset_Price_setHistoricalPrice(csv_path)
#    message +="\n"
#    """ US stocks => all us stocks  """
#    csv_path=Prism_path+"\\USCashPrice.csv"
#    message += "US assets: "+Asset_Price_setHistoricalPrice_from_csv(csv_path)
#    message +="\n"
    """
    Old data
    """
    message=""
    csv_path=Prism_path+"\\Past_Request\\"
    for k in range(1,7):
        csv_path_file=csv_path+"BloombergPrice"+str(k) + ".csv"
        message +=  "Old Prices: "+Asset_Price_setHistoricalPrice_from_csv(csv_path_file)
        message +="\n"
        
    """
    Last Prices
    """
    dbName = Last_DB()
    conn = sqlite3.connect(dbName)
    db_cursor=conn.cursor() 
    
    path_ticker = r'C:\Users\sjin\Documents\Local_Code\Database\AllEquityTickers.csv'
    tickerlist = pd.read_csv(path_ticker)['Tickers'].dropna().tolist()
    
    tickerId =Asset_getAssetsId_fromTicker(tickerlist, db_cursor) 
    
    request = ""
    for tickerid in tickerId:
         request+= "Asset_ID = " +str(tickerid) +" OR "
    request=request[0:len(request)-1-3]
    db_cursor.execute("Delete from Asset_Price where " +request)
    conn.commit()
    conn.close()
        
    csv_path=Prism_path+"\\New_Request\\"    
    for k in range(1,7):
        csv_path=Prism_path+"\\New_Request\\"+"BloombergPrice"+str(k) + ".csv"
        message +=  "New Prices: "+Asset_Price_setHistoricalPrice_from_csv(csv_path)
        message +="\n"
    
    """ Futures and strategies """
    csv_path=Prism_path+"\\BloombergPrice.csv"
    message +=  "Bloomberg: "+Asset_Price_setHistoricalPrice_from_csv(csv_path)
    message +="\n"
    
##    """ Strats"""
#    csv_path=r'C:\Users\tzercher\Desktop\dataJBL.csv'
#    message +=  "Strategies: "+Asset_Price_setHistoricalPrice_from_csv(csv_path)
#    message +="\n"
    return message


def DB_FillMarketCapTable(Prism_path):
    message = ""
    """
    Old data
    """
    message=""
    csv_path=Prism_path+"\\Past_Request\\"
    for k in range(1,7):
        csv_path_file=csv_path+"BloombergMarketCap"+str(k) + ".csv"
        message +=  "Old Prices: "+MarketCap_setMarketCap_from_csv(csv_path_file)
        message +="\n"
        
    dbName = Last_DB()
    conn = sqlite3.connect(dbName)
    db_cursor=conn.cursor() 
    
    path_ticker = r'C:\Users\sjin\Documents\Local_Code\Database\AllEquityTickers.csv'
    tickerlist = pd.read_csv(path_ticker)['Tickers'].dropna().tolist()
    
    tickerId =Asset_getAssetsId_fromTicker(tickerlist, db_cursor)   
    request = ""
    for tickerid in tickerId:
         request+= "Asset_ID = " +str(tickerid) +" OR "
    request=request[0:len(request)-1-3]
    db_cursor.execute("Delete from MarketCap where " +request)
    conn.commit()
    conn.close() 
    
    """
    New data
    """
    message=""
    csv_path=Prism_path+"\\New_Request\\"
    for k in range(1,7):
        csv_path_file=csv_path+"BloombergMarketCap"+str(k) + ".csv"
        message +=  "New Prices: "+MarketCap_setMarketCap_from_csv(csv_path_file)
        message +="\n"
        
        
    
    
 
def DB_FillVolumeTable(Prism_path):
    message = ""
    """
    Old data
    """
    message=""
    csv_path=Prism_path+"\\Past_Request\\"
    for k in range(1,7):
        csv_path_file=csv_path+"BloombergVolume"+str(k) + ".csv"
        message +=  "Old Prices: "+TradingVolume_setHistoricalVolume_from_csv(csv_path_file)
        message +="\n"
    
    
    dbName = Last_DB()
    conn = sqlite3.connect(dbName)
    db_cursor=conn.cursor() 
    
    path_ticker = r'C:\Users\sjin\Documents\Local_Code\Database\AllEquityTickers.csv'
    tickerlist = pd.read_csv(path_ticker)['Tickers'].dropna().tolist()
    
    tickerId =Asset_getAssetsId_fromTicker(tickerlist, db_cursor)   
    request = ""
    for tickerid in tickerId:
         request+= "Asset_ID = " +str(tickerid) +" OR "
    request=request[0:len(request)-1-3]
    db_cursor.execute("Delete from TradingVolume where " +request)
    conn.commit()
    conn.close() 
    
    """
    New data
    """
    message=""
    csv_path=Prism_path+"\\New_Request\\"
    for k in range(1,7):
        csv_path_file=csv_path+"BloombergVolume"+str(k) + ".csv"
        message +=  "New Prices: "+TradingVolume_setHistoricalVolume_from_csv(csv_path_file)
        message +="\n"
        
        
    return message
""" this function creates all existing baskets"""  

def DB_FillBasketTable(Prism_path):
    """ SPX """
    xml_path=Prism_path  + '\\response_Request_Basket_SPX.xml'
    tree = xml.etree.ElementTree.parse(xml_path).getroot()
    dct_Underlyings =BasketXML(tree)
    
    # *** Add the SPX baskets former to 2009 
    csv_path=Prism_path+"\Baskets.csv"
    df = pd.read_csv(csv_path)
    for column in df.columns:
        if column[0:3] == 'SPX':
            date = to_datetime(float(df[column][0]),0)
            Ticker = []
            WeightCoefficient=[]
            Weight=[]
            df_basket= df[[column]].dropna()
            for k in range(1,len(df_basket)):
                Ticker.append(df_basket[column][k])
                WeightCoefficient.append("")
                Weight.append("")
            Basket = pd.DataFrame({'Ticker':Ticker,'WeightCoefficient':WeightCoefficient,'Weight':Weight})
            dct_Underlyings['SPX'][date]=Basket
    
    message=Basket_set_BasketComponents(dct_Underlyings)
    message += "\n"
    

    """SX5E"""
    xml_path=Prism_path  + '\\response_Request_Basket_SX5E.xml'
    tree = xml.etree.ElementTree.parse(xml_path).getroot()
    dct_Underlyings =BasketXML(tree)

    # *** Add the SX5E baskets former to 2009 
    csv_path=Prism_path+"\Baskets.csv"
    df = pd.read_csv(csv_path)
    for column in df.columns:
        if column[0:2] == 'SX5E':
            date = to_datetime(df[column][0],0)
            Ticker = []
            WeightCoefficient=[]
            Weight=[]
            df_basket= df[[column]].dropna()
            for k in range(1,len(df_basket)):
                Ticker.append(df_basket[column][k])
                WeightCoefficient.append("")
                Weight.append("")
            Basket = pd.DataFrame({'Ticker':ticker,'WeightCoefficient':WeightCoefficient,'Weight':Weight})
            dct_Underlyings['SX5E'][date]=Basket
    message+=Basket_set_BasketComponents(dct_Underlyings)
    message += "\n"
    
    return message
    
    
""" this function creates all buyback hsty """    
  
def DB_FillBuybacks(Prism_path): 
    
#    xml_path=Prism_path  + '\\response_Request_Buybacks.xml'
#    tree = xml.etree.ElementTree.parse(xml_path).getroot()
#    dct_Buybacks =BuybacksXML(tree)
#    tickers = list(dct_Buybacks.keys())
#    icount = len(tickers)
#    
#    for ticker in tickers:
#        message_temp = Buybacks_setBuybacks(dct_Buybacks[ticker])
#    message = "BuyBacks: " +  str(icount) +" insertions done."
    
#    """ European stocks """
    for i in range(1,18):
        csv_path=Prism_path+"\\Past_Request\\BloombergBuyBacks"+str(i)+".csv"
        message = "Bloomberg: "+Buybacks_setBuybacks_from_csv(csv_path)
    """
    Last Prices
    """
    dbName = Last_DB()
    conn = sqlite3.connect(dbName)
    db_cursor=conn.cursor()
    
    path_ticker = r'C:\Users\sjin\Documents\Local_Code\Database\AllEquityTickers.csv'
    tickerlist = pd.read_csv(path_ticker)['Tickers'].dropna().tolist()
    
    tickerId =Asset_getAssetsId_fromTicker(tickerlist, db_cursor)   
    request = ""
    for tickerid in tickerId:
         request+= "Asset_ID = " +str(tickerid) +" OR "
    request=request[0:len(request)-1-3]
    db_cursor.execute("Delete from Buybacks where " +request)
    conn.commit()
    conn.close()

    csv_path=Prism_path+"\\New_Request\\"    
    for k in range(1,7):
        csv_path_file=csv_path+"BloombergBuybacks"+str(k) + ".csv"
        message += "Bloomberg: "+Buybacks_setBuybacks_from_csv(csv_path_file)
        message +="\n"
#    message+="\n"
#    """ US stocks => all us stocks  """
#    csv_path=Prism_path+ "\\USCashBuybacks.csv"
#    message += "US assets: "+ Buybacks_setBuybacks_from_csv(csv_path)    
    return message 
    

    
def DB_FillEarningYield(Prism_path):
    
#    xml_path=Prism_path  + '\\response_Request_EarningYield.xml'
#    tree = xml.etree.ElementTree.parse(xml_path).getroot()
#    dct_EarningYield =EarningYieldXML(tree)
#    
#    tickers = list(dct_EarningYield.keys())
#    icount = len(tickers)
#    for ticker in tickers:
#        message_temp = AssetEarningYield_setAssetEarningYield(dct_EarningYield[ticker])
#    message ="Earning Yield: " + str(icount) +" insertions done."
    
    for i in range(1,18):
        csv_path=Prism_path+"\\Past_Request\\BloombergEarningYield"+str(i)+".csv"
        message =  "Bloomberg: "+ AssetEarningYield_setAssetEarningYield_from_csv(csv_path)
    
    dbName = Last_DB()
    conn = sqlite3.connect(dbName)
    db_cursor=conn.cursor() 
    path_ticker = r'C:\Users\sjin\Documents\Local_Code\Database\AllEquityTickers.csv'
    tickerlist = pd.read_csv(path_ticker)['Tickers'].dropna().tolist()
    
    tickerId =Asset_getAssetsId_fromTicker(tickerlist, db_cursor)   
    request = ""
    for tickerid in tickerId:
         request+= "Asset_ID = " +str(tickerid) +" OR "  
  
    request=request[0:len(request)-1-3]
    db_cursor.execute("Delete from EarningYield where " +request)
    conn.commit()
    conn.close() 
    
    
    csv_path=Prism_path+"\\New_Request\\"        
    for k in range(1,2):
        csv_path_file=csv_path+"BloombergEarningYield"+str(k) + ".csv"
        message += "Bloomberg: "+AssetEarningYield_setAssetEarningYield_from_csv(csv_path_file)
        message +="\n"
#    csv_path=Prism_path+"\EuropeCashEarningYield.csv"
#    message="European assets "+ AssetEarningYield_setAssetEarningYield_from_csv(csv_path)
#    message+="\n"
#    csv_path=Prism_path+"\\USCashEarningYield.csv"
#    message="US assets "+ AssetEarningYield_setAssetEarningYield_from_csv(csv_path)
#    
    return message    
    
def DB_FillFreeCashFlow(Prism_path):
    
#    xml_path=Prism_path  + '\\response_Request_FreeCashFlow.xml'
#    tree = xml.etree.ElementTree.parse(xml_path).getroot()
#    dct_FreeCashFlow =FreeCashFlowXML(tree)
#    
#    tickers = list(dct_FreeCashFlow.keys())
#    icount = len(tickers)
#    for ticker in tickers:
#        message_temp = FreeCashFlow_setFreeCashFlow(dct_FreeCashFlow[ticker])
#    message = "Free Cash Flows: " +  str(icount) +" insertions done."
    
#    for i in range(1,18):
#        csv_path=Prism_path+"\\Past_Request\\BloombergFCF"+str(i)+".csv"
#        message= "Bloomberg: " + FreeCashFlow_setFreeCashFlow_from_csv(csv_path)
#    
#
#   
#    path_ticker = r'C:\Users\sjin\Documents\Local_Code\Database\AllEquityTickers.csv'
#    tickerlist = pd.read_csv(path_ticker)['Tickers'].dropna().tolist()
    dbName = Last_DB()
    conn = sqlite3.connect(dbName)
    db_cursor=conn.cursor() 
    message =""
#    tickerId =Asset_getAssetsId_fromTicker(tickerlist, db_cursor)   
#    request = ""    
#    
#      
#    request = ""
#    for tickerid in tickerId:
#         request+= "Asset_ID = " +str(tickerid) +" OR "
#    request=request[0:len(request)-1-3]
    db_cursor.execute("Delete from FreeCashFlow") 
    #where " +request)
    conn.commit()
    conn.close() 
    
    
    csv_path=Prism_path+"\\New_Request\\"        
    for k in range(1,7):
        csv_path_file=csv_path+"BloombergFCF"+str(k) + ".csv"
        message += "Bloomberg: "+FreeCashFlow_setFreeCashFlow_from_csv(csv_path_file)
        message +="\n"
#    message+="\n"
#    csv_path=Prism_path+"\\USCashFreeCashFlow.csv"
#    message+= "US assets: "+FreeCashFlow_setFreeCashFlow_from_csv(csv_path)
    return message  
    
    
def DB_FillDividends(Prism_path):
    
#    xml_path=Prism_path  + '\\response_Request_Dividends.xml'
#    tree = xml.etree.ElementTree.parse(xml_path).getroot()
#    dct_Dividends =DividendsXML(tree)
#    
#    tickers = list(dct_Dividends.keys())
#    icount = len(tickers)
#    for ticker in tickers:
#        message_temp = Dividends_setAssetDividend(dct_Dividends[ticker])
#        
#    message = "Dividends: " + str(icount) +" insertions done."
    for i in range(1,18):
        csv_path=Prism_path+"\\Past_Request\\BloombergDividends"+str(i)+".csv"
        message= "Bloomberg: "+Dividends_setAssetDividend_from_csv(csv_path)
    
    dbName = Last_DB()
    conn = sqlite3.connect(dbName)
    db_cursor=conn.cursor() 
    
    path_ticker = r'C:\Users\sjin\Documents\Local_Code\Database\AllEquityTickers.csv'
    tickerlist = pd.read_csv(path_ticker)['Tickers'].dropna().tolist()
    tickerId =Asset_getAssetsId_fromTicker(tickerlist, db_cursor)     
    request = ""
    for tickerid in tickerId:
         request+= "Asset_ID = " +str(tickerid) +" OR "
    request=request[0:len(request)-1-3]
    db_cursor.execute("Delete from Dividends where " +request)
    conn.commit()
    conn.close() 
    
    csv_path=Prism_path+"\\New_Request\\"  
    for k in range(1,7):
        csv_path_file=csv_path+"BloombergDividends"+str(k) + ".csv"
        message += "Dividends: "+Dividends_setAssetDividend_from_csv(csv_path_file)
        message +="\n"
#    csv_path=Prism_path+"\EuropeCashDividends.csv"
#    message= "European assets: "+Dividends_setAssetDividend_from_csv(csv_path)
#    message+="\n"    
#    csv_path=Prism_path+"\\USCashDividends.csv"
#    message+= "US assets: "+Dividends_setAssetDividend_from_csv(csv_path)
    return message
    
#**********************************
#                                 ASSET TABLE FUNCTIONS
#**********************************
def Asset_TableCreation():
    dbName = Last_DB()
    conn = sqlite3.connect(dbName)
    db_cursor=conn.cursor() 
    """ tests if the table already exists """
    res = db_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Asset'")
    result = []
    for elt in res :
        result.append(elt[0])
    if len(result)==0:
        db_cursor.execute('''CREATE TABLE Asset (id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT)''')
        message="Table Asset created."
    else:
        message="Table Asset already exists."
    conn.commit()
    conn.close()
    return message
    

""" this function checks if the tickers exist in the database"""
def Asset_bExistingTicker(ticker_list, db_cursor):
        
    if type(ticker_list)==str:
        ticker_temp= ticker_list
        ticker_list=[]
        ticker_list.append(ticker_temp)
    request=""
    icount = 0 
    dct_ticker_to_id = {}
    for k in range(0,len(ticker_list)):
        asset_ticker=ticker_list[k]
        request += "ticker= '" + str(asset_ticker) + "' or "
        if icount == 500 or k==len(ticker_list)-1:
            request += "ticker= 'end of request'  "
            db_cursor.execute("SELECT ticker,id FROM Asset WHERE " + request)
            response = db_cursor.fetchall()
            if len(response)>0:
                for elt in response :
                     dct_ticker_to_id[elt[0]]=elt[1]
            request =""
            icount= 0
        icount+=1
                     
    bExistingTicker = []
    for ticker in ticker_list:
        try:
            exists = dct_ticker_to_id[ticker]
            bExistingTicker.append(True)
        except KeyError:
            bExistingTicker.append(False)

    return bExistingTicker
    
""" this function insert tickers in the database
    it inserts only if the ticker doesn't exist
"""    
def Asset_setTickers(ticker_list):
    dbName = Last_DB()
    conn = sqlite3.connect(dbName)
    db_cursor=conn.cursor() 
    if type(ticker_list)==str:
        ticker_temp= ticker_list
        ticker_list=[]
        ticker_list.append(ticker_temp)
    bExistingTicker= Asset_bExistingTicker(ticker_list, db_cursor)
    icount = 0
    values = []
    for k in range(0,len(ticker_list)):
        ticker=ticker_list[k]
        bExists=bExistingTicker[k]
        if bExists==False:
            values.append((ticker,))
            icount +=1
    db_cursor.executemany("INSERT INTO Asset (ticker) VALUES (?)",values)
    message = str(icount) + " tickers inserted."
    conn.commit()
    conn.close()
    return message 
    
def Asset_UpdateTickers(ticker_list,ticker_list_updated):  
    dbName = Last_DB()
    conn = sqlite3.connect(dbName)
    db_cursor=conn.cursor() 
    if type(ticker_list)==str:
        ticker_temp= ticker_list
        ticker_list=[]
        ticker_list.append(ticker_temp)
    bExistingTicker= Asset_bExistingTicker(ticker_list, db_cursor)
    icount1 = 0
    icount2 = 0
    for k in range(0,len(ticker_list)):
        ticker = ticker_list[k]
        bExists=bExistingTicker[k]
        tickerUpdated=ticker_list_updated[k]
        if bExists:
            ticker_id = Asset_getAssetsId_fromTicker(ticker, db_cursor)
            db_cursor.execute("UPDATE Asset SET ticker = '"+tickerUpdated + "' WHERE id = '" +str(ticker_id) +"'")
            icount1+=1
        else:
            Asset_setTickers(tickerUpdated)
            icount2+=1   
    message = str(icount) +" tickers updated and " +str(icount2) +" tickers created"
    return message       
    

""" this function returns all the tickers in the database"""
def Asset_getAssetsTicker_All():
    dbName = Last_DB()
    conn = sqlite3.connect(dbName)
    db_cursor=conn.cursor() 
    res=db_cursor.execute("SELECT ticker FROM Asset ")
    Tickers=[]
    for asset in res:
        Tickers.append( asset[0]) 
    conn.close()
    return Tickers 
    
def Asset_getAssetsID_All(db_cursor):
    res=db_cursor.execute("SELECT id FROM Asset ")
    IDs=[]
    for asset in res:
        IDs.append( asset[0] ) 
    return IDs  

""" this function returns tickers given asset_id """   
def Asset_getAssetsTicker_fromID(asset_id_list,db_cursor):
    if type(asset_id_list) is not list:
        asset_id_list_temp = asset_id_list
        asset_id_list=[]
        asset_id_list.append(asset_id_list_temp)
        
    request=""
    icount = 0 
    dct_ticker_to_id = {}
    for k in range(0,len(asset_id_list)):
        asset_ticker=asset_id_list[k]
        request += "ticker= '" + str(asset_ticker) + "' or "
        if icount == 500 or k==len(asset_id_list)-1:
            request += "ticker= 'end of request'  "
            db_cursor.execute("SELECT ticker,id FROM Asset WHERE " + request)
            response = db_cursor.fetchall()
            if len(response)>0:
                for elt in response :
                     dct_ticker_to_id[elt[0]]=elt[1]
            request =""
            icount= 0
        icount+=1
                         
    request=""
    icount = 0     
    request=""
    dct_id_to_ticker={}
    for k in range(0,len(asset_id_list)):
        request += "id= '" + str(asset_id_list[k]) + "' or "
        if icount == 500 or k==len(asset_id_list)-1:
            request += "id= '0'  "
            db_cursor.execute("SELECT id,ticker FROM Asset ")
            response = db_cursor.fetchall() 
            dct_id_to_ticker={}
            if len(response)>0:
                for elt in response:
                    dct_id_to_ticker[elt[0]]=elt[1]
            request = ""
            icount = 0
        icount +=1

    Tickers = [] 
    
    for ids in asset_id_list:
        try:
            ticker = dct_id_to_ticker[ids]
            Tickers.append(ticker)
        except KeyError:
            Tickers.append("")
    return   Tickers

""" this function returns ids given a ticker list """  
def Asset_getAssetsId_fromTicker(ticker_list, db_cursor):
    if type(ticker_list)==str:
        ticker_temp= ticker_list
        ticker_list=[]
        ticker_list.append(ticker_temp)
    request=""
    icount = 0
    
    dct_ticker_to_id = {}
    for k in range(0,len(ticker_list)):
        asset_ticker=ticker_list[k]
        request += "ticker= '" + str(asset_ticker) + "' or "
        if icount ==500 or  k==len(ticker_list)-1 :
            request += "ticker= 'end of request'  "
            db_cursor.execute("SELECT ticker,id FROM Asset WHERE " + request)
            response = db_cursor.fetchall()
            
            if len(response)>0:
                for elt in response :
                     dct_ticker_to_id[elt[0]]=elt[1]
            request=""
            icount=0
        icount+=1
    
    IDs = []
    for ticker in ticker_list:
        try:
            exists = dct_ticker_to_id[ticker]
            IDs.append(exists)
        except KeyError:
            IDs.append("-1")
    return IDs
   

   
#**********************************
#                               PRICE TABLE FUNCTIONS
#**********************************    
def Asset_Price_TableCreation():
    dbName = Last_DB()
    conn = sqlite3.connect(dbName)
    db_cursor=conn.cursor() 
    """ tests if the table already exists """
    res = db_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Asset_Price'")
    result = []
    for elt in res :
        result.append(elt[0])
    if len(result)==0:
        db_cursor.execute('''CREATE TABLE Asset_Price(id INTEGER PRIMARY KEY AUTOINCREMENT, Asset_ID INTEGER, date TEXT, bid REAL, ask REAL, last REAL,open REAL, high REAL, low REAL)''')
        message="Table Asset_Price created."
    else:
        message="Table Asset_Price already exists."
    conn.commit()
    conn.close()
    return message
           

""" the following function insert Historical Price for a ticker list 
    It deletes old date and insert new Data
"""

def Asset_Price_setHistoricalPrice(ticker_list,dfPriceHsty):
    
    
    dct_fields ={}
    dct_fields['date']=['Date','DATE','date','Dates','DATES','dates']
    dct_fields['bid']=['Bid','bid','BID','PX_BID','Px_bid','Px_Bid','PX_bid','PX_Bid']
    dct_fields['ask']=['Ask','ask','ASK','PX_ASK','Px_ask','Px_Ask','PX_ask','PX_Ask']
    dct_fields['last']=['Last','last','PX_LAST','Px_last','Px_Last','PX_last','PX_Last']
    dct_fields['open']=['Open','open','PX_OPEN','Px_open','Px_Open','PX_open','PX_Open']
    dct_fields['high']=['High','high','PX_HIGH','Px_high','Px_High','PX_high','PX_High']
    dct_fields['low']=['Low','low','PX_LOW','Px_low','Px_Low','PX_low','PX_Low']
    
    modified_columns = []
    selection = []
    for column in dfPriceHsty.columns:
        position = column.find('.')
        if position != -1:
            field= column[0:position]
            ticker = column[position+1:len(column)]
            icount = 0
            for key in dct_fields.keys():
                
                if field in dct_fields[key]:
                    modified_columns.append(key +"."+ ticker)
                    selection.append(key +"."+ ticker)
                    icount+=1
            if icount == 0:
                modified_columns.append(column)
        else:
            modified_columns.append(column) 
                    
    dfPriceHsty.columns = modified_columns
    
    dbName = Last_DB()
    conn = sqlite3.connect(dbName)
    db_cursor=conn.cursor() 
    
    if type(ticker_list)==str:
        ticker_temp= ticker_list
        ticker_list=[]
        ticker_list.append(ticker_temp) 
    message =Asset_setTickers(ticker_list)
    ticker_id_list=Asset_getAssetsId_fromTicker(ticker_list, db_cursor)
    icount = 0
    
    values = []
    for k in range(0,len(ticker_id_list)):
        ticker=ticker_list[k]
        ticker_id=ticker_id_list[k]
        columns = []
        
        """ date management """
        if "date."+ ticker in dfPriceHsty.columns:
            columns.append("date."+ ticker)
        """ BID """
        if "bid."+ticker in dfPriceHsty.columns:
            columns.append("bid."+ ticker)
        """ last """
        if "last."+ticker in dfPriceHsty.columns:
            columns.append("last."+ ticker)
        """ last """
        if "ask."+ticker in dfPriceHsty.columns:
            columns.append("ask."+ ticker)
        """ open """
        if "open."+ticker in dfPriceHsty.columns:
            columns.append("open."+ ticker)
        """ high """
        if "high."+ticker in dfPriceHsty.columns:
            columns.append("high."+ ticker)
        """ low """
        if "low."+ticker in dfPriceHsty.columns:
            columns.append("low."+ ticker)
            
        dftemp=dfPriceHsty[columns]
                    
        if 'last.'+ticker in dftemp.columns:
            icount+=1
            dftemp=dftemp[np.isnan(dftemp['last.'+ticker])==False]
            required = ['date.'+ticker,'last.'+ticker,'bid.'+ticker,'ask.'+ticker,'open.'+ticker,'high.'+ticker,'low.'+ticker]
            for elt in required:
                if elt not in dftemp.columns:
                    dftemp[elt]=""     
            dftemp.set_index('date.'+ticker,inplace = True)
            if len(dftemp)>2:
                """ date management """
                dftemp.index=index_to_datetime(dftemp.index)
                """ delete old data if existing"""
    #            db_cursor.execute("DELETE FROM Asset_Price WHERE Asset_ID = ' " +str(ticker_id) +"'; ")
                """ insert new data """ 
                vdate = dftemp.index.tolist()
                vbid = dftemp['bid.'+ticker].fillna("").tolist()
                vlast=dftemp['last.'+ticker].fillna("").tolist()
                vask=dftemp['ask.'+ticker].fillna("").tolist()
                vdopen=dftemp['open.'+ticker].fillna("").tolist()
                vhigh=dftemp['high.'+ticker].fillna("").tolist()
                vlow = dftemp['low.'+ticker].fillna("").tolist()
                for k in range(0,len(vbid)):
                    values.append((ticker_id,str(vdate[k]),vbid[k],vlast[k],vask[k],vdopen[k],vhigh[k],vlow[k]))
    db_cursor.executemany("INSERT INTO Asset_Price (Asset_ID,date,bid,last,ask,open,high,low) VALUES (?,?,?,?,?,?,?,?)", values)
    message =str(icount)+" insertion(s) done"
    conn.commit()
    conn.close()
    return message
    

def Asset_Price_getPrices(ticker_list,field_list,dtBegin,dtEnd,bIntersection,calendar=np.nan):
    dfPrice = DB_BDH_Call(ticker_list,field_list,dtBegin,dtEnd,bIntersection,calendar)
    return dfPrice
    
def Asset_Price_DeletePrice(ticker_list):
    dbName = Last_DB()
    conn = sqlite3.connect(dbName)
    db_cursor=conn.cursor()
    if type(ticker_list)==str:
        ticker_temp= ticker_list
        ticker_list=[]
        ticker_list.append(ticker_temp)
    icount = 0
    ticker_id_list=Asset_getAssetsId_fromTicker(ticker_list, db_cursor)
    
    
    for ticker_id in ticker_id_list:
        db_cursor.execute("DELETE FROM Asset_Price  WHERE Asset_ID =  " + str(ticker_id))
        icount+=1
    conn.commit()
    conn.close()
    return str(icount) + " historical prices deleted"


def Asset_Price_setHistoricalPrice_from_csv(csv_path):
    dfPriceHsty = pd.read_csv(csv_path)
    ticker_list=dfPriceHsty['Tickers'].dropna().tolist()
    del dfPriceHsty['Tickers']
    
    message = Asset_Price_setHistoricalPrice(ticker_list,dfPriceHsty)
    return message
    
    
    
    
#**********************************
#                                 Calendar TABLE FUNCTIONS
#**********************************    

def Calendar_TableCreation():
    dbName = Last_DB()
    conn = sqlite3.connect(dbName)
    db_cursor=conn.cursor() 
    """ tests if the table already exists """
    res = db_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='TradingVolume'")
    result = []
    for elt in res :
        result.append(elt[0])
    if len(result)==0:
        db_cursor.execute('''CREATE TABLE Calendar(id INTEGER PRIMARY KEY AUTOINCREMENT, Asset_ID INTEGER, date TEXT)''')
        message="Table Calendar created."
    else:
        message="Table Calendar already exists."
    conn.commit()
    conn.close()
    return message
    
def Calendar_setCalendar(ticker_list,dfPriceHsty):
    dbName = Last_DB()
    conn = sqlite3.connect(dbName)
    db_cursor=conn.cursor() 
    
    if type(ticker_list)==str:
        ticker_temp= ticker_list
        ticker_list=[]
        ticker_list.append(ticker_temp)
        
    message =Asset_setTickers(ticker_list)
    ticker_id_list=Asset_getAssetsId_fromTicker(ticker_list, db_cursor)
    icount = 0
    
    for k in range(0,len(ticker_id_list)):
        ticker=ticker_list[k]
        ticker_id=ticker_id_list[k]
        columns = []
        dct_column = {}
        if "Calendar."+ ticker in dfPriceHsty.columns:
            columns.append("Calendar."+ ticker)
            dct_column["Calendar."+ ticker]='calendar'
        if "calendar."+ ticker in dfPriceHsty.columns:
            columns.append("calendar."+ ticker)
            dct_column["Calendar."+ ticker]='calendar'
     
        dftemp=dfPriceHsty[columns]
        names = []
        for i in range(0,len(columns)):
            names.append(dct_column[ dftemp.columns[i]])
        dftemp.columns = names
                    
        if 'calendar' in dftemp.columns:
            icount+=1
            dftemp=dftemp[np.isnan(dftemp['calendar'])==False]  
            """ date management """
            dftemp['calendar']=index_to_datetime(dftemp['calendar'])
            """ delete old data if existing"""
            db_cursor.execute("DELETE FROM Calendar WHERE Asset_ID = ' " +str(ticker_id) +"'; ")
            """ insert new data """
            values = []
            for date in dftemp['calendar']:
                values.append((ticker_id,str(date)))
            db_cursor.executemany("INSERT INTO Calendar (Asset_ID,date) VALUES (?,?)",values )
    message =str(icount)+" insertion(s) done"
    conn.commit()
    conn.close()
    return message      
    
def Calendar_setCalendar_from_csv(csv_path):   
    dfHsty = pd.read_csv(csv_path)
    ticker_list=dfHsty['Tickers'].dropna().tolist()
    del dfHsty['Tickers']
    message =Calendar_setCalendar(ticker_list,dfPriceHsty)
    return message 

def Calendar_getCalendar(ticker_list):   
    dbName = Last_DB()
    conn = sqlite3.connect(dbName)
    db_cursor=conn.cursor() 
    
    if type(ticker_list)==str:
        ticker_temp= ticker_list
        ticker_list=[]
        ticker_list.append(ticker_temp)
        
    Ticker_ID=Asset_getAssetsId_fromTicker(ticker_list, db_cursor)    
    dfCalendar = pd.DataFrame()
    for k in range(0,len(Ticker_ID)):
        assetid= Ticker_ID[k]
        res=db_cursor.execute("SELECT * FROM Calendar Where Asset_ID = '"+str(assetid) +"'")
        dates=[]
        for resultat in res:
           dates.append(resultat[2])
        result = pd.DataFrame({'Calendar.'+str(ticker_list[k]):dates})
        if len(dfCalendar) ==0:
            dfCalendar=result
        else:
            if len(result)>0:
                frames=[ dfCalendar,result]
                dfCalendar = pd.concat(frames,axis=1)             
    return dfCalendar    
        
        

    
# *********************************************
#                  Trading Volume
# *********************************************
    
def TradingVolume_TableCreation():
    dbName = Last_DB()
    conn = sqlite3.connect(dbName)
    db_cursor=conn.cursor() 
    """ tests if the table already exists """
    res = db_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='TradingVolume'")
    result = []
    for elt in res:
        result.append(elt[0])
    if len(result)==0:
        db_cursor.execute('''CREATE TABLE TradingVolume(id INTEGER PRIMARY KEY AUTOINCREMENT, Asset_ID INTEGER, date TEXT,volume REAL)''')
        message="Table TradingVolume created."
    else:
        message="Table TradingVolume already exists."
    conn.commit()
    conn.close()
    return message
    
def TradingVolume_setHistoricalVolume(ticker_list,dfVolumeHsty):
    dbName = Last_DB()
    conn = sqlite3.connect(dbName)
    db_cursor=conn.cursor() 
    
    if type(ticker_list)==str:
        ticker_temp= ticker_list
        ticker_list=[]
        ticker_list.append(ticker_temp) 
        
    message =Asset_setTickers(ticker_list)
    ticker_id_list=Asset_getAssetsId_fromTicker(ticker_list, db_cursor)
    icount = 0
    
    for k in range(0,len(ticker_id_list)):
        ticker=ticker_list[k]
        ticker_id=ticker_id_list[k]
        columns = []
        dct_column = {}
        if "Date."+ ticker in dfVolumeHsty.columns:
            columns.append("Date."+ ticker)
            dct_column["Date."+ ticker]='date'
        if "date."+ ticker in dfVolumeHsty.columns:
            columns.append("date."+ ticker)
            dct_column["date."+ ticker]='date'
        if "Volume."+ticker in dfVolumeHsty.columns:
            columns.append("Volume."+ ticker)  
            dct_column["Volume."+ ticker]='volume'
        if "volume."+ticker in dfVolumeHsty.columns:
            columns.append("volume."+ ticker)
            dct_column["volume."+ ticker]='volume'

        dftemp=dfVolumeHsty[columns]
        names = []
        for i in range(0,len(columns)):
            names.append(dct_column[ dftemp.columns[i]])
        dftemp.columns = names
                    
        if 'volume' in dftemp.columns:
            icount+=1
            dftemp=dftemp[np.isnan(dftemp['volume'])==False]     
            dftemp.set_index('date',inplace = True)
            if len(dftemp) >2 :
                """ date management """
                dftemp.index=index_to_datetime(dftemp.index)
                """ delete old data if existing"""
    #            db_cursor.execute("DELETE FROM TradingVolume WHERE Asset_ID = ' " +str(ticker_id) +"'; ")
                """ insert new data """ 
                values = []
                vvolume=dftemp['volume'].tolist()
                vdate=dftemp.index.tolist()
                
                for k in range(0,len(vdate)):
                    values.append((ticker_id,str(vdate[k]),vvolume[k]))
                db_cursor.executemany("INSERT INTO TradingVolume (Asset_ID,date,volume) VALUES (?,?,?)", values)
    message =str(icount)+" insertion(s) done"
    conn.commit()
    conn.close()
    return message    
    
def TradingVolume_setHistoricalVolume_from_csv(csv_path):   
    dfVolumeHsty = pd.read_csv(csv_path)
    ticker_list=dfVolumeHsty['Tickers'].dropna().tolist()
    del dfVolumeHsty['Tickers']
    message =TradingVolume_setHistoricalVolume(ticker_list,dfVolumeHsty)
    return message 
    

def TradingVolume_getVolume(ticker_list,dtBegin,dtEnd,bIntersection,calendar=np.nan):
    dfPrice = DB_BDH_Call(ticker_list,'Volume',dtBegin,dtEnd,bIntersection,calendar)
    return dfPrice    
    
#**********************************
#               BASKET TABLE FUNCTIONS
#**********************************   
def Basket_TableCreation():
    dbName = Last_DB()
    conn = sqlite3.connect(dbName)
    db_cursor=conn.cursor() 
    """ tests if the table already exists """
    res = db_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Basket'")
    result = []
    for elt in res :
        result.append(elt[0])
    if len(result)==0:
        db_cursor.execute('''CREATE TABLE Basket(id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, underlying TEXT)''')
        message="Table Basket created."
    else:
        message="Table Basket already exists."
    conn.commit()
    conn.close()
    return message
           

def Basket_bExistingBasket(underlying_list,dates_list, db_cursor):
    if type(underlying_list)==str:
        ticker_temp = underlying_list
        underlying_list=[]
        underlying_list.append( ticker_temp )
        
    if type(dates_list)==str:
        date_temp = dates_list
        dates_list=[]
        dates_list.append( date_temp )
        
    request = ""
    dct_underlying_to_id={}
    icount = 0
    for k in range(0,len(underlying_list)):
        underlying=underlying_list[k]
        date = dates_list[k]
        request += "( underlying ='" +str(underlying) +"' AND date = '" +str(date) + "') or "
        if icount ==100 or k == len(underlying_list)-1 :
            request +=  "(underlying ='impossible')" 
            db_cursor.execute("SELECT underlying,id FROM Basket WHERE " + request)
            response = db_cursor.fetchall()
            if len(response)>0:
                for elt in response:
                    dct_underlying_to_id[elt[0]]=elt[1] 
            request =""
            icount = 0
        icount+=1


    bexists=[] 
    
    for ticker in underlying_list:
        try:
            ids= dct_underlying_to_id[ticker]
            bexists.append(True)
        except KeyError:
            bexists.append(False)
    
    return bexists
           
def Basket_setBasket(underlying_list,dates_list):
    """
    dctBasket is a dictionnary. Each key corresponds to the basket on a specific date.
    """
    if type(underlying_list)==str:
        ticker_temp = underlying_list
        underlying_list=[]
        underlying_list.append( ticker_temp )
        
    if type(dates_list) is not list:
        date_temp = dates_list
        dates_list=[]
        dates_list.append( date_temp )
    """ connect to database"""
    dbName = Last_DB()
    conn = sqlite3.connect(dbName)
    db_cursor=conn.cursor()   
    
    icount=0

    """ get all basket names and evaluation dates """
    dates=[]
    underlyings=[]
    for k in range(0,len(underlying_list)):
        date=str(dates_list[k])
        underlying = underlying_list[k]
        dates.append(date)
        underlyings.append(underlying)
        
    """ already existing ? """
    bExisting =Basket_bExistingBasket(underlyings,dates, db_cursor)
    
    values = []
    for k in range(0,len( bExisting)):
        if not bExisting[k]:
            icount +=1
            values.append((dates[k],underlyings[k]))
            
    db_cursor.executemany("INSERT INTO Basket (date,underlying) VALUES (?,?)",values)
            
    conn.commit()
    conn.close() 
    message = str(icount) +" Baskets created." 
    return message
    

    
def Basket_getBasketId(underlying_list,dates_list, db_cursor):   
    if type(underlying_list)==str:
        ticker_temp = underlying_list
        underlying_list=[]
        underlying_list.append( ticker_temp )
        
    if type(dates_list)==str:
        date_temp = dates_list
        dates_list=[]
        dates_list.append( date_temp )
        
    request = ""
    dct_underlying_to_id={}
    icount = 0
    for k in range(0,len(underlying_list)):
        underlying=underlying_list[k]
        date = dates_list[k]
        request += "( underlying ='" +str(underlying) +"' AND date = '" +str(date) + "') or "
        if icount ==500 or k == len(underlying_list)-1 :
            request +=  "(underlying ='impossible')"   
            db_cursor.execute("SELECT underlying,id,date FROM Basket WHERE " + request)
            response = db_cursor.fetchall()
            if len(response)>0:
                for elt in response:
                    dct_underlying_to_id[(elt[0],elt[2])]=elt[1]
            icount =0
            request = ""
        icount+=1
                
    BasketId  =[] 
    
    for k in range(0,len(underlying_list)):
        ticker = underlying_list[k]
        date= dates_list[k]
        try:
            ids= dct_underlying_to_id[(ticker,date)]
            BasketId.append(ids)
        except KeyError:
            BasketId.append(np.nan)
    return  BasketId   
    
    
def Basket_getBasketId_Approx(underlying_list,dates_list, db_cursor):   
    
    if type(underlying_list)==str:
        ticker_temp = underlying_list
        underlying_list=[]
        underlying_list.append( ticker_temp )
        
    if type(dates_list)==str:
        date_temp = dates_list
        dates_list=[]
        dates_list.append( date_temp )
        
    
#    dct_underlying_to_id={}
#   
    BasketId  =[] 
    for j in range (0,len(dates_list)):
        for k in range(0,len(underlying_list)):
            request=""
            underlying=underlying_list[k]
            date = str(dates_list[j])[0:10]
            request += " underlying ='" +str(underlying) +"' AND date <= '" +str(date)  + "' ORDER BY date DESC LIMIT 1"
            db_cursor.execute("SELECT underlying,id,date FROM Basket WHERE " + request)
            
    #        dates_found=[]
            ids_found = []
            response = db_cursor.fetchall()
            for elt in response:
    #            dates_found.append(elt[2])
                ids_found.append(elt[1])
            if len(ids_found)    > 0 :
                BasketId.append(int(ids_found[0]))
            else:
                BasketId.append(np.nan)   
            
#        df=pd.DataFrame({'Dates':dates_found,'ids_found':ids_found})
#        df=df.sort('Dates',ascending = True)
#        
#        if len(df)>0:
#            BasketId.append(int(df['ids_found'][len(df)-1]))
#        else:
#            BasketId.append(np.nan)

    return  BasketId       
    
    
    
def Basket_getBasketUnderlying(basket_ID_list,db_cursor):
    if type(basket_ID_list) is not list :
        basket_ID_list_temp = basket_ID_list
        basket_ID_list=[]
        basket_ID_list.append(basket_ID_list_temp)
    BasketUnderlying=[]   
    for k in range(0,len(basket_ID_list)):
        basket_ID=str(basket_ID_list[k])   
        db_cursor.execute("SELECT underlying FROM Basket WHERE id='" + basket_ID + "'")
        try:
            BasketUnderlying.append(db_cursor.fetchall()[0][0] )
        except:
            BasketUnderlying.append(np.nan)
    return  BasketUnderlying
    
    
#**********************************
#               BASKET Components TABLE FUNCTIONS
#**********************************
def BasketComponents_TableCreation():
    dbName = Last_DB()
    conn = sqlite3.connect(dbName)
    db_cursor=conn.cursor() 
    
    """ tests if the table already exists """
    res = db_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Basket_Components'")
    result = []
    for elt in res :
        result.append(elt[0])
    if len(result)==0:
        db_cursor.execute('''CREATE TABLE Basket_Components(id INTEGER PRIMARY KEY AUTOINCREMENT,Basket_ID INTEGER, Asset_ID INTEGER, Asset_weight REAL, Asset_WeightCoefficient REAL)''')
        message="Table BasketComponents created."
    else:
        message="Table BasketComponents already exists."
    conn.commit()
    conn.close()
    return message


def Basket_set_BasketComponents(dct_Underlyings):
    """
    dct_Underlyings is a dictionnary. Each key corresponds to the basket on a specific date.
    """
    
    """ connect to database"""
    dbName = Last_DB()
    conn = sqlite3.connect(dbName)
    db_cursor=conn.cursor()   
    icount=0
    """ get all basket names and evaluation dates """
    values =[]
    """ set all baskets """
    underlyings=[]
    dates=[]
    for underlying in dct_Underlyings.keys():
        for date in dct_Underlyings[underlying].keys():
            underlyings.append(underlying)
            dates.append(str(date))     
    message = Basket_setBasket(underlyings,dates)
    
    """ Get Basket IDs """
    Basket_IDs = Basket_getBasketId( underlyings,dates, db_cursor )
    
    """ delete if existing Baskets """
#    values = []
#    for k in range(0,len(Basket_IDs)):
#        values.append((str(Basket_IDs[k]),))
#    db_cursor.executemany("DELETE FROM Basket_Components where Basket_ID = ?" ,values )    
#    conn.commit()    
    
    values = [] 
    k=0
    
    all_tickers = []
    
    for underlying in dct_Underlyings.keys():
        for date in dct_Underlyings[underlying].keys():       
            dfBasketComponents = dct_Underlyings[underlying][date]
            Components=dfBasketComponents['Ticker'].tolist()
            for elt in Components:
                all_tickers.append(elt)
                
    all_tickers=list(set(all_tickers)) 
    message =Asset_setTickers(all_tickers)
    Ticker_ID=Asset_getAssetsId_fromTicker(all_tickers, db_cursor )
    
    dct_ticker_Id = {}
    for k in range(0,len(Ticker_ID)) :
        dct_ticker_Id[all_tickers[k]]=str(Ticker_ID[k])
    
    values = []  
    for underlying in dct_Underlyings.keys():
        k=0
        for date in dct_Underlyings[underlying].keys(): 
            dfBasketComponents = dct_Underlyings[underlying][date]            
            Components=dfBasketComponents['Ticker'].tolist()
            for i in range(0,len(Components)):                
                ticker_id = dct_ticker_Id[Components[i]]
                Asset_weight=dfBasketComponents['Weight'][i]
                Asset_WeightCoefficient=dfBasketComponents['WeightCoefficient'][i]
                values.append((str(Basket_IDs[icount]),ticker_id,Asset_weight,Asset_WeightCoefficient))
            icount+=1
        db_cursor.executemany("INSERT INTO Basket_Components (Basket_ID, Asset_ID,Asset_weight,Asset_WeightCoefficient) VALUES (?,?,?,?)",values ) 
    message += "\n" +" "+ str(icount)+" Basket inserted."                       
    conn.commit()
    conn.close()
    return message
    
#def Basket_set_BasketComponents_from_csv(csv_path):
#    """ connect to database"""
#    dbName = Last_DB()
#    conn = sqlite3.connect(dbName)
#    db_cursor=conn.cursor()   
#    
#    """ get data from csv"""
#    
#    dfData=pd.read_csv(csv_path)
#    columns = dfData.columns.tolist()
#    columns[0]=str(columns[0])+"."
#    dates_list=index_to_datetime(dfData.loc[dfData.index[0]]).tolist()
#    for k in range(0,len(columns)):
#        underlying = columns[k][0:columns[k].find('.')]
#        date = str(dates_list[k])
#        
#        dfComponents = dfData[columns[k]].dropna().tolist()
#        dfComponents.pop(0)
#        Ticker_ID=Asset_getAssetsId_fromTicker( Components, db_cursor )
#        for i in range(1,len(dfComponents)):
#            
#        
#       
#    
##    dct_Underlyings={}
##    underlyings = dfData.columns.tolist()
##    
#    for underlying in underlyings:
#        
#        
#    """ extract data """
#    
#    baskets[0]=str(baskets[0])+"."
#    dfData.columns=baskets
#    baskets = dfData.columns
#
#    baskets=baskets.tolist()
#    
#    
#    bExisting =Basket_bExistingBasket(baskets,dates_list, db_cursor)    
#        
#    
#    message = Basket_set_BasketComponents(dfData)
#    return message

def Basket_get_BasketComponents(underlying_list,dates_list): 
    """ connect to database"""
    dbName = Last_DB()
    conn = sqlite3.connect(dbName)
    db_cursor=conn.cursor()  
    if type(underlying_list)==str:
        underlying_list_temp= underlying_list
        underlying_list=[]
        underlying_list.append(underlying_list_temp)
        
    if type(dates_list)!=list:
        temp=[]
        temp.append(dates_list)
        dates_list=temp
        
    """ get baskets ids """
    for j in range(0,len(underlying_list)):
        Basket_id =Basket_getBasketId_Approx(underlying_list[j],dates_list, db_cursor)
        Baskets = pd.DataFrame()
        for k in range(0,len(Basket_id)):
            basketID =Basket_id[k]
            date = dates_list[k]
            underlying = underlying_list[j]
            Components=[]
            if basketID is not np.nan:
                basketID=str(basketID)
                res=db_cursor.execute("SELECT Asset_ID FROM Basket_Components WHERE Basket_ID='" + basketID + "'")
                for asset_id in res:
                    Components.append(asset_id[0])
                Components=Asset_getAssetsTicker_fromID(Components, db_cursor)
            Components.insert(0,date)
            Basketstemp=pd.DataFrame({underlying:Components})
            if len(Baskets)>0:
                 frames = [Baskets, Basketstemp]
                 Baskets=pd.concat(frames,axis=1) 
            else:
                Baskets=Basketstemp
    Baskets.columns=dates_list
    conn.close()        
    return Baskets   
   

# *********************************************
#                   Dividends table
# *********************************************

def Dividends_TableCreation():
    dbName = Last_DB()
    conn = sqlite3.connect(dbName)
    db_cursor=conn.cursor() 
    """ tests if the table already exists """
    res = db_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Dividends'")
    result = []
    for elt in res :
        result.append(elt[0])
    if len(result)==0:
        db_cursor.execute('''CREATE TABLE Dividends(id INTEGER PRIMARY KEY AUTOINCREMENT, Asset_ID INTEGER, DeclarationDate TEXT,ExecutionDate TEXT,SettlementDate TEXT,PaymentDate TEXT, DividendGrossValue REAL, DividendFrequency TEXT , DividendType TEXT)''')
        message="Table Dividends created."
    else:
        message="Table Dividends already exists."
    conn.commit()
    conn.close()
    return message

def Dividends_setAssetDividend(dfData,ticker_list):
    if type(ticker_list)==str:
        ticker_list=ticker_list.split()
    """ connect to database"""
    dbName = Last_DB()
    conn = sqlite3.connect(dbName)
    db_cursor=conn.cursor()   
    """ extract data """
    message =Asset_setTickers(ticker_list)
    Ticker_ID=Asset_getAssetsId_fromTicker(ticker_list, db_cursor)
    icount = 0
    values = []
    for k in range(0,len(ticker_list)):
        ticker = ticker_list[k]
        ticker_id = Ticker_ID[k]
        dfDividends = pd.DataFrame({'DeclarationDate.'+str(ticker):dfData['DeclarationDate.'+str(ticker)].dropna().tolist()})
        if len(dfDividends>0):
            
            try:
                dfDividends['DeclarationDate.'+str(ticker)]=index_to_datetime(dfDividends['DeclarationDate.'+str(ticker)])
            except KeyError:
                dfDividends['DeclarationDate.'+str(ticker)]= np.nan
            try:
                dfDividends['ExecutionDate.'+str(ticker)]=index_to_datetime(dfData['ExecutionDate.'+str(ticker)].fillna(float(50000)))
            except KeyError:
                dfDividends['ExecutionDate.'+str(ticker)]=np.nan
            try:
                dfDividends['SettlementDate.'+str(ticker)]=index_to_datetime(dfData['SettlementDate.'+str(ticker)].fillna(float(50000)))
            except KeyError:
                dfDividends['SettlementDate.'+str(ticker)]=np.nan
                
            try:
                dfDividends['PaymentDate.'+str(ticker)]=index_to_datetime(dfData['PaymentDate.'+str(ticker)].fillna(float(50000)))
            except KeyError:
                dfDividends['PaymentDate.'+str(ticker)]=np.nan
                
            try:
                dfDividends['DividendGrossValue.'+str(ticker)]=dfData['DividendGrossValue.'+str(ticker)]
            except KeyError:
                dfDividends['DividendGrossValue.'+str(ticker)]=np.nan
                
            try:
                dfDividends['DividendFrequency.'+str(ticker)]=dfData['DividendFrequency.'+str(ticker)]
            except KeyError:
                dfDividends['DividendFrequency.'+str(ticker)]=np.nan
                
            try:
                dfDividends['DividendType.'+str(ticker)]=dfData['DividendType.'+str(ticker)]
            except KeyError:
                dfDividends['DividendType.'+str(ticker)]=np.nan                
                
            """ delete existing data """
#            db_cursor.execute("DELETE FROM Dividends WHERE Asset_ID = ' " +str(ticker_id) +"'")   
            for i in range(0,len(dfDividends)):
                DeclarationDate=str(dfDividends.loc[dfDividends.index[i],'DeclarationDate.'+str(ticker)])
                try:
                    ExecutionDate=str(dfDividends.loc[dfDividends.index[i],'ExecutionDate.'+str(ticker)])
                except TypeError:
                    ExecutionDate=DeclarationDate
                try:
                    SettlementDate=str(dfDividends.loc[dfDividends.index[i],'SettlementDate.'+str(ticker)])
                except TypeError:
                    SettlementDate=ExecutionDate
                try:    
                    PaymentDate=str(dfDividends.loc[dfDividends.index[i],'PaymentDate.'+str(ticker)])
                except TypeError:
                    PaymentDate=ExecutionDate    
                try:      
                    DividendGrossValue=float(dfDividends.loc[dfDividends.index[i],'DividendGrossValue.'+str(ticker)])
                except TypeError:
                    DividendGrossValue=0.0
                try:      
                    DividendFrequency=dfDividends.loc[dfDividends.index[i],'DividendFrequency.'+str(ticker)]
                except TypeError:
                    DividendFrequency=""
                    
                try:      
                    DividendType=dfDividends.loc[dfDividends.index[i],'DividendType.'+str(ticker)]
                except TypeError:
                    DividendType=""
                    
                values.append((ticker_id,DeclarationDate,ExecutionDate,SettlementDate,PaymentDate,DividendGrossValue,DividendFrequency,DividendType))    
                    
            icount+=1
    db_cursor.executemany("INSERT INTO Dividends (Asset_ID,DeclarationDate,ExecutionDate,SettlementDate,PaymentDate,DividendGrossValue,DividendFrequency,DividendType) VALUES (?,?,?,?,?,?,?,?)",values)               
    message += "\n" +" "+ str(icount)+" Dividends inserted."   
    conn.commit()
    conn.close()
    return message

def Dividends_setAssetDividend_from_csv(csv_path):
    dfData=pd.read_csv(csv_path)
    ticker_list=dfData['Tickers'].dropna().tolist()
    message = Dividends_setAssetDividend(dfData,ticker_list)
    return message
#
#def Dividends_getAssetDividends(ticker_list,DateBegin=datetime.datetime(1900,1,1),DateEnd=datetime.datetime.today()):
#    """ connect to database"""
#    dbName = Last_DB()
#    conn = sqlite3.connect(dbName)
#    db_cursor=conn.cursor()   
#    if type(ticker_list)==str:
#        ticker_temp= ticker_list
#        ticker_list=[]
#        ticker_list.append(ticker_temp)
#    
#    Ticker_ID=Asset_getAssetsId_fromTicker(ticker_list, db_cursor)
#    dfDvidends = pd.DataFrame()
#    for k in range(0,len(ticker_list)):
#        assetid= Ticker_ID[k]
#        res=db_cursor.execute("SELECT * FROM Dividends Where ExecutionDate >= '" + str(DateBegin) + "' and ExecutionDate <= '" +str(DateEnd) +"' AND Asset_ID = '"+str(assetid) +"'")
#        DeclarationDate=[]
#        ExecutionDate=[]
#        SettlementDate=[]
#        PaymentDate=[]
#        DividendGrossValue=[]
#        DividendNetValue=[]
#        for resultat in res:
#           DeclarationDate.append(resultat[2])
#           ExecutionDate.append(resultat[3])
#           SettlementDate.append(resultat[4])
#           PaymentDate.append(resultat[5])
#           DividendGrossValue.append(resultat[6])
#           DividendNetValue.append(resultat[7])
#        result = pd.DataFrame({'DeclarationDate.'+str(ticker_list[k]):DeclarationDate,'ExecutionDate.'+str(ticker_list[k]):ExecutionDate,'SettlementDate.'+str(ticker_list[k]):SettlementDate,'PaymentDate.'+str(ticker_list[k]):PaymentDate,'DividendGrossValue.'+str(ticker_list[k]):DividendGrossValue,'DividendNetValue.'+str(ticker_list[k]):DividendNetValue})
#        if len(dfDvidends ) ==0:
#            dfDvidends=result
#        else:
#            if len(result)>0:
#                frames=[dfDvidends,result]
#                dfDividends = pd.concat(frames,axis=1)
#    conn.close()
#    return dfDividends
        
def Dividends_getAssetDividend(ticker_list,field_list='Dividends'):
    dfPrice = DB_BDS_Call(ticker_list,field_list)
    return dfPrice
    
    
# *********************************************
#                  BuyBacks table
# *********************************************

def Buybacks_TableCreation():
    dbName = Last_DB()
    conn = sqlite3.connect(dbName)
    db_cursor=conn.cursor() 
    """ tests if the table already exists """
    res = db_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='BuyBacks'")
    result = []
    for elt in res :
        result.append(elt[0])
    if len(result)==0:
        db_cursor.execute('''CREATE TABLE BuyBacks(id INTEGER PRIMARY KEY AUTOINCREMENT, Asset_ID INTEGER, date TEXT,Amount REAL,Currency TEXT, Type TEXT)''')
        message="Table Buybacks created."
    else:
        message="Table Buybacks already exists."
    conn.commit()
    conn.close()
    return message


def Buybacks_setBuybacks_from_csv(csv_path):
    dfData=pd.read_csv(csv_path)
    ticker_list=dfData['Tickers'].dropna().tolist()
    message = Buybacks_setBuybacks(dfData,ticker_list)
    return message
    
def Buybacks_setBuybacks(dfData,ticker_list):
    if type(ticker_list)==str:
        ticker_temp= ticker_list
        ticker_list=[]
        ticker_list.append(ticker_temp)
    """ connect to database"""
    dbName = Last_DB()
    conn = sqlite3.connect(dbName)
    db_cursor=conn.cursor()   
    """ extract data """
    message =Asset_setTickers(ticker_list)
    Ticker_ID=Asset_getAssetsId_fromTicker(ticker_list, db_cursor)
    icount = 0
    
    values = []
    for k in range(0,len(ticker_list)):
        ticker = ticker_list[k]
        ticker_id = Ticker_ID[k]
        dfBuybacks = pd.DataFrame({'date.'+str(ticker):dfData['Date.'+str(ticker)].dropna().tolist()})
        if len(dfBuybacks)>0:
            if type(dfBuybacks['date.'+str(ticker)][0]) != str:
                dfBuybacks['date.'+str(ticker)]=index_to_datetime(dfBuybacks['date.'+str(ticker)])
                dfBuybacks['Amount.'+str(ticker)]=dfData['Amount.'+str(ticker)].fillna(0)
                dfBuybacks['Currency.'+str(ticker)]=dfData['Currency.'+str(ticker)].fillna("None")
                dfBuybacks['Type.'+str(ticker)]=dfData['Type.'+str(ticker)].fillna("None")
            
                """ delete existing data """
        #        db_cursor.execute("DELETE FROM BuyBacks WHERE Asset_ID = ' " +str(ticker_id) +"'")
                
                vdate=dfBuybacks['date.'+str(ticker)].tolist()
                vAmount = dfBuybacks['Amount.'+str(ticker)]
                vCurrency = dfData['Currency.'+str(ticker)]
                vType = dfData['Type.'+str(ticker)]
            
                for k in range(0,len(dfBuybacks)):
                    values.append((ticker_id,str(vdate[k]),vAmount[k],vCurrency[k],vType[k]))
                icount+=1
    db_cursor.executemany("INSERT INTO BuyBacks (Asset_ID,date,Amount,Currency,Type) VALUES (?,?,?,?,?)",values)
    message += "\n" +" "+ str(icount)+" Buybacks inserted."   
    conn.commit()
    conn.close()
    return message
#
#
#def Buybacks_getAssetBuyBacks(ticker_list,DateBegin=datetime.datetime(1900,1,1),DateEnd=datetime.datetime.today()):
#    """ connect to database"""
#    dbName = Last_DB()
#    conn = sqlite3.connect(dbName)
#    db_cursor=conn.cursor()   
#    
#    if type(ticker_list)==str:
#        ticker_temp= ticker_list
#        ticker_list=[]
#        ticker_list.append(ticker_temp)
#    Ticker_ID=Asset_getAssetsId_fromTicker(ticker_list, db_cursor)
#    dfBuybacks = pd.DataFrame()
#    for k in range(0,len(ticker_list)):
#        assetid= Ticker_ID[k]
#        res=db_cursor.execute("SELECT * FROM BuyBacks Where  Asset_ID = '"+str(assetid) +"' AND date >= '" + str(DateBegin) +"' AND date <= '" + str(DateEnd) +"'" )
#        date = []
#        Amount=[]
#        Currency = []
#        Type =[]
#        for resultat in res:
#            date.append(resultat[2])
#            Amount.append(resultat[3])
#            Currency.append(resultat[4])
#            Type.append(resultat[5])
#            dfBuybackstemp= pd.DataFrame({'date.'+str(ticker_list[k]):date,'Amount.'+str(ticker_list[k]):Amount,'Currency.'+str(ticker_list[k]):Currency,'Type.'+str(ticker_list[k]):Type})
#        if len(dfBuybacks)==0:
#            dfBuybacks=dfBuybackstemp
#        else:
#            if len(dfBuybackstemp)>0:
#                frames = [dfBuybacks,dfBuybackstemp]
#                dfBuybacks=pd.concat(frames,axis=1)
#    conn.close()
#    return dfBuybacks
#  
      
def Buybacks_getAssetBuyBacks(ticker_list,field_list='BuyBacks'):
    dfPrice = DB_BDS_Call(ticker_list,field_list)
    return dfPrice
    
def Buybacks_getAssetBuyBacksTop(ticker_list,dateBegin,dateEnd,iTop):
    
    if type(ticker_list) is not list:
        ticker_list_temp = ticker_list
        ticker_list=[]
        ticker_list.append(ticker_list_temp)

    dbName = Last_DB()
    conn = sqlite3.connect(dbName)
    cursor=conn.cursor()   
    
    """
    ticker_list analysis
    """
    
    listId = Asset_getAssetsId_fromTicker(ticker_list, cursor)
  
    ID_Tickers={}
    for k in range(0,len( ticker_list)):
        asset=ticker_list[k]
        assetid=listId[k]
        ID_Tickers[assetid,'ticker']=asset    
    
    
    Asset_BuyBacks_request="Select Asset_ID,date,Amount FROM BuyBacks WHERE "
    buyBacks={}
    sqlCommand=""
    icount_request=0
    for i in range(0,len(listId)):
        sqlCommand = sqlCommand + ' Asset_ID = ' + str(listId[i]) + ' OR '
        icount_request+=1
        
        
        if icount_request>600 or i == len(listId)-1:
            sqlCommand = sqlCommand + "Asset_ID = '0' " 
            tmp = cursor.execute(Asset_BuyBacks_request + sqlCommand)
            
            for elem in tmp:
                if (ID_Tickers[elem[0],'ticker'] in buyBacks.keys()):
                    buyBacks[ID_Tickers[elem[0],'ticker']].append((elem[0],elem[1],elem[2]))
                else:
                    buyBacks.update({ID_Tickers[elem[0],'ticker'] : [(elem[0],elem[1],elem[2])]}) 
                    
            icount_request=0
            sqlCommand =""    
    Buybacks=[]
    Assets=[]
    
    for k in range(0,len(ticker_list)):
        asset=ticker_list[k]
        Assets.append(asset)
        sum_Buyback = 0.0
        try:
            for l in range(0,len(buyBacks[asset])):
                date = buyBacks[asset][l][1]
                date = date[0:10]
                date =  datetime.datetime.strptime(date,"%Y-%m-%d")
                if date>=dateBegin and date <dateEnd:
                    sum_Buyback+=buyBacks[asset][l][2]  
        except KeyError:
            pass 
        Buybacks.append(sum_Buyback)
    
    dfBuybacks = pd.DataFrame({'Amount':Buybacks}, index = Assets)    
    dfBuybacks=dfBuybacks.sort('Amount',ascending = False)
    
    return dfBuybacks.head(iTop)
    
    
    
# *********************************************
#                  Free Cash FLOW
# *********************************************
def FreeCashFlow_TableCreation():
    dbName = Last_DB()
    conn = sqlite3.connect(dbName)
    db_cursor=conn.cursor() 
    """ tests if the table already exists """
    res = db_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='FreeCashFlow'")
    result = []
    for elt in res :
        result.append(elt[0])
    if len(result)==0:
        db_cursor.execute('''CREATE TABLE FreeCashFlow(id INTEGER PRIMARY KEY AUTOINCREMENT, Asset_ID INTEGER, date TEXT,FCF REAL)''')
        message="Table FreeCashFlow created."
    else:
        message="Table FreeCashFlow already exists."
    conn.commit()
    conn.close()
    return message

def FreeCashFlow_setFreeCashFlow_from_csv(csv_path):
    dfData=pd.read_csv(csv_path)
    ticker_list=dfData['Tickers'].dropna().tolist()
    message = FreeCashFlow_setFreeCashFlow(dfData,ticker_list)
    return message
    
def FreeCashFlow_setFreeCashFlow(dfData,ticker_list):
    if type(ticker_list)==str:
        ticker_list=ticker_list.split()
    """ connect to database"""
    dbName = Last_DB()
    conn = sqlite3.connect(dbName)
    db_cursor=conn.cursor()   
    """ extract data """
    message =Asset_setTickers(ticker_list)
    Ticker_ID=Asset_getAssetsId_fromTicker(ticker_list, db_cursor)
    icount = 0
    values = []
#    delete=[]
    
    for k in range(0,len(ticker_list)):
        ticker = ticker_list[k]
        ticker_id = Ticker_ID[k]
#        delete.append((ticker_id,))
        dfFreeCashFlow = pd.DataFrame({'date.'+str(ticker):dfData['Date.'+str(ticker)].dropna().tolist()})
        
        if len(dfFreeCashFlow['date.'+str(ticker)])>0: 
            dfFreeCashFlow['date.'+str(ticker)]=index_to_datetime(dfFreeCashFlow['date.'+str(ticker)])
            try:
                dfFreeCashFlow['FCF.'+str(ticker)]=dfData['FreeCashFlow.'+str(ticker)]
            except KeyError:
                dfFreeCashFlow['FCF.'+str(ticker)]= np.nan
                
            vdate = dfFreeCashFlow['date.'+str(ticker)].tolist()
            vFCF = dfFreeCashFlow['FCF.'+str(ticker)].fillna(0).tolist()
            
            for i in range(0,len(vdate)):
                values.append((ticker_id,str(vdate[i]),vFCF[i]))
            icount+=1    
    """ delete existing data """
    #db_cursor.executemany("DELETE FROM freeCashFlow WHERE Asset_ID = ? ",delete)    
    db_cursor.executemany("INSERT INTO FreeCashFlow(Asset_ID,date,FCF ) VALUES (?,?,?)",values)

            
    message += "\n" +" "+ str(icount)+" FreeCashFlows inserted."   
    conn.commit()
    conn.close()
    return message
        

def FreeCashFlow_getFreeCashFlow(ticker_list,dtBegin,dtEnd,bIntersection,calendar=np.nan):
    field_list='FreeCashFlow'
    dfPrice = DB_BDH_Call(ticker_list,field_list,dtBegin,dtEnd,bIntersection,calendar)
    return dfPrice
    
def FreeCashFlow_getFreeCashFlowTop(ticker_list,dateEnd,iTop):
    
    if type(ticker_list) is not list:
        ticker_list_temp = ticker_list
        ticker_list=[]
        ticker_list.append(ticker_list_temp)

    dbName = Last_DB()
    conn = sqlite3.connect(dbName)
    cursor=conn.cursor()   
    
    """
    ticker_list analysis
    """
    
    listId = Asset_getAssetsId_fromTicker(ticker_list, cursor)
  
    ID_Tickers={}
    for k in range(0,len( ticker_list)):
        asset=ticker_list[k]
        assetid=listId[k]
        ID_Tickers[assetid,'ticker']=asset    
    
    
    Asset_FCF_request="Select Asset_ID,date,fcf FROM FreeCashFlow WHERE "
    FCF={}
    sqlCommand=""
    icount_request=0
    for i in range(0,len(listId)):
        sqlCommand = sqlCommand + ' Asset_ID = ' + str(listId[i]) + ' OR '
        icount_request+=1

        if icount_request>600 or i == len(listId)-1:
            sqlCommand = sqlCommand + "Asset_ID = '0' " 
            tmp = cursor.execute(Asset_FCF_request + sqlCommand)
            
            for elem in tmp:
                if (ID_Tickers[elem[0],'ticker'] in FCF.keys()):
                    FCF[ID_Tickers[elem[0],'ticker']].append((elem[0],elem[1],elem[2]))
                else:
                    FCF.update({ID_Tickers[elem[0],'ticker'] : [(elem[0],elem[1],elem[2])]}) 
                    
            icount_request=0
            sqlCommand ="" 
            
    fcf_Value=[]
    Assets=[]
    
    for k in range(0,len(ticker_list)):
        fcf_asset=[]
        Date=[]
        asset=ticker_list[k]
        Assets.append(asset)
        try:
            for l in range(0,len(FCF[asset])):
                date = FCF[asset][l][1]
                date = date[0:10]
                date =  datetime.datetime.strptime(date,"%Y-%m-%d")
                Date.append(date)
                fcf_asset.append(FCF[asset][l][2])
        except KeyError:
            pass 
        
        priceHstytemp=pd.DataFrame({'Free_Cash_Flow.'+str(asset):fcf_asset,'Date':Date})   
        
        if len(priceHstytemp) >0:
            priceHstytemp=priceHstytemp.sort('Date',ascending = True)
            priceHstytemp=priceHstytemp[priceHstytemp['Date']<=dateEnd]
            if len(priceHstytemp)>0:
                fcf_Value.append(priceHstytemp.loc[priceHstytemp.index[len(priceHstytemp)-1],'Free_Cash_Flow.'+str(asset)])
            else:
                fcf_Value.append(0)
        else:
            fcf_Value.append(0)
         
    dfFCF = pd.DataFrame({'Amount':fcf_Value}, index = Assets)    
    dfFCF=dfFCF.sort('Amount',ascending = False)
    
    return dfFCF.head(iTop)
    
# *********************************************
#                  Market Cap
# *********************************************
def MarketCap_TableCreation():
    dbName = Last_DB()
    conn = sqlite3.connect(dbName)
    db_cursor=conn.cursor() 
    """ tests if the table already exists """
    res = db_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='MarketCap'")
    result = []
    for elt in res :
        result.append(elt[0])
    if len(result)==0:
        db_cursor.execute('''CREATE TABLE MarketCap(id INTEGER PRIMARY KEY AUTOINCREMENT, Asset_ID INTEGER, date TEXT,MarketCap REAL)''')
        message="Table MarketCap created."
    else:
        message="Table MarketCap already exists."
    conn.commit()
    conn.close()
    return message
    
def MarketCap_setMarketCap_from_csv(csv_path):
    dfData=pd.read_csv(csv_path)
    ticker_list=dfData['Tickers'].dropna().tolist()
    message = MarketCap_setMarketCap(dfData,ticker_list)
    return message
     
def MarketCap_setMarketCap(dfData,ticker_list):
    if type(ticker_list)==str:
        ticker_list_temp = ticker_list
        ticker_list=[]
        ticker_list.append(ticker_list_temp)
    """ connect to database"""
    dbName = Last_DB()
    conn = sqlite3.connect(dbName)
    db_cursor=conn.cursor()   
    """ extract data """
    message =Asset_setTickers(ticker_list)
    Ticker_ID=Asset_getAssetsId_fromTicker(ticker_list, db_cursor)
    icount = 0
    for k in range(0,len(ticker_list)):
        ticker = ticker_list[k]
        ticker_id = Ticker_ID[k]
        dfMarketCap = pd.DataFrame({'date.'+str(ticker):dfData['Date.'+str(ticker)].dropna().tolist()})
        if len( dfMarketCap) > 2:
            dfMarketCap['date.'+str(ticker)]=index_to_datetime(dfMarketCap['date.'+str(ticker)])
            dfMarketCap['MarketCap.'+str(ticker)]=dfData['MarketCap.'+str(ticker)]
    
            """ delete existing data """
    #        db_cursor.execute("DELETE FROM MarketCap WHERE Asset_ID = ' " +str(ticker_id) +"'")
            values = []
            vdate= dfMarketCap['date.'+str(ticker)].tolist()
            vMarketCap=dfMarketCap['MarketCap.'+str(ticker)].fillna("").tolist()
            
            for i in range(0,len(dfMarketCap)):
                values.append((ticker_id,str(vdate[i]),vMarketCap[i]))
                icount+=1
            db_cursor.executemany("INSERT INTO MarketCap(Asset_ID,date,MarketCap ) VALUES (?,?,?)",values)
        
            
    message += "\n" +" "+ str(icount)+" MarketCap inserted."   
    conn.commit()
    conn.close()
    return message
    
    
#def MarketCap_getMarketCap(ticker_list,DateBegin=datetime.datetime(1900,1,1),DateEnd=datetime.datetime.today()):
#    """ connect to database"""
#    dbName = Last_DB()
#    conn = sqlite3.connect(dbName)
#    db_cursor=conn.cursor()   
#    
#    if type(ticker_list)==str:
#        ticker_temp= ticker_list
#        ticker_list=[]
#        ticker_list.append(ticker_temp)
#    Ticker_ID=Asset_getAssetsId_fromTicker(ticker_list, db_cursor)
#    dfMarketCap = pd.DataFrame()
#    for k in range(0,len(ticker_list)):
#        assetid= Ticker_ID[k]
#        res=db_cursor.execute("SELECT * FROM freeCashFlow Where  Asset_ID = '"+str(assetid) +"' AND date >= '" + str(DateBegin) +"' AND date <= '" + str(DateEnd) +"'" )
#        date = []
#        FCF=[]
#        for resultat in res:
#            date.append(resultat[2])
#            FCF.append(resultat[3])
#            dfMarketCaptemp= pd.DataFrame({'date.'+str(ticker_list[k]):date,'MarketCap.'+str(ticker_list[k]):MarketCap})
#            
#        if len(dfMarketCap)==0:
#            dfMarketCap=dfMarketCaptemp
#        else:
#            if len(dfMarketCaptemp)>0:
#                frames = [dfMarketCap,dfMarketCaptemp]
#                dfMarketCap=pd.concat(frames,axis=1)
#        return dfMarketCap

def MarketCap_getAssetMarketCap(ticker_list,dtBegin,dtEnd,bIntersection,calendar=np.nan):
    field_list='MarketCap'
    dfPrice = DB_BDH_Call(ticker_list,field_list,dtBegin,dtEnd,bIntersection,calendar)
    return dfPrice
# *********************************************
#                 EarningYield
# *********************************************
def EarningYield_TableCreation():
    dbName = Last_DB()
    conn = sqlite3.connect(dbName)
    db_cursor=conn.cursor() 
    """ tests if the table already exists """
    res = db_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='EarningYield'")
    result = []
    for elt in res :
        result.append(elt[0])
    if len(result)==0:
        db_cursor.execute('''CREATE TABLE EarningYield(id INTEGER PRIMARY KEY AUTOINCREMENT, Asset_ID INTEGER, date TEXT,Yield REAL)''')
        message="Table EarningYield created."
    else:
        message="Table EarningYield already exists."
    conn.commit()
    conn.close()
    return message
    
def AssetEarningYield_setAssetEarningYield_from_csv(csv_path):
    dfData=pd.read_csv(csv_path)
    ticker_list=dfData['Tickers'].dropna().tolist()
    message = AssetEarningYield_setAssetEarningYield(dfData,ticker_list)
    return message
    
def AssetEarningYield_setAssetEarningYield(dfData,ticker_list):
    if type(ticker_list)==str:
        ticker_list=ticker_list.split()
    """ connect to database"""
    dbName = Last_DB()
    conn = sqlite3.connect(dbName)
    db_cursor=conn.cursor()   
    """ extract data """
    message =Asset_setTickers(ticker_list)
    Ticker_ID=Asset_getAssetsId_fromTicker(ticker_list, db_cursor)
    ticker_delete=[]
    icount = 0
    values = []
    
    
    for k in range(0,len(ticker_list)):
        ticker = ticker_list[k]
        ticker_id = Ticker_ID[k]
        ticker_delete.append((ticker_id,))
        dfAssetEarningYield = pd.DataFrame({'date.'+str(ticker):dfData['Date.'+str(ticker)].dropna().tolist()})

        if len(dfAssetEarningYield)>0:
            dfAssetEarningYield['date.'+str(ticker)]=index_to_datetime(dfAssetEarningYield['date.'+str(ticker)])
            try:
                dfAssetEarningYield['EarningYield.'+str(ticker)]=dfData['EarningYield.'+str(ticker)]
            except KeyError:
                dfAssetEarningYield['EarningYield.'+str(ticker)]=0.0
                
            v_date=dfAssetEarningYield['date.'+str(ticker)].tolist()
            v_EarningYield = dfAssetEarningYield['EarningYield.'+str(ticker)].fillna(0).tolist()
            
            for i in range(0,len(v_EarningYield)):
                values.append((ticker_id,str(v_date[i]),v_EarningYield[i]))
                
#            for i in range(0,len(dfAssetEarningYield)):
#                date=str(dfAssetEarningYield.loc[dfAssetEarningYield.index[i],'date.'+str(ticker)])
#                try:
#                    EarningYield=float(dfAssetEarningYield.loc[dfAssetEarningYield.index[i],'EarningYield.'+str(ticker)])
#                except ValueError:
#                    EarningYield=0.0
#                values.append((ticker_id,date,EarningYield))
            
            icount+=1
            
#    db_cursor.executemany("DELETE FROM EarningYield WHERE Asset_ID=?",ticker_delete)       
    db_cursor.executemany("INSERT INTO EarningYield(Asset_ID,date,Yield ) VALUES (?,?,?)",values)        
    message += "\n" +" "+ str(icount)+" EarningYield inserted."   
    conn.commit()
    conn.close()
    return message

def AssetEarningYield_getAssetEarningYield(ticker_list,dtBegin,dtEnd,bIntersection,calendar=np.nan):
    field_list='EarningYield'
    dfPrice = DB_BDH_Call(ticker_list,field_list,dtBegin,dtEnd,bIntersection,calendar)
    return dfPrice

      
      
"""                               DATABASE CALLS                            """
       
"""
This function is the equivalent of a BDH data request
=> it is for daily data, between two given dates
=> a calendar (list) can be provided to get data at those dates
=> if bIntersection = False: join   else: intersection
"""  
def DB_BDH_Call(ticker_list,field_list,dtBegin,dtEnd,bIntersection,calendar=np.nan):  
    bdhcall = Data_Request()
    possible_fields =bdhcall.BDH_Fields
    bFldsError = False
    if type(ticker_list) is not list:
        ticker_list_temp = ticker_list
        ticker_list=[]
        ticker_list.append(ticker_list_temp)
    
    if type(field_list) is not list:
        field_list_temp = field_list
        field_list=[]
        field_list.append(field_list_temp)
        
    """
    clean the field list
    """
    
    field_list_temp = []
    for field in field_list:
        if field in ['price','Price','Last_price','Last_Price','Px_last','Px_Last','Last','last','PX_LAST']:
            field_list_temp.append('Last_Price')
        else:
            if field in ['bid','Bid','Px_Bid','Px_bid','px_bid','PX_BID']:
                field_list_temp.append('Bid')
            else:
                if field in ['ask','Ask','Px_Ask','Px_ask','px_ask','PX_ASK']:
                    field_list_temp.append('Ask')
                else:
                    if field in ['volume','Volume','Px_Volume','Px_volume','px_volume','PX_VOLUME']:
                        field_list_temp.append('Volume')
                    else:
                        if field in ['fcf','FCF','Free_Cash_Flow','free_cash_flow','FreeCashFlow','freecashflow']:
                            field_list_temp.append('Free_Cash_Flow')
                        else:
                            if field in ['Market_Cap','market_cap','curr_mkt_cap','MC','mc','MarketCap','marketcap']:
                                field_list_temp.append('Market_Cap')
                            else:
                                if field in ['Earning_Yield','earning_yield','EarningYield','earningyield','ey','EY']:
                                    field_list_temp.append('Earning_Yield')
                                else:
                                    priceHsty = "Fields error: " +"\n" +bdhcall.Help("DB_BDH_Call")
                                    bFldsError =True
                                    
           
    field_list=field_list_temp 
    
    
    if bFldsError == False:
        if type(ticker_list)==str:
            ticker_temp= ticker_list
            ticker_list=[]
            ticker_list.append(ticker_temp)
        ticker_list = list(set(ticker_list))
        """
        connect to the last database
        """
        dbName = Last_DB()
        conn = sqlite3.connect(dbName)
        cursor=conn.cursor()   
        """
        ticker_list analysis
        """
        listId = Asset_getAssetsId_fromTicker(ticker_list, cursor)
        
        ID_Tickers={}
        for z in range(0,len(ticker_list)):
            asset = ticker_list[z]
            assetid=listId[z]
            ID_Tickers[assetid,'ticker']=asset
            
        
        """
        fields analysis
        """
        #calls to the Asset_Price table
        Asset_Price_request="Select Asset_ID,date, bid, last, ask, open, high, low FROM Asset_Price WHERE (date>='"+dtBegin.strftime("%Y-%m-%d %H:%M:%S")+ "' and date<='"+dtEnd.strftime("%Y-%m-%d %H:%M:%S")+ "') and ( "
        Asset_Volume_request="Select Asset_ID,date, Volume FROM TradingVolume WHERE (date>='"+dtBegin.strftime("%Y-%m-%d %H:%M:%S")+ "' and date<='"+dtEnd.strftime("%Y-%m-%d %H:%M:%S")+ "') and ( "
        Asset_Free_Cash_Flow_request  = "Select  Asset_ID, date ,FCF FROM FreeCashFlow WHERE (date>='"+dtBegin.strftime("%Y-%m-%d %H:%M:%S")+ "' and date<='"+dtEnd.strftime("%Y-%m-%d %H:%M:%S")+ "') and ( "
        Asset_Market_Cap_request  = "Select  Asset_ID, date ,MarketCap FROM MarketCap WHERE (date>='"+dtBegin.strftime("%Y-%m-%d %H:%M:%S")+ "' and date<='"+dtEnd.strftime("%Y-%m-%d %H:%M:%S")+ "') and ( " 
        Asset_EarningYield_request="Select  Asset_ID, date ,Yield  FROM EarningYield WHERE (date>='"+dtBegin.strftime("%Y-%m-%d %H:%M:%S")+ "' and date<='"+dtEnd.strftime("%Y-%m-%d %H:%M:%S")+ "') and ( "
        
        Asset_columns = []
        
        if 'Last_Price' in field_list:
            for asset in ticker_list:
                Asset_columns.append('Last_Price.'+asset)
        if 'Bid' in field_list:
            for asset in ticker_list:
                Asset_columns.append('Bid.'+asset)
        if 'Ask' in field_list:
            for asset in ticker_list:
                Asset_columns.append('Ask.'+asset)
        if 'Open' in field_list:
            for asset in ticker_list:
                Asset_columns.append('Open.'+asset)
                
        if 'High' in field_list:
            for asset in ticker_list:
                Asset_columns.append('High.'+asset)
                
        if 'Low' in field_list:
            for asset in ticker_list:
                Asset_columns.append('Low.'+asset)
        
        if 'Volume' in field_list:
            for asset in ticker_list:
                Asset_columns.append('Volume.'+asset)
        
        if 'Free_Cash_Flow' in field_list:
            for asset in ticker_list:
                Asset_columns.append('Free_Cash_Flow.'+asset)
            
        if 'Market_Cap' in field_list:
            for asset in ticker_list:
                Asset_columns.append('Market_Cap.'+asset)
            
        if 'Earning_Yield' in field_list:
            for asset in ticker_list:
                Asset_columns.append('Earning_Yield.'+asset)
            
        icount_request = 0
        sqlCommand = ''
        price = {}
        volume = {}
        fcf = {}
        mkt_cap = {}
        earningYield = {}

        for i in range(0,len(listId)):
            sqlCommand = sqlCommand + ' Asset_ID = ' + str(listId[i]) + ' OR '
            icount_request+=1
            if icount_request>500 or i == len(listId)-1:
                sqlCommand = sqlCommand + " Asset_ID = '0' ) "
                
                """load bid, last, ask """
                if 'Last_Price' in field_list:
                    tmp = cursor.execute(Asset_Price_request+ sqlCommand)
                    for elem in tmp:
                        if (ID_Tickers[elem[0],'ticker'] in price.keys()):
                            price[ID_Tickers[elem[0],'ticker']].append((elem[0],elem[1],elem[2],elem[3],elem[4],elem[5],elem[6],elem[7]))
                        else:
                            price.update({ID_Tickers[elem[0],'ticker'] : [(elem[0],elem[1],elem[2],elem[3],elem[4],elem[5],elem[6],elem[7])]})
                    """load volume"""
                if 'Volume' in   field_list: 
                    tmp = cursor.execute(Asset_Volume_request + sqlCommand)
                    for elem in tmp:
                        if (ID_Tickers[elem[0],'ticker'] in volume.keys()):
                            volume[ID_Tickers[elem[0],'ticker']].append((elem[0],elem[1],elem[2]))
                        else:
                            volume.update({ID_Tickers[elem[0],'ticker'] : [(elem[0],elem[1],elem[2])]})
                        
                """ load free_cash_flow """
                if 'Free_Cash_Flow' in   field_list:  
                    tmp = cursor.execute(Asset_Free_Cash_Flow_request  + sqlCommand)
                    for elem in tmp:
                        if (ID_Tickers[elem[0],'ticker'] in fcf.keys()):
                            fcf[ID_Tickers[elem[0],'ticker']].append((elem[0],elem[1],elem[2]))
                        else:
                            fcf.update({ID_Tickers[elem[0],'ticker'] : [(elem[0],elem[1],elem[2])]}) 
                """ load Market_Cap """  
                if 'Market_Cap' in   field_list:     
                    tmp = cursor.execute(Asset_Market_Cap_request  + sqlCommand)
                    for elem in tmp:
                        if (ID_Tickers[elem[0],'ticker'] in mkt_cap.keys()):
                            mkt_cap [ID_Tickers[elem[0],'ticker']].append((elem[0],elem[1],elem[2]))
                        else:
                            mkt_cap .update({ID_Tickers[elem[0],'ticker'] : [(elem[0],elem[1],elem[2])]})                
                                               
                """ load EarningYield """ 
                if 'Earning_Yield' in   field_list:     
                    tmp = cursor.execute(Asset_EarningYield_request + sqlCommand)
                    for elem in tmp:
                        if (ID_Tickers[elem[0],'ticker'] in earningYield.keys()):
                            earningYield[ID_Tickers[elem[0],'ticker']].append((elem[0],elem[1],elem[2]))
                        else:
                            earningYield.update({ID_Tickers[elem[0],'ticker'] : [(elem[0],elem[1],elem[2])]})                    
           
                icount_request=0
                sqlCommand =""
            
        priceHsty=pd.DataFrame()
        if 'Last_Price' in field_list:
            for asset in ticker_list:
                Date=[]
                Bid = []
                Last=[]
                Ask=[]
                try:
                    for l in range(0,len(price[asset])):
                        Date.append(price[asset][l][1])
                        Bid.append(price[asset][l][2])
                        Last.append(price[asset][l][3])
                        Ask.append(price[asset][l][4])
                except KeyError:
                    pass 
                priceHstytemp=pd.DataFrame({'Bid.'+str(asset):Bid,'Last_Price.'+str(asset):Last,'Ask.'+str(asset):Ask},index =Date)
                if len(priceHstytemp)>0:
                    frames = [priceHsty, priceHstytemp]
                    priceHsty=pd.concat(frames,axis=1)
                
        """load volume"""
        if 'Volume' in   field_list:     
            for asset in ticker_list:
                Volume = []
                Date=[]
                try:
                    for l in range(0,len(volume[asset])):
                        Date.append(volume[asset][l][1])
                        Volume.append(volume[asset][l][2])
                except KeyError:
                    pass 
                priceHstytemp=pd.DataFrame({'Volume.'+str(asset):Volume},index = Date)   
                
                if len(priceHstytemp)>0 :
                    frames = [priceHsty, priceHstytemp]
                    priceHsty=pd.concat(frames,axis=1)
                    
                
        """ load free_cash_flow """
        if 'Free_Cash_Flow' in   field_list:     
            for asset in ticker_list:
                FCF=[]
                Date=[]
                try:
                    for l in range(0,len(fcf[asset])):
                        Date.append(fcf[asset][l][1])
                        FCF.append(fcf[asset][l][2])
                except KeyError:
                    pass 
                
                priceHstytemp=pd.DataFrame({'Free_Cash_Flow.'+str(asset):FCF},index = Date)   
                if len(priceHstytemp)>0 :
                    frames = [priceHsty, priceHstytemp]
                    priceHsty=pd.concat(frames,axis=1)
              
        """ load Market_Cap """  
        if 'Market_Cap' in   field_list:       
            for asset in ticker_list:
                MKT_cap = []
                Date=[]
                try:
                    for l in range(0,len(mkt_cap[asset])):
                        #change from fcf to mkt
                        Date.append(mkt_cap[asset][l][1])
                        MKT_cap.append(mkt_cap[asset][l][2])

                except KeyError:
                    pass 
                
                priceHstytemp=pd.DataFrame({'Market_Cap.'+str(asset):MKT_cap},index = Date)   
                #Shenrui Jin: Change 
                #From: priceHstytemp=pd.DataFrame({'Market_Cap.'+str(asset):MKT_cap},index = Date)   
                #To: priceHstytemp=pd.DataFrame({'Market_Cap.'+str(asset):mkt_cap},index = Date)   
                if len(priceHstytemp)>0 :
                    frames = [priceHsty, priceHstytemp]
                    priceHsty=pd.concat(frames,axis=1)
        
        """ load EarningYield """ 
        if 'Earning_Yield' in   field_list:     
            for asset in ticker_list:
                EarningYield = []
                Date=[]
                try:
                    for l in range(0,len(earningYield[asset])):
                        Date.append(earningYield[asset][l][1])
                        EarningYield.append(earningYield [asset][l][2])
                except KeyError:
                    pass 
                priceHstytemp=pd.DataFrame({'Earning_Yield.'+str(asset):EarningYield},index = Date) 
                if len(priceHstytemp)>0 :
                    frames = [priceHsty, priceHstytemp]
                    priceHsty=pd.concat(frames,axis=1)
            
        priceHsty.index=priceHsty.index.map(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
        columns = priceHsty.columns            
        for column in columns:
            if column not in Asset_columns:
                del priceHsty[column]   
                
        if type(calendar) == list:
            dftemp = pd.DataFrame(index = calendar)
            columns = priceHsty.columns
            for column in columns:
                dftemp[column]=priceHsty[column]
            priceHsty = dftemp
            
        if bIntersection == True:
            priceHsty=priceHsty.dropna()
            
        priceHsty=priceHsty[priceHsty.index>=dtBegin]
        priceHsty=priceHsty[priceHsty.index<=dtEnd]
        conn.close()
    else:
        priceHsty = "Fields error: " +"\n" +bdhcall.Help("DB_BDH_Call")
        
    return priceHsty
    

def DB_BDH_Call_to_csv(ticker_list,field_list,dtBegin,dtEnd,bIntersection,calendar=np.nan):
      priceHsty=DB_BDH_Call(ticker_list,field_list,dtBegin,dtEnd,bIntersection,calendar)
      now = datetime.datetime.today()
      day = now.day
      month = now.month
      year =now.year
      hour = now.hour
      minute = now.minute
      title = "BDH Request "+str(month) + str(day) + str(year) +"_" + str(hour)+"h"+str(minute)+"min.csv"
      if type(priceHsty) == str :
          print(priceHsty)
      else:
          priceHsty.to_csv('K:\\ED_ExcelTools\\Transfert\\Structuring\\Proprietary Indices\\NYC_Engineering_Python\\DB_requests\\'+title)
          return 'K:\\ED_ExcelTools\\Transfert\\Structuring\\Proprietary Indices\\NYC_Engineering_Python\\DB_requests\\'+title

 
def DB_BDS_Call(ticker_list,field_list):    
    """
    connect to the last database
    """
    if type(ticker_list) is not list:
        ticker_list_temp = ticker_list
        ticker_list=[]
        ticker_list.append(ticker_list_temp)
    
    if type(field_list) is not list:
        field_list_temp = field_list
        field_list=[]
        field_list.append(field_list_temp)
    
    dbName = Last_DB()
    conn = sqlite3.connect(dbName)
    cursor=conn.cursor()   
    """
    ticker_list analysis
    """
    listId = Asset_getAssetsId_fromTicker(ticker_list, cursor)
  
    ID_Tickers={}
    for k in range(0,len( ticker_list)):
        asset=ticker_list[k]
        assetid=listId[k]
        ID_Tickers[assetid,'ticker']=asset    
    """
    fields analysis
    """
    #calls to the Asset_Price table
    Asset_Dividends_request="Select Asset_ID , DeclarationDate ,ExecutionDate,SettlementDate ,PaymentDate, DividendGrossValue , DividendFrequency, DividendType from Dividends WHERE ("
    Asset_BuyBacks_request="Select Asset_ID,date,Amount,Currency, Type FROM BuyBacks Where  "
    #WHERE (date>='"+dtBegin.strftime("%Y-%m-%d %H:%M:%S")+ "' and date<='"+dtEnd.strftime("%Y-%m-%d %H:%M:%S")+ "') and

    icount_request = 0
    
    sqlCommand = ''
    dividend = {}
    buyBacks = {}
    
    Asset_columns = []
    for i in range(0,len(listId)):
        sqlCommand = sqlCommand + 'Asset_ID = ' + str(listId[i]) + ' OR '
        if icount_request >500 or i == len(listId)-1:
            sqlCommand = sqlCommand + "Asset_ID = '0') and DividendType='Regular Cash' "
            #and ExecutionDate>='"+dtBegin.strftime("%Y-%m-%d %H:%M:%S")+ "' and ExecutionDate<='"+dtEnd.strftime("%Y-%m-%d %H:%M:%S")+ "'
            if 'Dividends' in field_list:
                tmp = cursor.execute(Asset_Dividends_request+ sqlCommand)
                for elem in tmp:
                    #print (elem[0],elem[1],elem[2],elem[3],elem[4],elem[5],elem[6],elem[7])
                    if (ID_Tickers[elem[0],'ticker'] in dividend .keys()):
                        dividend[ID_Tickers[elem[0],'ticker']].append((elem[0],elem[1],elem[2],elem[3],elem[4],elem[5],elem[6],elem[7]))                        
                    else:
                        dividend.update({ID_Tickers[elem[0],'ticker'] : [(elem[0],elem[1],elem[2],elem[3],elem[4],elem[5],elem[6],elem[7])]})
          
            if 'BuyBacks' in field_list:
                tmp = cursor.execute(Asset_BuyBacks_request + sqlCommand)
                

                for elem in tmp:
                    if (ID_Tickers[elem[0],'ticker'] in buyBacks.keys()):
                        buyBacks[ID_Tickers[elem[0],'ticker']].append((elem[0],elem[1],elem[2],elem[3],elem[4]))
                    else:
                        buyBacks.update({ID_Tickers[elem[0],'ticker'] : [(elem[0],elem[1],elem[2],elem[3],elem[4])]})   
            icount_request=0
            sqlCommand =""
        icount_request+=1
    
 
    dfResult = pd.DataFrame()
    
    for k in range(0,len(ticker_list)):
        asset=ticker_list[k]
        """dividends """
        if 'Dividends' in field_list or 'Div' in fields_list or 'Dvd' in fields_list or 'dvd' in fields_list or 'dividends' in fields_list:
            Asset_columns.append('DeclarationDate.'+asset)
            Asset_columns.append('ExecutionDate.'+asset)
            Asset_columns.append('SettlementDate.'+asset)
            Asset_columns.append('PaymentDate.'+asset)
            Asset_columns.append('DividendGrossValue.'+asset)
            Asset_columns.append('DividendFrequency.'+asset)
            Asset_columns.append('DividendType.'+asset)
            
            DeclarationDate=[]
            ExecutionDate=[]
            SettlementDate=[]
            PaymentDate=[]
            DividendGrossValue=[]
            DividendFrequency=[]
            DividendType=[]
            if asset in dividend.keys():
                for l in range(0,len(dividend[asset])):
                    DeclarationDate.append(dividend[asset][l][1])
                    ExecutionDate.append(dividend[asset][l][2])
                    SettlementDate.append(dividend[asset][l][3])
                    PaymentDate.append(dividend[asset][l][4])
                    DividendGrossValue.append(dividend[asset][l][5])
                    DividendFrequency.append(dividend[asset][l][6])
                    DividendType.append(dividend[asset][l][7])

                dfHstytemp=pd.DataFrame({'DeclarationDate.'+str(asset):DeclarationDate,'ExecutionDate.'+str(asset):ExecutionDate,'SettlementDate.'+str(asset):SettlementDate,'PaymentDate.'+str(asset):PaymentDate,'DividendGrossValue.'+str(asset):DividendGrossValue,'DividendFrequency.'+str(asset):DividendFrequency,'DividendType.'+str(asset):DividendType})
                if len(dfHstytemp)>0:
                    frames = [dfResult,dfHstytemp]
                    dfResult=pd.concat(frames,axis=1)
             
        """load buybacks"""
        if 'BuyBacks' in field_list or 'bb'in field_list  or 'BB'in field_list or 'buybacks' in field_list or 'buyback' in field_list or 'Buyback' in field_list:
            Asset_columns.append('BuyBackDate.'+str(asset))
            Asset_columns.append('Amount.'+asset)
            Asset_columns.append('Currency.'+asset)
            Asset_columns.append('Type.'+asset)
            Amount = []
            Currency=[]
            Type=[]
            Date=[]
            try:
                for l in range(0,len(buyBacks[asset])):
                    Date.append(buyBacks[asset][l][1])
                    Amount.append(buyBacks[asset][l][2])
                    Currency.append(buyBacks[asset][l][3])
                    Type.append(buyBacks[asset][l][4])
            except KeyError:
                pass 
            
            dfHstytemp=pd.DataFrame({'BuyBackDate.'+str(asset):Date,'Amount.'+str(asset):Amount,'Currency.'+str(asset):Currency,'Type.'+str(asset):Type})   
            if len(dfHstytemp)>0:
                frames = [dfResult,dfHstytemp]
                dfResult=pd.concat(frames,axis=1)
         
    columns = dfResult.columns            
    for column in columns:
        if column not in Asset_columns:
            del dfResult[column]   

    conn.close()
    return dfResult
    

def DB_BDS_Call_to_csv(ticker_list,field_list):
      priceHsty=DB_BDS_Call(ticker_list,field_list)
      now = datetime.datetime.today()
      day = now.day
      month = now.month
      year =now.year
      hour = now.hour
      minute = now.minute
      title = "BDS Request "+str(month) + str(day) + str(year) +"_" + str(hour)+"h"+str(minute)+"min.csv"
      priceHsty.to_csv('K:\\ED_ExcelTools\\Transfert\\Structuring\\Proprietary Indices\\NYC_Engineering_Python\\DB_requests\\'+title)
      return 'K:\\ED_ExcelTools\\Transfert\\Structuring\\Proprietary Indices\\NYC_Engineering_Python\\DB_requests\\'+title
