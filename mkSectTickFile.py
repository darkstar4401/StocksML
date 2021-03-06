import datetime
import pandas as pd
import pandas_datareader.data as web
import os
import requests
from bs4 import BeautifulSoup
urls = ["https://finviz.com/screener.ashx?v=111&f=geo_usa,sec_basicmaterials","https://finviz.com/screener.ashx?v=111&f=geo_usa,sec_consumergoods",
"https://finviz.com/screener.ashx?v=111&f=geo_usa,sec_financial","https://finviz.com/screener.ashx?v=111&f=geo_usa,sec_healthcare",
"https://finviz.com/screener.ashx?v=111&f=geo_usa,sec_industrialgoods","https://finviz.com/screener.ashx?v=111&f=geo_usa,sec_services",
"https://finviz.com/screener.ashx?v=111&f=geo_usa,sec_technology","https://finviz.com/screener.ashx?v=111&f=geo_usa,sec_utilities"]

sectors = ["basic_materials", "consumer_goods", "financial", "healthcare", "industrial_goods", "services", "technology", "utilities"]

def getallstocks():
    print (datetime.datetime.now())
    print ("Finviz Performance Start")
    allticks = []
    pages = [20,15,163,32,15,33,30,5]
    pages2 = ["381","281","3261","621","281","641","581","81"]
    #pages = [5,5,5,5,5,5,5,5]

    for i, url in enumerate(urls):
        sectorurl = urls[i]
        currentpage  = pages2[i]
        tickerlist = []
        pgcount = 1
        while pgcount<pages[i]:
            print(pgcount , " page(s) done")
            sectorurl = sectorurl+"&r="+str(currentpage)
            secondresponse = requests.get(sectorurl)
            secondhtml = secondresponse.content
            secondsoup = BeautifulSoup(secondhtml, "lxml")
            stockdata = secondsoup.find_all('a', {"class" : "screener-link"})
            stockticker = secondsoup.find_all('a', {"class" : "screener-link-primary"})
            datalength = len(stockdata)
            tickerdatalength = len(stockticker)

            while datalength > 0:
                tickerlist.append(stockticker[tickerdatalength-1].text)
                datalength -= 15
                tickerdatalength -= 1
            currentpage = int(currentpage)
            currentpage-= 20
            
            pgcount +=1

        path = r"/home/me/Soran/stock_hist_data/"+sectors[i]
        sector = sectors[i]
        mkFile(path, tickerlist, sector)
        #file_path = mksectF(path, tickerlist)
        allticks.append(tickerlist)

    return allticks
def mkFile(path, tickerlist,sector):
    fileName = path+"/"+sector+".txt"
    with open(fileName, 'w') as file:
        for tick in tickerlist:
            file.write(tick)
            file.write("\n")
    file.close()

def mksectF(path, tickerlist):
    errors =[]
    n = 365*5
    start = datetime.datetime.now() - datetime.timedelta(days=n)
    end = datetime.datetime.now()

    for ticker in tickerlist:
        try:
            file_name = path + r"/" + ticker
            tick = ticker.upper()
            df = web.DataReader(tick, 'yahoo', start, end)
            df.to_csv(file_name)
        except Exception as e:
            errors.append(ticker)
    return path








alltickerlist = getallstocks()

