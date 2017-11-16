import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import pandas_datareader.data as web
import os
import requests
def create_dataset(dataset):
        dataX = [dataset[n+1] for n in range(len(dataset)-2)]
        return np.array(dataX), dataset[2:]
def getTicks(sector):
    path = r"C:/Users/ME/Desktop/soran/soran/stock_hist_data/"+sector
    ticks = os.listdir(path)
    return ticks
def getstock(path, stock):
    opens= []
    dates=[]
    filename = path+r"/"+stock
    with open(filename) as f:
        for n, line in enumerate(f):
            if n != 0:
                opens.append(float(line.split(',')[0]))
                dates.append(float(line.split(',')[1]))
    dataset = pd.Series(opens,index=dates)
    return dataset
def getsector(path):
    opens= []
    dates=[]
    for stock in os.listdir(path):
        filename = path+r"/"+stock
        with open(filename) as f:
            for n, line in enumerate(f):
                if n != 0:
                    opens.append(float(line.split(',')[0]))
                    dates.append(float(line.split(',')[1]))
                #if line == line no


    dataset = pd.Series(opens,index=dates)
    return dataset
def gettickerhist(tick):
    filename = "getHist.csv"
    url = 'http://www.google.com/finance/historical?q=NASDAQ%3A'+tick+'&output=csv'
    r = requests.get(url, stream=True)

    if r.status_code != 400:
        with open(FILE_NAME, 'wb') as f:
            for chunk in r:
                f.write(chunk)
    opens= []
    dates=[]
    with open(FILE_NAME) as f:
        for n, line in enumerate(f):
            if n != 0:
                opens.append(float(line.split(',')[0]))
                opens.append(float(line.split(',')[1]))

    dataset = pd.Series(opens,index=dates)
    return dataset, len(dataset)
def fit_lstm(train, batch_size=1, nb_epoch=20, neurons=4):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    #each epoch stock in sector
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    return model
def getallsectdataset():
    sectors = ["basic_materials", "consumer_goods", "financial", "healthcare", "industrial_goods", "services", "technology", "utilities"]
    for i,sector in enumerate(sectors):
        path = r"C:/Users/ME/Desktop/soran/soran/stock_hist_data/"+sector
        if i==0:
            basic_materials=getsector(path)
            basic_materials_tick = getTicks(sector)
        if i==1:
            consumer_goods=getsector(path)
            consumer_goods = getTicks(sector)
        if i==2:
            financial=getsector(path)
            financial = getTicks(sector)
        if i==3:
            healthcare=getsector(path)
            healthcare = getTicks(sector)
        if i==4:
            industrial_goods=getsector(path)
            industrial_goods = getTicks(sector)
        if i==5:
            services=getsector(path)
            services = getTicks(sector)
        if i==6:
            technology=getsector(path)
            technology = getTicks(sector)
        if i==7:
            utilities=getsector(path)
            utilities = getTicks(sector)
    return basic_materials,consumer_goods,financial,healthcare,industrial_goods,services,technology,utilities
# frame a sequence as a supervised learning problem
def timeseriesdf(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df
# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)
# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]
# scale train and test data to [-1, 1]
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled
# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = numpy.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0,0]
"""
basic_materials,consumer_goods,financial,healthcare,industrial_goods,services,technology,utilities = getallsectdataset()
bmX, bmY = create_dataset(basic_materials)
cgX, cgY = create_dataset(consumer_goods)
fX, fY = create_dataset(financial)
hX, hY = create_dataset(healthcare)
igX, igY = create_dataset(industrial_goods)
servX, servY = create_dataset(services)
techX, techY = create_dataset(technology)
utilX, utilY = create_dataset(utilities)
"""
path = r"C:/Users/ME/Desktop/soran/soran/stock_hist_data/"+sector[0]
#loop through finviz tickers here
tick = "ALDW"


tickerdata =gettickerhist(tick)
length = len(tickerdata)
sectordata = getsector(path,length)
trainX = np.append(sectordata,tickerdata)


