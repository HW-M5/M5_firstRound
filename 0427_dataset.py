import pandas as pd 
import os,sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import datetime as dt


#===============define data path =============================#
rootPath = os.getcwd()

sales= pd.read_csv(os.path.join(rootPath,"sales_train_validation.csv"))
sell_prices= pd.read_csv(os.path.join(rootPath,"sell_prices.csv"))
calendar= pd.read_csv(os.path.join(rootPath,"calendar.csv"))

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics: 
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max< np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def dataSet():
    global sales,sell_prices,calendar
    train = reduce_mem_usage(sales)
    calendar = reduce_mem_usage(calendar)
    date_index = calendar['date']
    dates = date_index[0:1913]
    dates_list = [dt.datetime.strptime(date, '%Y-%m-%d').date() for date in dates]

    train['item_store_id'] = train.apply(lambda x: x['item_id']+'_'+x['store_id'],axis=1)
    DF_Sales = train.loc[:,'d_1':'d_1913'].T
    DF_Sales.columns = train['item_store_id'].values
    DF_Sales = pd.DataFrame(DF_Sales).set_index([dates_list])
    DF_Sales.index = pd.to_datetime(DF_Sales.index)

    halfData=int(len(DF_Sales.columns)/2)
    firstHalfDF=DF_Sales[DF_Sales.columns[0:halfData]]
    data = np.array(firstHalfDF)
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(data.reshape(-1, 1))

    train_size = int(len(dataset) * 0.7)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    X_train,y_train = create_dataset(train,28)
    X_test,y_test = create_dataset(test,28)
    return X_train,y_train,X_test,y_test 


def createModel(trainX, y_train):      
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import LSTM
    from keras.callbacks import ModelCheckpoint, EarlyStopping

    model = Sequential()
    model.add(LSTM(512, input_shape=(28,1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=["accuracy"])
    history=model.fit(trainX, y_train, epochs=30, batch_size=1, verbose=2,callbacks=[ckptCallback])

    traning_pred = model.predict(trainX)
    train_pred = pd.Series(scaler.inverse_transform(traning_pred).flatten())
    print(train_pred)

    plt.figure(num=None, figsize=(19, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(train_pred)
    plt.plot(train)
    plt.legend(["Predicted","Real"])
    plt.savefig('1.png')


    test_pred = scaler.inverse_transform(model.predict(testX)).flatten()
    plt.figure(num=None, figsize=(19, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(test_pred)
    plt.plot(test)
    plt.legend(["Predicted","Real"])
    plt.savefig('2.png')

if __name__ == '__main__':
    X_train,y_train,X_test,y_test= dataSet()
    #below start to train process...
    createModel(trainX, y_train)