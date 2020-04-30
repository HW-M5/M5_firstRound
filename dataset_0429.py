import pandas as pd 
import os,sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import datetime as dt
from datetime import datetime
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,Dropout
from keras.layers import RepeatVector,TimeDistributed
from keras.models import Sequential, load_model

#===============define data path =============================#
rootPath = os.getcwd()
train_sales= pd.read_csv(os.path.join(rootPath,"sales_train_validation.csv"))
sell_prices= pd.read_csv(os.path.join(rootPath,"sell_prices.csv"))
calendar= pd.read_csv(os.path.join(rootPath,"calendar.csv"))
currentDate = datetime.today().strftime('%Y-%m-%d')
n_features = 9
n_out_seq_length =28

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

def transform(data):
    nan_features= ['event_name_1','event_type_1','event_name_2','event_type_2']
    for feature in nan_features:
        data[feature].fillna('unknown',inplace=True)
    cat = ['event_name_1','event_type_1','event_name_2','event_type_2','snap_CA','snap_TX','snap_WI']
    for feature in cat:
        encoder=preprocessing.LabelEncoder()
        data[feature]=encoder.fit_transform(data[feature])
    return data
def min_max(df):
    return (df-df.mean())/df.std()

def Normalize(list):
    list = np.array(list)
    low, high = np.percentile(list, [0, 100])
    delta = high - low
    if delta != 0:
        for i in range(0, len(list)):
            list[i] = (list[i]-low)/delta
    return  list,low,high

def FNoramlize(list,low,high):
    delta = high - low
    if delta != 0:
        for i in range(0, len(list)):
            list[i] = list[i]*delta + low
    return list

def Normalize2(list,low,high):
    list = np.array(list)
    delta = high - low
    if delta != 0:
        for i in range(0, len(list)):
            list[i] = (list[i]-low)/delta
    return  list

def dataSet():
    global train_sales,sell_prices,calendar,n_out_seq_length,n_features

    num=30490 #traindata(items)
    days = range(1, 1970)
    time_series_columns = [f'd_{i}' for i in days]
    transfer_cal = pd.DataFrame(calendar[['event_name_1','event_type_1','event_name_2','event_type_2','snap_CA','snap_TX','snap_WI']].values.T, index=['event_name_1','event_type_1','event_name_2','event_type_2','snap_CA','snap_TX','snap_WI'], columns= time_series_columns)
    transfer_cal = transfer_cal.fillna(0)
    event_name_1_se = transfer_cal.loc['event_name_1'].apply(lambda x: x if re.search("^\d+$", str(x)) else np.nan).fillna(10)
    event_name_2_se = transfer_cal.loc['event_name_2'].apply(lambda x: x if re.search("^\d+$", str(x)) else np.nan).fillna(10)
    calendar['date'] = pd.to_datetime(calendar['date'])
    calendar = calendar[calendar['date']>= '2016-1-27']  #reduce memory
    calendar= transform(calendar)
    # Attempts to convert events into time series data.
    transfer_cal = pd.DataFrame(calendar[['event_name_1','event_type_1','event_name_2','event_type_2','snap_CA','snap_TX','snap_WI']].values.T,
                                index=['event_name_1','event_type_1','event_name_2','event_type_2','snap_CA','snap_TX','snap_WI'])
    print(transfer_cal)
    price_fea = calendar[['wm_yr_wk','date']].merge(sell_prices, on = ['wm_yr_wk'], how = 'left')
    price_fea['id'] = price_fea['item_id']+'_'+price_fea['store_id']+'_validation'
    df = price_fea.pivot('id','date','sell_price')
    price_df = train_sales.merge(df,on=['id'],how= 'left').iloc[:,-145:]
    price_df.index = train_sales.id
    price_df.head()  
    days = range(1, 1913 + 1)
    time_series_columns = [f'd_{i}' for i in days]
    time_series_data = train_sales[time_series_columns]  #Get time series data
    calendar= pd.read_csv(os.path.join(rootPath,"calendar.csv"))
    X = []   #build a data with two features(salse and event1)
    for i in tqdm(range(time_series_data.shape[0])):
        X.append([list(t) for t in zip(transfer_cal.loc['event_name_1'][-(100+28):-(28)],
                                    transfer_cal.loc['event_type_1'][-(100+28):-(28)],
                                    transfer_cal.loc['event_name_2'][-(100+28):-(28)],     #emmmm.....Those features didn't work for me...
                                    transfer_cal.loc['event_type_2'][-(100+28):-(28)],
                                    transfer_cal.loc['snap_CA'][-(100+28):-(28)],
                                    transfer_cal.loc['snap_TX'][-(100+28):-(28)],
                                    transfer_cal.loc['snap_WI'][-(100+28):-(28)],
                                    price_df.iloc[i][-(100+28):-(28)],
                                    time_series_data.iloc[i][-100:])]) 

    X = np.asarray(X, dtype=np.float32)
    np.random.seed(7)
    n_steps = 28
    train_n,train_low,train_high = Normalize(X[:,-(n_steps*2):,:])
    X_train = train_n[:,-28*2:-28,:]
    y = train_n[:,-28:,8]  #     
    # reshape from [samples, timesteps] into [samples, timesteps, features]

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
    y_train = y.reshape((y.shape[0], y.shape[1], 1))
    print(X_train.shape)
    return X_train,y_train,n_features

def createModel():
    global currentDate,n_out_seq_length,n_features
    # define model
    model = Sequential()
    model.add(LSTM(128, activation='selu', input_shape=(28, n_features),return_sequences=False))
    model.add(RepeatVector(n_out_seq_length))
    model.add(LSTM(32, activation='selu',return_sequences=True))
    #model.add(Dropout(0.1))  
    model.add(TimeDistributed(Dense(1)))   # num_y means the shape of y,in some problem(like translate), it can be many.
                                                #In that case, you should set the  activation= 'softmax'
    model.compile(optimizer='adam', loss='mse')
    # demonstrate prediction
    return model 
   

if __name__ == '__main__':
    #data generate
    X_train, y_train, n_features = dataSet() 
    model =createModel()
    ckptCallback = ModelCheckpoint(f"{currentDate}.h5",monitor='loss',verbose=0,save_best_only=True,save_weights_only=True,mode='auto',period=30)
    history=model.fit(X_train, y_train, epochs=1000, batch_size=512,callbacks=[ckptCallback])
    model.save('lastest_0429.h5')
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc = 'upper right')
    plt.savefig(f"{currentDate}.png")

    # prediction process
    x_input = np.array(X_train[:,-n_steps*1:])
    x_input = x_input.reshape((num, n_steps*1, n_features))
    print(x_input.shape)
    #x_input = Normalize2(x_input,train_low,train_high)
    yhat = model.predict(x_input[:,-n_steps:], verbose=0)
    x_input=np.concatenate((x_input[:,:,8].reshape(x_input.shape[0],x_input.shape[1]),yhat.astype(np.float32).reshape(x_input.shape[0],x_input.shape[1])),axis=1).reshape((x_input.shape[0],x_input.shape[1]+28,1))
    #print(yhat)
    print(x_input.shape)
    x_input = FNoramlize(x_input,train_low,train_high)
    x_input = np.rint(x_input)
    forecast = pd.DataFrame(x_input.reshape(x_input.shape[0],x_input.shape[1])).iloc[:,-28:]
    forecast.columns = [f'F{i}' for i in range(1, forecast.shape[1] + 1)]
    forecast[forecast < 0] =0
    forecast.head()
    validation_ids = train_sales['id'].values
    evaluation_ids = [i.replace('validation', 'evaluation') for i in validation_ids]
    ids = np.concatenate([validation_ids, evaluation_ids])
    predictions = pd.DataFrame(ids, columns=['id'])
    forecast = pd.concat([forecast]*2).reset_index(drop=True)
    predictions = pd.concat([predictions, forecast], axis=1)
    predictions.to_csv('submission_0429.csv', index=False)  #Generate the csv file.