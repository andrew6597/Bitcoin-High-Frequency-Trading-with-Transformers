import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
from data_prep_tf import train_data_pipe, test_data_pipe, z_score
from model_builider_tf import TransLOB
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

def count_prices(df,t,k):
    current_ask = df.loc[t,'p1_a']
    current_bid = df.loc[t,'p1_b']
    
    future_asks = df.loc[t+1:t+k,'p1_a'] 
    future_bids = df.loc[t+1:t+k,'p1_b'] 

    higher_prices = (future_bids > current_ask).sum()
    lower_prices = (future_asks < current_bid).sum()

    higher_prices_per = higher_prices/k
    lower_prices_per = lower_prices/k
    neutral_prices_per = 1-higher_prices_per-lower_prices_per
    
    return [lower_prices_per, neutral_prices_per, higher_prices_per]

def lr_schedule(epoch, lr):
    if epoch % 2 == 0 and epoch != 0:  # Check if epoch is multiple of 2 and not the initial epoch
        lr = lr * 0.5
    return lr

if __name__ == '__main__':
    print('Started running...')

    # Check if GPU is available
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")

    # Set GPU memory growth
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print('GPU memory growth set to True')
    else:
        print("No GPU found")

    # Load and split df
    df = pd.read_csv('/content/drive/My Drive/LOBseries_100ms.csv').drop(columns=['Unnamed: 0', 'E'])
    #df = pd.read_csv('LOBseries_100ms.csv').drop(columns=['Unnamed: 0','E']).iloc[:10000]
    val_point = len(df)*2//3
    df = df[:val_point]
    print('Loaded df', df.shape)
    # Set params
    # Prompt the user for input and store it in a variable
    n_dim = int(input("Please enter n_dim: "))
    k = int(input("Please enter k: "))
    window_size = int(input("Please enter window_size: "))
    a = float(input("Please enter a: "))
    epochs = int(input("Please enter epochs: "))
    lr = float(input("Please enter lr: "))
    batch_size = int(input("Please enter batch_size: "))
    model_name = str(input('Please enter the name of the model: '))
    
    print('Setting up Targets...')
    df['targets']  = [count_prices(df, t, k) if t < len(df) - k else [0.16, 0.68, 0.16] for t in range(len(df))]
    
    # Print the target mean of each class 
    targets_array  = np.array(list(df['targets'])).reshape(-1,3)
    print('done labeling')    
    print('Down:',targets_array[:,0].mean())
    print('Neutral:',targets_array[:,1].mean())
    print('Up:',targets_array[:, 2].mean())


    val_point = int(len(df) * 2 / 3)
    test_point = val_point + 350000
    X_train = df[:val_point].drop('targets',axis=1)
    y_train = df.loc[:val_point, 'targets']
    
    X_val = df[val_point:test_point].drop('targets',axis=1)
    y_val =  df.loc[val_point:test_point, 'targets']

    
    X_train.reset_index(inplace = True, drop =True)
    y_train.reset_index(inplace = True, drop =True)
    X_val.reset_index(inplace = True, drop =True)
    y_val.reset_index(inplace = True, drop =True)
    
    # Scale the data (Not neccessary if we do it with pct change)
    print('Scaling...')
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train))
    X_val_scaled = pd.DataFrame(scaler.transform(X_val))
    print('Done Scaling')
    

    print('train shape',X_train_scaled.shape)
    print('val shape',X_val_scaled.shape)
    print('Splitting data into timeseries...')
          
    # Train Data
    X_train_series = []
    y_train_series = []
    for t in range(0,len(X_train) - window_size, 1): 
        X_train_series.append(X_train_scaled.iloc[t:t+window_size, :n_dim])
        y_train_series.append(np.array(y_train[t+window_size]))

    np.array(y_train_series)
    N = len(X_train) #Number of total series sized T

    X_train_series = np.array(X_train_series).reshape(-1,window_size,n_dim)
    y_train_series = np.array(y_train_series).reshape(-1,3)
    print('X_train shape:', X_train_series.shape, 'y_train shape:', y_train.shape)
    target_means_train = np.mean(y_train_series, axis=0)
    print('Training Target distribution:', target_means_train)
    
    
    # Validation data
    X_val_series = []
    y_val_series = []
    for t in range(0,len(X_val) - window_size, 1): 
        X_val_series.append(X_val_scaled.iloc[t:t+window_size, :n_dim])
        y_val_series.append(np.array(y_val[t+window_size]))

    np.array(y_val_series)
    N = len(X_val) #Number of total series sized T

    X_val_series = np.array(X_val_series).reshape(-1,window_size,n_dim)
    y_val_series = np.array(y_val_series).reshape(-1,3)
    print('X_val shape:', X_val_series.shape, 'y_val shape:', y_val.shape)
    target_means_val = np.mean(y_val_series, axis=0)
    print('Validation Target distribution:', target_means_val)

    
    # Create and compile the model
    model = TransLOB(window_size, n_dim) 
    model.save(f'/content/drive/My Drive/{model_name}_pre.h5')
    model.compile(
        tf.keras.optimizers.Adam(
            learning_rate=lr,
            beta_1=0.9,
            beta_2=0.999,
            name="Adam",
        ),
        loss=tf.keras.losses.KLDivergence(),
        metrics=['accuracy'],
    )
  
    # Fit the model
    r = model.fit(X_train_series, y_train_series, epochs=epochs, batch_size=batch_size, validation_data=(X_val_series, y_val_series))

    # Save the model
    model.save(f'/content/drive/My Drive/{model_name}.h5')
