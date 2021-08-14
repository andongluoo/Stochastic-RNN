import tensorflow
import yfinance as yf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,GRU
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.metrics import MeanSquaredError
import matplotlib.pyplot as plt

Stock = yf.download('AAPL', 
                      start='2013-01-01', 
                      end='2020-12-31', 
                      progress=False)
all_data = Stock[['Adj Close','Open', 'High', 'Low', 'Close', 'Volume']].round(2)


def ts_train_test_normalize(all_data,time_steps,for_periods):

    ts_train = all_data[:'2019'].iloc[:,0:1].values
    ts_test  = all_data['2020':].iloc[:,0:1].values
    ts_train_len = len(ts_train)
    ts_test_len = len(ts_test)
    # scale the data
    sc = MinMaxScaler(feature_range=(0,1))
    ts_train_scaled = sc.fit_transform(ts_train)

    # create training data of s samples and t time steps
    X_train = []
    y_train = []
    for i in range(time_steps,ts_train_len-1): 
        X_train.append(ts_train_scaled[i-time_steps:i,0])
        y_train.append(ts_train_scaled[i:i+for_periods,0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshaping X_train
    X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))

    #X_test   
    inputs = pd.concat((all_data["Adj Close"][:'2019'], all_data["Adj Close"]['2020':]),axis=0).values
    inputs = inputs[len(inputs)-len(ts_test) - time_steps:]
    inputs = inputs.reshape(-1,1)
    inputs  = sc.transform(inputs)

    X_test = []
    for i in range(time_steps,ts_test_len+time_steps-for_periods):
        X_test.append(inputs[i-time_steps:i,0])
        
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
    return X_train, y_train , X_test, sc

X_train, y_train, X_test, sc = ts_train_test_normalize(all_data,2,2)

def GRU_model(X_train, y_train, X_test, sc):

    my_GRU_model = Sequential()
    my_GRU_model.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
    my_GRU_model.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
    my_GRU_model.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
    my_GRU_model.add(GRU(units=50, activation='tanh'))
    my_GRU_model.add(Dense(units=2))
    my_GRU_model.compile(optimizer=SGD(learning_rate=0.01, decay=1e-7, momentum=0.9, nesterov=False),loss='mean_squared_error')
    my_GRU_model.fit(X_train,y_train,epochs=50,batch_size=150, verbose=0)

    GRU_prediction = my_GRU_model.predict(X_test)
    GRU_prediction = sc.inverse_transform(GRU_prediction)

    return my_GRU_model, GRU_prediction







def SVAE_model(X_train, y_train, X_test, sc):

    nfeatures = 1
    timesteps = 2
    intermediate_dim = 64
    latent_dim = 50


    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0],K.int_shape(z_mean)[1]),
                                mean=0., stddev=0.1)
        return z_mean + K.exp(z_log_sigma) * epsilon

    #ENCODER
    inputs = keras.Input(shape=(timesteps,nfeatures,))
    h = GRU(intermediate_dim, return_sequences=True, activation='relu')(inputs)
    h1 = GRU(intermediate_dim, return_sequences=True,  activation='relu')(h)
    h2 = GRU(intermediate_dim, return_sequences=True,  activation='relu')(h1)
    h3 = GRU(intermediate_dim, return_sequences=False,  activation='relu')(h2)
    z_mean = layers.Dense(latent_dim)(h3)
    z_log_sigma = layers.Dense(latent_dim)(h3)
    z = layers.Lambda(sampling,output_shape=(latent_dim,))([z_mean, z_log_sigma])
    encoder = keras.Model(inputs, z, name='encoder')



    #DECODER
    latent_inputs = keras.Input(shape=(latent_dim), name='z_sampling')
    decoder_repeated = keras.layers.RepeatVector(timesteps)(latent_inputs)
    x = GRU(intermediate_dim, return_sequences=True, activation='relu')(decoder_repeated)
    x1 = GRU(intermediate_dim, return_sequences=True,  activation='relu')(x)
    x2 = GRU(intermediate_dim, return_sequences=True,  activation='relu')(x1)
    x3 = GRU(nfeatures, return_sequences=True,  activation='relu')(x2)
    outputs1 = layers.TimeDistributed(Dense(nfeatures*2))(x3)
    outputs = layers.TimeDistributed(Dense(nfeatures))(outputs1)
    decoder = keras.Model(latent_inputs, outputs, name='decoder')


    outputs = decoder(encoder(inputs))
    vae = keras.Model(inputs, outputs, name='vae_mlp')

    #Loss Function
    rec_loss = (tensorflow.keras.backend.mean(
        tensorflow.keras.losses.mse(inputs, outputs))*nfeatures)
    kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
    kl_loss = K.mean(kl_loss)
    kl_loss *= -0.5


    vae_loss = rec_loss+kl_loss
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam',loss="mse")
    vae.fit(X_train,y_train,epochs=100,batch_size=150, verbose=0)
    VAE_prediction = vae.predict(X_test)[:,:,0]

    VAE_prediction = sc.inverse_transform(VAE_prediction)
    return vae, VAE_prediction



def actual_pred_plot(preds):
    actual_pred = pd.DataFrame(columns = ['Adj. Close', 'prediction'])
    actual_pred['Adj. Close'] = all_data.loc['2020':,'Adj Close'][0:len(preds)]
    actual_pred['prediction'] = preds[:,0]


    m = MeanSquaredError()
    m.update_state(np.array(actual_pred['Adj. Close']),np.array(actual_pred['prediction']))
    return (m.result().numpy(), actual_pred.plot())


def future_predict(model,predictions,future_pred_count):
    sc = MinMaxScaler(feature_range=(0,1))
    predictions = sc.fit_transform(predictions)
    predictions = np.reshape(predictions, (predictions.shape[0],predictions.shape[1],1))
    future = []
    currentStep = predictions[-1:,:,:] #last step from the previous prediction

    for i in range(future_pred_count):
        currentStep = model.predict(currentStep) #get the next step
        print(currentStep)
        future.append(currentStep[0,0]) #store the future steps
        currentStep = sc.transform(currentStep)
        currentStep = np.reshape(currentStep, (currentStep.shape[0],currentStep.shape[1],1))
    return future



my_GRU_model, GRU_prediction = GRU_model(X_train, y_train, X_test, sc)

print(actual_pred_plot(GRU_prediction))
my_SVAE_model, SVAE_prediction = SVAE_model(X_train, y_train, X_test, sc)

print(actual_pred_plot(SVAE_prediction))

#GRU_future = future_predict(my_GRU_model,GRU_prediction,50)
#plt.plot(GRU_future)
plt.show()





