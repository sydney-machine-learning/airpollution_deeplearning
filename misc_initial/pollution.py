                                                                                                                                                                                                                                                                        from math import sqrt
import numpy as np
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from pandas import read_csv
from datetime import datetime
from pandas import read_csv
from matplotlib import pyplot
from numpy import array
import matplotlib.pyplot as plt
import datetime

def parse(x):
	return datetime.strptime(x, '%Y %m %d %H')



def data_arrange(area_name):
	dataset = read_csv('data/'+area_name+'.csv',index_col=0)
	dataset.columns = ['AT','BP','PM10','Benzene','Toluene','NH3','NO','NO2','NOx','RH','SR','WS','WD','Ozone','SO2','CO','PM2.5']
	dataset.index.name = 'Date'
	dataset['PM2.5'].fillna(0, inplace=True)
	print(dataset.head(5))
	dataset.to_csv('pollution'+area_name+'.csv')

def visualize_columns(values):
	# specify columns to plot
	groups=[]
	for i in range(values.shape[1]):
    	groups.append(i)
# groups = [0, 1, 2, 3, 5, 6, 7,8,9,10,11,12,13,14,]
# print(groups)
	i = 1
# plot each column
	pyplot.figure()
	for group in groups:
    	pyplot.subplot(len(groups), 1, i)
    	pyplot.plot(values[:, group])
#     print(group)
    	pyplot.title(dataset.columns[group], y=1, loc='right')
   		i += 1
def data_prep(area_name):
	data_arrange(area_name)
	dataset = read_csv('pollution'+area_name'.csv', header=0, index_col=0)
	values = dataset.values
	print(values.shape)
	visualize_columns(values)
	
	

def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
    # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out-1
    # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)





def MODEL(train_X,test_X,train_y,test_y,val_X,val_y,n_out,n_hours,n_features):
    model = Sequential()
    model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_hours,n_features)))
    model.add(LSTM(100, activation='relu'))
    model.add(Dense(n_out))
    model.compile(optimizer='adam', loss='mse')
# fit model
    model.fit(train_X, train_y, epochs=200, batch_size=20, validation_data=(val_X, val_y), verbose=0, shuffle=False)
    yhat11 = model.predict(test_X, verbose=0)
    print("RMSE error using ",n_hours," state size to predict ",n_out," step is =", np.sqrt(((np.sum((yhat11-test_y)**2,axis=0)))))
    return(np.sqrt(((np.sum((yhat11-test_y)**2,axis=0)))))


def plot_scatter(n_out_max,n_hour_max,r):
	xlist = np.linspace(1, n_out_max, n_out_max)
	ylist = np.linspace(1, n_out_max, n_out_max)
	X, Y = np.meshgrid(xlist, ylist)
# Z = np.sqrt(X**2 + Y**2)
	fig,ax=plt.subplots(1,1)
	cp = ax.contourf(X, Y, r)
	fig.colorbar(cp) # Add a colorbar to a plot
	ax.set_title('Number of steps used in multivariate analysis x ')
#ax.set_xlabel('x (cm)')
	ax.set_ylabel('Number of steps predicted y')
	plt.show()
	print("Normalized RMSE Test")




def show_results(r):
    x = datetime.datetime.now()
    u=np.mean(r,axis=0)
    sigma=np.std(r,axis=0)
    lower_bar= u-1.96*(sigma/(np.sqrt(r.shape[0])))
    upper_bar= u+1.96*(sigma/(np.sqrt(r.shape[0])))
    barWidth = 0.3
    bars1=u
    rl= np.arange(len(bars1))+1
    plt.bar(rl, bars1, width = barWidth, color = 'red', edgecolor = 'black', yerr=1.96*(sigma/(np.sqrt(r.shape[0]))), capsize=7, label='LSTM with using 10 multivariate steps for 30 iterations')
    plt.ylabel('mean of normalized RMSE for 30 iterations')
    plt.xlabel('Number of steps')
    plt.legend()
    plt.show()
    np.savetxt("10i_"+str(r.shape[1])+"o_LSTM_"+str(r.shape[0])+"i.csv"+str(datetime.datetime.now()), r, delimiter=",")    



def main(n_out_max,n_hour_max):
	data_prep(area_name)

    dataset = read_csv('pollution'+area_name+'.csv', header=0, index_col=0)
    values = dataset.values  
    values = values.astype('float32')
# normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    print(values.shape)
    scaled = scaler.fit_transform(values)
    print(len(scaled))
    result= []
	for i,k in enumerate(out_range):
    	for j in range(1,31):
        	for m in range(10,11):
            	n_out=k
            	n_hours=m
            	print("Number of steps predicted= ",n_out,"Number of steps Multivariate threads ussed= ",n_hours, "iteration number= ",j)
            	X, y = split_sequences(scaled, n_hours, n_out)
            	train_X, train_y = X[0:300,:,:], y[0:300,:]
            	val_X, val_y = X[300:330,:,:], y[300:330,:]
            	test_X, test_y = X[330:,:,:], y[330:,:]
            	print("train_X.shape, train_y.shape, test_X.shape, test_y.shape=",train_X.shape, train_y.shape, test_X.shape, test_y.shape)
            	print("val X, val y",val_X.shape, val_y.shape)
            	n_features = X.shape[2]
            	error=MODEL(train_X,test_X,train_y,test_y,val_X,val_y,n_out,n_hours,n_features)
#             result[i,m-1]=result[i,m-1]+error
            	result.append(error)
      
    show_reults(result)
    return result
            
            
            




            
         
            