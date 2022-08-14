# airpollution_deeplearning
Deep learning for prediction of air pollution trend for four monitoring stations in Delhi, India. 

The experiments feature different deep learning models such as FNN and LSTM based models. 
The experiments also feature multivariate and univariate analysis using different training strategies 
with Bi-LSTM model (best) for multistep ahead prediction of PM2.5 values. The experiments further provide one month
ahead forecast of PM2.5 values with uncertainty quantification.

## Code
We have a unified code for all four stations with proper comments.
* The python notebook for implementation can be found here: [Implement](https://github.com/sydney-machine-learning/airpollution_deeplearning/blob/master/dl_implementation.ipynb)
* The python notebook for data visualization can be found here: [Visualize](https://github.com/sydney-machine-learning/airpollution_deeplearning/blob/master/data_visualization.ipynb) 
  
## Data
The updated data used in experiments can be found here: [DATA](https://github.com/sydney-machine-learning/airpollution_deeplearning/tree/master/data_new)

## Experiments
Sample results (30 runs) for different stations using different models and training strategies can be found here: [results](https://github.com/sydney-machine-learning/airpollution_deeplearning/tree/master/results). The analysis and plots for different monitoring stations can be found here: [plots](https://github.com/sydney-machine-learning/airpollution_deeplearning/tree/master/plots)

## Data Addon
Two more experiments have been include with two different time series data related to climate parameters in Delhi:
* The climate data has parameters including Mean Temperature, Wind Speed, Mean Pressure and Humidity for a particular day in Delhi. The training data contains data for each day from 1 January, 2013 till 1 January, 2017. The test data contains data for each day from 1 January, 2017 till 24 April, 2017. In this experiment we try to model the humidity values (y) using different multivariate time series analysis including both univariate and multivariate time series. [Climate_Data](https://github.com/animeshrdso/airpollution_deeplearning/tree/master/data_new/delhi_climate)
* The second data is the rainfall data which contains the average rainfall for different months in a year for Delhi and NCR region as a whole. The entire data spans the rainfall data from 1901 to 2017. In this experiment we try to model the rainfall values for the month of July in each year. Both multivariate and univariate analysis has been performed. The time series data has been split in ratio (60:20:20) for train, validation and test data respectively. [Rainfall_Data](https://github.com/animeshrdso/airpollution_deeplearning/blob/master/data_new/delhi_climate/Sub_Division_IMD_2017.csv)
