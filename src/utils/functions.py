import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import UserDict
    
def calculate_mase(n,seas,horizon,y,forecast_series):
    # n - number of instances in the training data
    # seas - length of the seasonal cycle (1 day ahead, seas = 48)
    # horizon - number of steps in predicting horizon 
    # y - training time series of the forecasted variable
    # forecast_series - forecasted time series
    errors = (n - seas) / horizon * (np.sum(np.abs(y[n:n+horizon] - forecast_series), axis=0)
       / np.sum(np.abs(y[seas:n] - y[:n-seas]), axis=0))
    return np.mean(errors)

def calculate_mase2(pred_array, true_array, train_naive_pred_array, train_true_array):
    # pred_array is a vector of predictions
    # true_array is a vector of actual values
    # train_naive_pred_array is a vector of predictions done by a naive seasonal method for train subset
    # train_true_array is a vector of actuals for train subset
    mae_test = np.sum(np.abs(pred_array - true_array)) / true_array.size
    mae_train = np.sum(np.abs(train_naive_pred_array - train_true_array)) / train_true_array.size
    return mae_test / mae_train



def MASE(training_series, testing_series, prediction_series):
    """
    Computes the MEAN-ABSOLUTE SCALED ERROR forcast error for univariate time series prediction.

    See "Another look at measures of forecast accuracy", Rob J Hyndman

    parameters:
        training_series: the series used to train the model, 1d numpy array
        testing_series: the test series to predict, 1d numpy array or float
        prediction_series: the prediction of testing_series, 1d numpy array (same size as testing_series) or float
        absolute: "squares" to use sum of squares and root the result, "absolute" to use absolute values.

    """
    # print "Needs to be tested."
    n = training_series.shape[0]
    horizon = training_series.shape[1]
    acc = 0
    for i in range(horizon):
      d = np.abs(  np.diff( training_series[:,i]) ).sum()/(n-horizon)
      errors = np.abs(testing_series[:,i] - prediction_series[:,i] )
      acc += errors.mean()/d
    return acc/horizon
    
def validation(forecasted, real, parameter):
    ''' 
    compute some important parameters to compare forecasting results
    '''
    value = 0
    value_1 = 0
    value_2 = 0

    if parameter == 'SMAPE':
        for i in range(len(forecasted)):
            if real[i] + forecasted[i] == 0:
                value += 0
            else: 
                value += ((abs(real[i] - forecasted[i])) / (real[i] + forecasted[i])) * 100
        final_value = value / len(forecasted)  

    elif parameter == 'MAPE':
        for i in range(len(forecasted)):
            if real[i] == 0:
                value += 0
            else: 
                value += (abs(real[i] - forecasted[i]))/real[i]
        final_value = value / len(forecasted) * 100

    elif parameter == 'RMSE':
        for i in range(len(forecasted)):
            value += (real[i] - forecasted[i]) ** 2
        final_value = (value / len(forecasted)) ** (1 / 2) 

    elif parameter == 'MAE':
        for i in range(len(forecasted)):
            value += abs(real[i] - forecasted[i])
        final_value = value / len(forecasted)
        
    elif parameter == 'R':
        for i in range(len(forecasted)):
            value += (real[i] - np.mean(real)) * (forecasted[i] - np.mean(forecasted))
            value_1 += (real[i] - np.mean(real)) ** 2
            value_2 += (forecasted[i] - np.mean(forecasted)) ** 2

        if value_1 == 0 or value_2 == 0:
            final_value = 100
        else:
            final_value = (value / ((value_1 ** (1 / 2)) * (value_2 ** (1 / 2))))*100

    return float(final_value)


class TimeSeriesTensor(UserDict):
    """A dictionary of tensors for input into the RNN model.

    Use this class to:
      1. Shift the values of the time series to create a Pandas dataframe containing all the data
         for a single training example
      2. Discard any samples with missing values
      3. Transform this Pandas dataframe into a numpy array of shape 
         (samples, time steps, features) for input into Keras
    The class takes the following parameters:
       - **dataset**: original time series
       - **target** name of the target column
       - **H**: the forecast horizon
       - **tensor_structures**: a dictionary discribing the tensor structure of the form
             { 'tensor_name' : (range(max_backward_shift, max_forward_shift), [feature, feature, ...] ) }
             if features are non-sequential and should not be shifted, use the form
             { 'tensor_name' : (None, [feature, feature, ...])}
       - **freq**: time series frequency (default 'H' - hourly)
       - **drop_incomplete**: (Boolean) whether to drop incomplete samples (default True)
    """

    def __init__(self, dataset, target, H, tensor_structure, freq='H', drop_incomplete=True):
        self.dataset = dataset
        self.target = target
        self.tensor_structure = tensor_structure
        self.tensor_names = list(tensor_structure.keys())

        self.dataframe = self._shift_data(H, freq, drop_incomplete)
        self.data = self._df2tensors(self.dataframe)

    def _shift_data(self, H, freq, drop_incomplete):

        # Use the tensor_structures definitions to shift the features in the original dataset.
        # The result is a Pandas dataframe with multi-index columns in the hierarchy
        #     tensor - the name of the input tensor
        #     feature - the input feature to be shifted
        #     time step - the time step for the RNN in which the data is input. These labels
        #         are centred on time t. the forecast creation time
        df = self.dataset.copy()

        idx_tuples = []
        for t in range(1, H+1):
            df['t+'+str(t)] = df[self.target].shift(t*-1, freq=freq)
            idx_tuples.append(('target', 'y', 't+'+str(t)))

        for name, structure in self.tensor_structure.items():
            rng = structure[0]
            dataset_cols = structure[1]
            #print(rng)
            #print(dataset_cols)

            for col in dataset_cols:

            # do not shift non-sequential 'static' features
                if rng is None:
                    df['context_'+col] = df[col]
                    idx_tuples.append((name, col, 'static'))

                else:
                    for t in rng:
                        sign = '+' if t > 0 else ''
                        shift = str(t) if t != 0 else ''
                        period = 't'+sign+shift
                        shifted_col = name+'_'+col+'_'+period
                        df[shifted_col] = df[col].shift(t*-1, freq=freq)
                        idx_tuples.append((name, col, period))

        df = df.drop(self.dataset.columns, axis=1)
        idx = pd.MultiIndex.from_tuples(idx_tuples, names=['tensor', 'feature', 'time step'])
        df.columns = idx

        if drop_incomplete:
            df = df.dropna(how='any')

        return df

    def _df2tensors(self, dataframe):

        # Transform the shifted Pandas dataframe into the multidimensional numpy arrays. These
        # arrays can be used to input into the keras model and can be accessed by tensor name.
        # For example, for a TimeSeriesTensor object named "model_inputs" and a tensor named
        # "target", the input tensor can be acccessed with model_inputs['target']

        inputs = {}
        y = dataframe['target']
        y = y.to_numpy()
        inputs['target'] = y

        for name, structure in self.tensor_structure.items():
            rng = structure[0]
            cols = structure[1]
            tensor = dataframe[name][cols].to_numpy()
            if rng is None:
                tensor = tensor.reshape(tensor.shape[0], len(cols))
            else:
                tensor = tensor.reshape(tensor.shape[0], len(cols), len(rng))
                tensor = np.transpose(tensor, axes=[0, 2, 1])
            inputs[name] = tensor

        return inputs

    def subset_data(self, new_dataframe):

        # Use this function to recreate the input tensors if the shifted dataframe
        # has been filtered.

        self.dataframe = new_dataframe
        self.data = self._df2tensors(self.dataframe)
        
    