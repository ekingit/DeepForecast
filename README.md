# Sequential Data Modeling Weather Prediction

## Table of Contents 

- [Introduction](#introduction)
- [Derivations](#derivations)
- [Synthetic Data](#synthetic-data)
- [Weather Application](#weather-application)

## Introduction

This repository focuses on analyzing sequential data using deep learning models for sequence prediction, with a particular emphasis on recurrent models applied specifically to weather data.

Sequential data refers to data points that are ordered and dependent on previous entries in the sequence. This kind of data captures patterns and dependencies over time or across specific ordered steps, where each entry has a temporal or sequential relationship with the others. Examples include time-series data, such as daily stock prices or temperature measurements, where each value depends on the preceding values, as well as sequences in natural language, where the meaning of words is influenced by the words that come before and after them. 

Sequential data analysis and predictive modeling are a versatile field with a wide range of applications, such as weather forecasting, stock market prediction, sales forecasting, energy consumption tracking, patient health monitoring, natural language processing, and modeling physical or dynamical systems.

A powerful approach for predictive modeling in deep learning is the Recurrent Neural Network (RNN). RNNs contain a linear layer that embeds the sequence into a higher-dimensional space, followed by a recurrent layer. This recurrent layer, represented by a square matrix, encodes temporal dependencies through recurrence relations, with an activation function (typically tanh) applied afterward. While RNNs are effective, they present computational challenges, particularly when dealing with long sequences, as they often struggle with the vanishing gradient problem, which can hinder effective training.

To address these issues, Long Short-Term Memory (LSTM) networks were introduced as an RNN variant. LSTMs employ specialized gates—input, forget, and output gates—to manage information flow, enabling the model to capture long-term dependencies more effectively. A related architecture, the Gated Recurrent Unit (GRU), has a similar design but uses only reset and update gates to regulate information retention and forgetfulness, simplifying the structure while retaining much of the LSTM's functionality.

## Derivations

[nn.RNN](https://github.com/ekingit/DeepForecast/blob/main/derivations/1.Derivation_of_RNN.ipynb) and [nn.LSTM](https://github.com/ekingit/DeepForecast/blob/main/derivations/2.Derivation_of_LSTM.ipynb) were implemented from scratch to explore their underlying mechanics and compare outputs with PyTorch's implementations. By closely replicating the structure of these models, consistent results were observed between the custom implementations and those provided by nn.RNN and nn.LSTM. This comparison allowed for a deeper understanding of how sequential dependencies are handled by these models.

## Synthetic Data

RNN, LSTM, and GRU models were applied to synthetic data to gain insights into the strengths and weaknesses of each architecture. By using controlled datasets, it was possible to observe how each model performs under different conditions, highlighting their unique capabilities and limitations. This approach provided a clearer understanding of where each model excels or struggles, offering valuable perspective on their suitability for various types of sequential data.

### [Data = Sine](https://github.com/ekingit/DeepForecast/tree/main/synthetic_data/sinus_example)

**Models**

[`1.MLP`](https://github.com/ekingit/DeepForecast/blob/main/synthetic_data/sinus_example/1.MLP.ipynb): Sequence of length (1000) divided into seq_len=4 chunks to produce 996 examples are used to predict next value.

`2.RNN_analyzing_parameters`: An initial impulse `[1, 0, ..., 0]` is given to the RNN model and data is used in the loss function. Model parameters and the learning cycle are analyzed.

`3.Solution_with_state_spaces`: Use a state-space representation and coordinate transformation to produce the sine wave. Results are compared with that of RNN.
  
`4.LSTM`: An initial impulse `[1, 0, ..., 0]` is given to the LSTM model and data is given to the loss function.

`5.GRU`: An initial impulse `[1, 0, ..., 0]` is given to the GRU model and data is given to the loss function.

`6.Autoregressive_RNN`: Teacher forcing and closed loop methods for learning are discussed. A part (65 of 100) of data is given to the model to predict next steps. The predictions, then, used to make further predictions. 

`7.RNN_with_batches`: Sequence of length (1000) divided into seq_len=150 chunks to produce 850 examples are fed into the model with batches (of size 100) to predict next value.

### [Data = Sine with noise](https://github.com/ekingit/DeepForecast/tree/main/synthetic_data/sinus_w_noise)

`1.GRU`: An initial impulse `[1, 0, ..., 0]` is given to the GRU model and data is given to the loss function.

`2.LSTM_with_batches`: Sequence of length (1000) divided into seq_len=150 chunks to produce 850 examples are fed into the LSTM model with batches (of size 100) to predict next value.

`3.RNN+LSTM`:RNN model is used to predict data without noise. This prediction is used to decompose data into sine and noise components. Noise is predicted with an LSTM model.

`4.RNN+LSTM_transfer_learning`: Just like before, RNN model is used to predict data without noise. Then, LSTM model first learns the simpler model and after that, LSTM learns noisy data. We do this because LSTM fails to learn noisy data right away. That's why we learn a simpler data, and use it as initialization of the more complex data. 

## Weather Application

**Data** [Dataset](https://github.com/florian-huber/weather_prediction_dataset) contains daily recorded weather data from 2000 to 2010 for 18 European cities. For this project, we focus on the maximum temperature in Basel.

**Aim** Develop a model to predict 7-day weather forecasts from scratch.

**Challange** Identifying the right model architecture and hyperparameters to effectively minimize prediction loss.


**Approach:** 

*1. Local Model*

 - Train an autoregressive LSTM model that uses previous k-days data predict the next 7 days. 
 - Experiment with key hyperparameters, including input sequence length, hidden layer size, and the number of hidden layers, to identify the optimal configuration.

*2. Global Model*

 - Develop a recurrent model designed for long-range forecasting by leveraging extended historical data.
 - Conduct a thorough search for the best-performing hyperparameters to improve the accuracy of long-term predictions.
 
*3. Hybrid Model*

 - Use the global model to generate long-range weather predictions.
 - Refine these predictions with the local model to achieve greater accuracy for the immediate 7-day forecast.

 

**Results** Table1: Local Model, Hybrid Model 7 days prediction MSE

The dataset is split into training, validation, and test sets, covering 8 years, 1 year, and 1 year, respectively.

**1. Local Model**

 - This model uses an autoregressive approach to sequential prediction. Given the past seq_len values, we predict the next value, then iteratively use this prediction to forecast the following 7 days. (See Figure plot1 for illustration.)

![plot1](https://github.com/ekingit/DeepForecast/blob/main/weather_application/Results/local_LSTM_description.png)

$input = data[i:i+14]$,  $\forall i\in$ len(data)-seq_len-7+1

model = Autoregressive LSTM

model: (batch_size$\times 14$) $\rightarrow$ (batch_size$\times 7$)

loss = MSE Loss

loss: (batch_ size $\times$ 7)$\times$ (batch_ size$\times 7$)$\rightarrow 1$

table2

**Optimal parameters:**

* seq_len = 14

* hidden size = 20

* number of hidden layers = 3


**2. Global Model**

- This model uses a sine wave as input to train an RNN for predicting future values in a weather dataset. By capturing the periodic behavior of the sine wave, the model learns general seasonal patterns that aid in forecasting the weather data.

input = Sine wave with the period 365, len(input) = len(data)

model = RNN

$model: len(sine) \rightarrow len(data)$

loss = MSE Loss

$loss: len(sine)\times len(data)\rightarrow 1$

table 3

**Optimal parameters:**

* hidden size = 10

* hidden layers = 3

**3. Hybrid Model**

- This model combines long-range forecasting with a residual noise correction to improve weather predictions.

- Use the global model to generate long-range predictions based on a sine wave. This captures the periodic seasonal pattern in the data.

X_raw = weather_data

X_sine = sine wave with period 365

pretrain_model = RNN

pretrain_model: X_sine --> weather_data

- Calculate the difference between the long-range predictions and the original weather data to isolate the residuals, or "noise."

X_periodic = pretrain_model(X_sine)

X_noise = X_raw - X_periodic

- Train an autoregressive LSTM model on the extracted noise to correct for short-term deviations.

model = LSTM with batches seq_len=14

model: (batch_sizex14) of X_noise --> (batch_sizex7) of X_noise








