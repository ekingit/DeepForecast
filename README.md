# Sequential Data, Modeling Weather Prediction

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

[`2.RNN_analyzing_parameters`](https://github.com/ekingit/DeepForecast/blob/main/synthetic_data/sinus_example/2.RNN_analyzing_parameters.ipynb): An initial impulse `[1, 0, ..., 0]` is given to the RNN model and data is used in the loss function. Model parameters and the learning cycle are analyzed.

[`3.Solution_with_state_spaces`](https://github.com/ekingit/DeepForecast/blob/main/synthetic_data/sinus_example/3.Solution_w_state_spaces.ipynb): Use a state-space representation and coordinate transformation to produce the sine wave. Results are compared with that of RNN.
  
[`4.LSTM`](https://github.com/ekingit/DeepForecast/blob/main/synthetic_data/sinus_example/4.LSTM.ipynb): An initial impulse `[1, 0, ..., 0]` is given to the LSTM model and data is given to the loss function.

[`5.GRU`](https://github.com/ekingit/DeepForecast/blob/main/synthetic_data/sinus_example/5.GRU.ipynb): An initial impulse `[1, 0, ..., 0]` is given to the GRU model and data is given to the loss function.

[`6.Autoregressive_RNN`](https://github.com/ekingit/DeepForecast/blob/main/synthetic_data/sinus_example/6.RNN_teacher_forcing%2Bclosed%20loop.ipynb): Teacher forcing and closed loop methods for learning are discussed. A part (65 of 100) of data is given to the model to predict next steps. The predictions, then, used to make further predictions. 

[`7.RNN_batched`](https://github.com/ekingit/DeepForecast/blob/main/synthetic_data/sinus_example/7.RNN_teacher_forcing_w_batches.ipynb): Sequence of length (1000) divided into seq_len=150 chunks to produce 850 examples are fed into the model with batches (of size 100) to predict next value.

### [Data = Sine with noise](https://github.com/ekingit/DeepForecast/tree/main/synthetic_data/sinus_w_noise)

[`1.GRU`](https://github.com/ekingit/DeepForecast/blob/main/synthetic_data/sinus_w_noise/1.GRU.ipynb): An initial impulse `[1, 0, ..., 0]` is given to the GRU model and data is given to the loss function.

[`2.LSTM_batched](https://github.com/ekingit/DeepForecast/blob/main/synthetic_data/sinus_w_noise/2.LSTM_with_batch.ipynb)`: Sequence of length (1000) divided into seq_len=150 chunks to produce 850 examples are fed into the LSTM model with batches (of size 100) to predict next value.

[`3.RNN+LSTM`](https://github.com/ekingit/DeepForecast/blob/main/synthetic_data/sinus_w_noise/3.RNN%2BLSTM.ipynb):RNN model is used to predict data without noise. This prediction is used to decompose data into sine and noise components. Noise is predicted with an LSTM model.

[`4.RNN+LSTM_transfer_learning`](https://github.com/ekingit/DeepForecast/blob/main/synthetic_data/sinus_w_noise/4.Transfer_Learning.ipynb): Just like before, RNN model is used to predict data without noise. Then, LSTM model first learns the simpler model and after that, LSTM learns noisy data. We do this because LSTM fails to learn noisy data right away. That's why we learn a simpler data, and use it as initialization of the more complex data. 

## [Weather Application](https://github.com/ekingit/DeepForecast/tree/main/weather_application)

### Summary

**Data:** [Dataset](https://github.com/florian-huber/weather_prediction_dataset) contains daily recorded weather data from 2000 to 2010 for 18 European cities. For this project, we focus on the maximum temperature in Basel.

**Aim:** Develop a model to predict 7-day weather forecasts from scratch.

**Challange:** Identifying the right model architecture and hyperparameters to effectively minimize prediction loss.

**Models:** 

*1. Local Model*

 - Train an autoregressive LSTM model that uses previous k-days data predict the next 7 days. 
 - Experiment with key hyperparameters, including input sequence length, hidden layer size, and the number of hidden layers, to identify the optimal configuration.

*2. Global Model*

 - Develop a recurrent model designed for long-range forecasting by leveraging extended historical data.
 - Conduct a thorough search for the best-performing hyperparameters to improve the accuracy of long-term predictions.
 
*3. Hybrid Model*

 - Use the global model to generate long-range weather predictions.
 - Refine these predictions with the local model to achieve greater accuracy for the immediate 7-day forecast.

 *4. Regularized Local Model*

 - Enhance the complexity and robustness of the local LSTM by introducing additional linear layers, layer normalization, and residual blocks.

**Results:** The local model, hybrid model, regularized local model and moving average are compared on the test set using mean squared error (MSE) as the loss function. Combining the local model with a periodicity of one year (365 days) resulted in improved model performance.
![Table1: Local Model, Hybrid Model, Moving Avarage - 7 days prediction MSE](https://github.com/ekingit/DeepForecast/blob/main/weather_application/Results/daily_loss.png)



### Models and Results

The dataset is split into training, validation, and test sets, covering 8 years, 1 year, and 1 year, respectively.

**[1. Local Model](https://github.com/ekingit/DeepForecast/blob/main/weather_application/1_0_LSTM_7_days_prediction.ipynb)**

 - This model uses an autoregressive approach for sequential prediction. The data is segmented with a window size of `seq_len + 7`, with each segment split into two parts: the first seq_len values serve as inputs, while the next 7 values are the target outputs. For each seq_len values, the model predicts the next value, then iteratively uses this prediction to forecast the following 7 days. (See Figure below for illustration.)


![plot1](https://github.com/ekingit/DeepForecast/blob/main/weather_application/Results/local_LSTM_description.png)

`input shape = stack(data[i:i+14])`,  $\forall i\in N$ where `N = len(data)-seq_len-7+1`

`model = Autoregressive LSTM`

`model: (batch_size x seq_len) --> (batch_size x 7)`

`loss = Mean Squared Error`

`loss: (batch_size x 7) x (batch_size x 7) --> 1`

**Remark:** This model captures local dependencies by analyzing patterns within a 15-day window to predict the next 7 days. It does not account for global dependencies, such as seasonal variations (e.g., hotter temperatures in summer and colder in winter) or the specific day of the year.

**[Optimal parameters](https://github.com/ekingit/DeepForecast/blob/main/weather_application/1_1_Parameter_opt.ipynb):**

![table2](https://github.com/ekingit/DeepForecast/blob/main/weather_application/Results/local_param_table.png)

* `seq_len = 14`

* `hidden size = 20`

* `number of hidden layers = 3`


**[2. Global Model](https://github.com/ekingit/DeepForecast/blob/main/weather_application/2_0_Periodic_RNN.ipynb)**

- This model uses a sine wave as input to train an RNN for predicting future values in a weather dataset. By capturing the periodic behavior of the data, the model learns general seasonal patterns that aid in forecasting the weather data.

`input = Sine wave with the period 365, len(input) = len(data)`

`model = RNN`

`model: len(sine) --> len(data)`

`loss = Mean Squared Error`

`loss: len(sine) x len(data) --> 1`

**Remark:** This model captures only global dependencies, as it does not use actual data inputs. Unlike the local LSTM model, it does not capture short-term patterns or local dependencies.

![table 3](https://github.com/ekingit/DeepForecast/blob/main/weather_application/Results/periodic_param_table.png)

**[Optimal parameters](https://github.com/ekingit/DeepForecast/blob/main/weather_application/2_1_Hyperparameter_opt.ipynb):**

* `hidden size = 10`

* `hidden layers = 3`

**[3. Hybrid Model](https://github.com/ekingit/DeepForecast/blob/main/weather_application/3_hybrid.ipynb)**

- This model combines long-range forecasting with a residual noise correction to improve weather predictions.

- Use the global model to generate long-range predictions based on a sine wave. This captures the periodic seasonal pattern in the data.

`X_raw = weather_data`

`X_sine = sine wave with period 365`

`pretrain_model = RNN with hidden_size=10, hidden_layers=3`

`pretrain_model: X_sine --> weather_data`

- Calculate the difference between the long-range predictions and the original weather data to isolate the residuals, or "noise."

`X_periodic = pretrain_model(X_sine)`

`X_noise = X_raw - X_periodic`

- Train an autoregressive LSTM model on the extracted noise to correct for short-term deviations.

`model = LSTM with seq_len=14, hidden_size=20, num_layers=3`

`model: (batch_size x 14) of X_noise --> (batch_size x 7) of X_noise`

**Remark:** This model captures both global/periodic dependencies and local patterns through separate models. However, it cannot capture correlated dependencies—such as seasonal differences in daily variability, where consecutive days might differ more in winter than in summer.


**[4. Regularized Local Model](https://github.com/ekingit/DeepForecast/blob/main/weather_application/3_hybrid.ipynb)**?

 Finally, we employ a sophisticated architecture—a local LSTM integrated with a residual block—to generate 7-day forecasts for comparison. This architecture is adapted with minor modifications from Luke Ditria’s work, as seen in his GitHub [repository](https://github.com/LukeDitria/pytorch_tutorials/blob/main/section12_sequential/solutions/Pytorch3_Autoregressive_LSTM.ipynb) and explained in detail in his [YouTube tutorial]((https://www.youtube.com/watch?v=lyUT6dOARGs)).
  
- This model incorporates linear layers that compress and decompress the input before applying the local LSTM. The output of the local LSTM then passes through a residual block to produce the next-day predictions. Originally introduced in computer vision, the residual block regularizes the model during backpropagation by mapping $x\mapsto f(x)+x$ where $f$ represents multiple layers in a deep neural network. This approach effectively addresses the vanishing gradient problem.


 1. The data is divided into segments with a window size of seq_len + 7, allowing seq_len to be used for predicting 7 days ahead, similar to the local LSTM approach.

 2. Each `batch_size` stack of `seq_len` segments is passed through two linear layers, with an ELU activation function in between, producing 64 nodes that serve as a "state space representation."

 3. These 64 nodes are then fed into the LSTM to generate state space representations for the next day's prediction.

 4. The resulting 64-dimensional state representation is passed through a residual block that first normalizes the data, then projects it onto a 32-dimensional space, and finally embeds it back into a 64-dimensional space. The original 64-dimensional input is also transformed by a square matrix and bias, and the two outputs are summed, enhancing regularization.

 5. The 64-dimensional state representation is projected onto a single dimension representing the next day's prediction.

 6. Using each day's prediction as input, the model iteratively forecasts the entire 7-day sequence.


 **Remark:** This model incorporates multiple regularization techniques, including layer normalization and residual layers, enhancing its robustness. These features make it adaptable for application to a variety of complex systems.

