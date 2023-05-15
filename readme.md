# DA-HMTS: A Dual-Stage Attention-Based Hybrid Model for Multivariate Time Series Forecasting

### Abstract
> In order to improve the prediction accuracy, computational efficiency, and interpretability of time series models, as well as to solve the problem of ineffective time series feature fusion caused by feature redundancy and high nonlinearity from multiple data sources in traditional methods for multivariate time series, we designed a dual attention mechanism to optimize the Temporal Convolutional Network (TCN) and Long Short-Term Memory (LSTM) models, and proposed a multivariate time series prediction hybrid model named DA-HMTS. This hybrid model consists mainly of two stages of attention mechanisms, including feature attention and temporal attention. In the first stage, we dynamically select variables using a designed Variable Selection Module (VSM) to avoid feature redundancy, and use a TCN network structure with multiple parallel convolution kernels and feature attention mechanisms to improve the efficiency of extracting multivariate time series features. In the second stage, we use LSTM networks to freely capture key time periods for automatic encoding and extract row vector features from the hidden state matrix. We also design a one-dimensional convolutional network to achieve differentiated weight allocation for different variables at different time steps. Finally, we feed the output feature vectors into a fully connected layer and design a quantile loss function to achieve interval prediction. By comparing experimental results on four different real-world datasets, we found that the DA-HMTS model performs better than some baseline models.

## DA-HMTS Architecture
The network structure of this model mainly includes feature and temporal attention mechanisms acting on the TCN and LSTM network layers, which can be used for prediction in the following steps:
First stage is feature extraction. Based on the maximum joint mutual information analysis of the correlation between input features and output targets in historical data, we adaptively adjust the input information of the TCN model according to the importance of features. We also design a parallel network with multiple convolutional kernels to improve the computational efficiency of the TCN network and extract important features from multivariate time series data in parallel.
Second stage is temporal extraction. The feature sequence output by the TCN layer is input into the LSTM layer, which improves the efficiency of the LSTM hidden layer memory unit processing. At the same time, the temporal attention mechanism is introduced into the LSTM to extract temporal relationships between multivariate sequences, thereby achieving analysis and prediction of longer period sequences.
Finally, the output of the LSTM layer is sent to the fully connected layer, and the target sequence interval prediction is achieved by combining with the quantile loss function.
RESULTS AND DISCUSSIONS
## RESULTS AND DISCUSSIONS
### Dataset
This article uses PyCharm compilation software and implements the DA-HMTS model using Python 3.10 on the Windows 10 ×64 operating system. Our dataset was selected as a challenging multivariate time series data prediction, which can reflect commonly existing features. In order to establish a baseline and position related to previous academic work, we not only used a self-built landslide monitoring dataset but also evaluated the performance of the model on datasets in the fields of electric power, transportation, and weather. Here is a brief description of each dataset.
Electricity: It contains six months of electricity consumption data for a household (15-min sampling rate), gathered between January 2007 and June 2007. The data includes information on global active power, global reactive power, voltage, global intensity, sub-metering 1 (kitchen), sub-metering 2 (laundry room), and sub-metering 3 (electric water heater and air conditioner). With 260,640 measurements in total, this dataset can provide crucial insights into understanding household electricity consumption.
Traffic: This dataset describes the occupancy rates (between 0 and 1) measured by 862 different sensors on San Francisco Bay area freeways. The data are sampled every 1 h from 2015 to 2016. Following [50], we convert the data to reflect hourly consumption.
Weather: The dataset describes the weather data provided by the Moratazarzal meteorological station in Madrid. This data is hourly data from January 2019 to January 2022.
Landslide: This dataset is a self-built multidimensional time series dataset that describes the geological disaster landslide monitoring data of Yangchan Village in Huangshan City. The data covers the period from April 2022 to March 2023 at a daily level.
### Network parameter tuning
Given that the hyperparameters that have a significant impact on the DA-HMTS network model
                                                        Tab.1. Network Model Parameter Setting
|parameters|Training Results|
|:---:|:---:|
|convolution kernel size|5×1|
|number of convolution kernels|64|
|epoch|2000|
|activation function|ReLU|
|loss function|mse|
|adopting an optimizer|Adam|
|learning rate|0.001|
|number of hidden layer nodes|64|
|number of hidden layers|4|
|expansion factor|2|
### Model performance evaluation
Prediction accuracy，Computational efficiency，Model interpretability
![landslide](http://aiitbeidou.cn:8080/DA-HMTS/img1.png)
![landslide](http://aiitbeidou.cn:8080/DA-HMTS/img2.png)
![landslide](http://aiitbeidou.cn:8080/DA-HMTS/img3.png)
![landslide](http://aiitbeidou.cn:8080/DA-HMTS/img4.png)
![landslide](http://aiitbeidou.cn:8080/DA-HMTS/img5.png)
### Conclusion
DA-HMTS model is a high-performance hybrid model for multivariate time series prediction based on dual attention mechanisms. First, the model designs a variable selection module that adaptively selects feature variables based on the information gain rate criterion. Second, the feature attention mechanism and the temporal attention mechanism are applied to the time convolutional neural network and the long short-term memory network layer, respectively. Finally, a quantile loss function is designed to achieve interval prediction. Additionally, the proposed model is compared and analyzed with other methods using different datasets from various fields, and the following conclusions are drawn: 
(1) The DA-HMTS model uses feature attention mechanism and temporal attention mechanism to optimize the output features of the TCN model and the output of the LSTM model, which improves the prediction accuracy of key moments. 
(2) Compared with other methods, the DA-HMTS model has the best overall performance, reducing the MSE by at least 55.24% and the MAE by at least 37.60% under the same conditions. 
(3) The ablation experiment results show that removing different modules will lead to different degrees of loss increase, indicating that the model structure optimization is reasonable. 
(4) The DA-HMTS model has good interpretability and can accurately identify significant events in critical time periods (such as identifying landslides during the rainy season). 
(5) The DA-HMTS model has high computational efficiency and achieves good results in terms of computational efficiency, prediction accuracy, and interpretability compared to other deep learning models with the same input sample size. Although the XGBoost and ARIMAX methods have faster training efficiency than the DA-HMTS model, their predictive accuracy and interpretability are relatively poor.
