# Machine Learning - Sliding window approach and train/test/validation data split from scratch

This script implements a sliding window approach with a dataset split procedure in training, testing and validation data sets.

The sliding window approach, is a method used mainly for time series problems, which consist in restructuring a time series dataset into an input matrix with lagged data and target features in order to be apply a supervised machine learning algorithm.
Using this approach it is possible to restructure a time series dataset as a supervised learning problem by using the value at the previous time step to predict the value at the next time-step.

This script is also capable to split data into training, testing and validation, and also a data normalization procedure.


Make sure to install the necessary libraries
* pip install -r requirements.txt
