# RESPIRATION_RATE_ESTIMATION
This repository contains the respiration rate estimation algorithm developed on the PPG Dalia dataset using Machine Learning and Deep Learning. The primary purpose of the work is to estimate the respiratory rate during ambulatory activities using ECG and accelerometer signals. Extraction of the respiration signal from the ECG is done by tracing the variations in Rpeak amplitude and RR interval. The respiratory signal from the accelerometer is extracted by using adaptive filtering. The respiration rate in BrPM from the three individual respiratory signals and reference respiratory signals is obtained by using the advanced counting algorithm.

# MACHINE LEARNING-BASED METHOD
This method involves the respiration rate estimation by fusing the features obtained from the respiration signal obtained from individual modality(Rpeak amplitude, RR interval, and accelerometer) using a machine learning algorithm. Features based on frequency domain analysis, morphology-based analysis are used for the fusion process. Frequency domain-based features are obtained using methods like Fourier transform, autocorrelation, autoregression. Morphology-based features obtained are coefficient of variations, mean peak to peak, true maxima true minima, coefficient of variation minima.

A vector formed by features obtained for the individual modality is fed as the independent variable to the ML model. The absolute error between the respiratory reference rate and the respiratory rate obtained from the respiration signal corresponds to individual modality is fed as the dependent variable. Different ML models like Ridge Regression, Random Forest Regression, Support Vector Regression, and Bayesian Ridge Regression have been tested. Hyperparameter tuning for each model is done by random search five-fold cross-validation.

# FILE DESCRIPTION OF MACHINE LEARNING-BASED METHOD
     ppg_dalia_data_extraction.py - This file extract the data from the each subjects of PPG Dalia Dataset and store them into the dictionary in the windows of 32 sec.
                                     The ECG signal, accelerometer signal, Reference Respiratory waveform, Rpeak locations and amplitude are extracted for each subject.
                                  
     edr_adr_signal_extraction.py - This file extract the respiratory signals from the raw signals and return them in the patches of 32 sec.
     rqi_extraction.py - This file takes the respiratory signals as an input and return the rqi corresponding to different methods.
     rr_extraction.py - This file takes respiratory signals as asn input and return the average breathing duration, other morphology based features.
     machine_learning.py - This files contains the definition of different ML models it also contain the definition of random k fold cross validation.
     testbench.py - This file contains the code related to the evaluation of the method. This file calls all the function and store the MAE adn RMSE into a .csv file.
     filters.py - Contains the definition of the filters used in the method.
     plots.py - Contain the script of Bland-Altman plot.
     preprocessing.py,extract_features.py -  These functions are used to frame the time stamps for interpolation, to be called in edr_adr_signal_extraction.py
     Ref_signal_Testbench.ipynb , Respiratory_signal_plot_testbench .ipynb -- Used for visualization of the plots.
 
# DEEP LEARNING-BASED METHOD
The deep learning-based approach includes developing a multitasking model based on the encoder, decoder, and IncResNet unit. The respiration signals from the individual modalities are fed as an independent variable to the multitasking model and the reference respiratory waveform and reference respiratory rate as the dependent variables. The model's output is the respiration waveform and average respiration rate; the respiration waveform can be used to extract the instantaneous respiration rate.

Different architectures based on different combination of inputs are being developed as part of the work. The detail of input-output combinations and the corresponding architecture is given in the figure below:
![image](https://user-images.githubusercontent.com/63348709/120197134-48cbaa80-c23e-11eb-94b8-c2ef57776b92.png)

The central architecture developed here is CONF-D, which takes respiration signals from three modalities and gives the final respiratory rate and respiratory signal. The respiratory signal output from the model can generate the instantaneous breathing peaks, which gives an instantaneous breathing rate. Other architecture was developed to thoroughly analyze the multitasking network's accuracy against the different sets of inputs.

# FILE DESCRIPTION OF DEEP LEARNING-BASED METHOD
     data_extraction.py - This file extract the data from the each subjects of PPG Dalia Dataset and store them into the dictionary in the windows of 32 sec.
                           The ECG signal, accelerometer signal, Reference Respiratory waveform, Rpeak locations and amplitude are extracted for each subject.
     resp_signal_extraction.py - This file extract the respiratory signals from the raw signals and return them in the patches of 32 sec.
     rr_extration.py - This file takes respiratory signals as asn input and return the average breathing duration , and relevent extremas.
     filters.py - Contains the definition of the filters used in the method.
     model.py - Contains the architecture of different model.
     test_model.py - This file test the model architecture.
     testbench4.py - This file trains the model after framing the torch dataframes.
     DL_eval.ipynb -  This file is for evaluation purpose.

# RESULTS
Evaluation of the multitasking model is done in terms of Absolute Error. The algorithm is evaluated against the individual modalities, and its performance is also evaluated during different activities. Box plots of the corresponding analysis are presented in the results section, which shows the algorithm's accuracy in estimating respiration rate during ambulatory activities.

