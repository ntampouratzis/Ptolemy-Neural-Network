# Ptolemy-Neural-Network
This repo contains a fully connected neural network implementation and also a GRU and LSTM network implementation in Ptolemy II.


# Fully-Connected Network 
A two hidden layer architecture trained for handwritten digit recognition.

It gets as input a 28x28 grayscale image of a handwritten digit and it produces the probabilities related to the 10 classes of digits from 0 to 9.

It is implemented in Ptolemy II using the Discrete-Event (DE) and the Synchronous Dataflow (SDF) Models of Computation (MoCs).

![test](https://github.com/ntampouratzis/Ptolemy-Neural-Network/blob/master/ptolemy-models/DE-SDF/HandWrittenDigitRecognition/de-sdf_handwrittenDigitImageRecognitionNeuralNet.png)


# GRU and LSTM Networks
These architectures are trained to get hourly weather measurements of 10 days as input and produce as output the prediction of the temperature of the next day at the same specific day-hour.

They are implemented in Ptolemy using an hierarchy of DE-ModalModel-SDF MoCs.

![test](https://github.com/ntampouratzis/Ptolemy-Neural-Network/blob/master/ptolemy-models/DE-Modal-SDF/GRU/temperaturePredictionUsingGRU.png | width=1226)

![test](https://github.com/ntampouratzis/Ptolemy-Neural-Network/blob/master/ptolemy-models/DE-Modal-SDF/LSTM/temperaturePredictionLSTM.png | width=1226)

