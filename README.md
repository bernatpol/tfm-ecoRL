# TFM: Learning to Learn in Two-Armed Bandit Tasks with Interpretable Recurrent Neural Networks

This repository contains all the code for reproducing all plots and agents used in the TFM.

Before plotting, the agents need to be trained and evaluated. This can be done using the scripts in `src/training`, and it is computationally expensive. We can choose the amount of agents we want to train and the specific environment. Once we have all agents trained, we use the scripts in `src/results`, numbered according to the subsection where they are shown.

The specific definition of the neural networks are in `src/neural_networks`, where the agents using an LSTM, RNN, low-rank RNN and GRU are coded in PyTorch. Additionally, we find in `agents_1_LSTM_haiku.py` an LSTM agent coded in Jax used only in the beggining of the project, which is not used in the project but it is kept there for completeness.

Finally, in `src/utils` we find the code for evaluating the agents, logging and some plotting functions.