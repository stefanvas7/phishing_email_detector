## NLP pipeline implementing numerous model architectures for phishing email detection  

## An end-to-end pipeline using FNN, LSTM-RNN, and transformer based networks trained for phishing email detection onvolving data processing, model trainings, EDA and manual testing.

Haven't had the opportunity to train it yet, only the pipeline architecture.

I created this project to clean up the notebooks I had roughly created while writing my extended essay in the midst of the IBDP program. The research focus of my extended essay was to compare the efficiency of feedforward neural networks, recurrent neural networks with LSTM and transformer based networks for phishing email detection. 


## Technologies
 - Tensorflow keras
 - Pandas
 - Matplotlib
 - YAML
 - Logging 


## Features
 - Data handling 
    - CSV loading and train/validation/test split from pre-determined YAML configuration that produces tensorflow Dataset batches
 - Model registry
    - A mapping system for model configs to model instances for that configuration. Model ID's that allow for keras models to be saved/loaded
 - Configuration system
    - Functionality to create configurations as dataclasses or YAML files which setup hyperparameters to be used throughout the whole pipeline
 - CLI
    - A CLI entrypoint making it easier to launch trainings scripts with specific configurations. Or to run a saved model on an input
    

### Current limitations

Development was done locally on machine with a M2 processor, some libraries had compatibility issues with it so less up-to-date versions had to be used. Tensorflow-text was incompatible with the processor so a transformer model hasn't been implemented yet.
