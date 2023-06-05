# TAAoWPF

This code package implements various types of adversarial attacks on deep learning models for wind power forecasting from the paper "Targeted Adversarial Attacks on Wind Power Forecasts" by René Heinrich (Fraunhofer IEE, University of Kassel), Stephan Vogt (University of Kassel), Malte Lehna (Fraunhofer IEE, University of Kassel), and Christoph Scholz (Fraunhofer IEE, University of Kassel).

In particular, we studied the vulnerability of two different forecasting models to targeted, semi-targeted, and untargeted adversarial attacks. 
The first model was a Long Short-Term Memory (LSTM) network for forecasting the power production of individual wind farms. 
The second model was a Convolutional Neural Network (CNN) for forecasting wind power generation across Germany.

## Setup
All requirements are listed in `requirements.txt`. They can be installed via
```{python}
pip install -r requirements.txt
```

## Data
The LSTM model used wind speed forecasts for a single wind farm in the form of a univariate time series as input data. 
The target values represented the wind power measurements of the wind farm. 
For the robustness evaluation of the LSTM model, the wind speed and wind power data of 10 wind farms were used from the [GEFCom2014 dataset](https://www.sciencedirect.com/science/article/abs/pii/S0169207016000133), which is available [here](https://www.dropbox.com/s/pqenrr2mcvl0hk9/GEFCom2014.zip?dl=0).

The CNN model used wind speed forecasts for all of Germany in the form of weather maps as input data. 
The target values represented the total wind power generation in Germany. 
For the CNN model, the wind speed forecasts of the [ICON-EU model](https://www.dwd.de/DWD/forschung/nwv/fepub/icon_database_main.pdf) of the German Weather Service (DWD) were used, which are available [here](https://opendata.dwd.de/weather/nwp/icon-eu/). 
Data on wind power generation in Germany were obtained from the ENTSO-E website and can be downloaded [here](https://transparency.entsoe.eu/generation/r2/actualGenerationPerProductionType/show). 
For the experiments, these wind power measurements were normalized by the installed wind power capacity in Germany, which is available [here](https://transparency.entsoe.eu/generation/r2/installedGenerationCapacityAggregation/show).

## Instructions for training the models

LSTM model
1. Perform hyperparameter tuning for the LSTM model using the Jupyter notebook /models/lstm/hyperparameter-tuning_lstm.ipynb.
2. Train the LSTM model with ordinary training using the Jupyter notebook /models/lstm/training_lstm.ipynb and setting the parameter p_adv_training to 0.
3. Train the LSTM model with adversarial training using the Jupyter notebook /models/lstm/training_lstm.ipynb and setting the parameter p_adv_training to 1.

CNN model
1. Train the CNN model with ordinary training using the Jupyter notebook /models/cnn/training_cnn.ipynb and setting the parameter p_adv_training to 0.
2. Train the CNN model with adversarial training using the Jupyter notebook /models/cnn/training_cnn.ipynb and setting the parameter p_adv_training to 1.

## Instructions for evaluating the overall robustness of the models

LSTM model
1. Calculate various metrics to quantify the overall robustness of the model using the Jupyter Notebook /robustness_evaluation/lstm/robustness_evaluation_lstm.ipynb.
2. Display the means and standard deviations of the robustness metrics in tables using the Jupyter Notebook /visualization/lstm/tables_robustness_lstm.ipynb.

CNN model
1. Calculate various metrics to quantify the overall robustness of the model using the Jupyter Notebook /robustness_evaluation/cnn/robustness_evaluation_cnn.ipynb.
2. Display the means and standard deviations of the robustness metrics in tables using the Jupyter Notebook /visualization/cnn/tables_robustness_cnn.ipynb.
3. Visualize the impact of targeted adversarial attacks on the test data samples in the form of a box plot using the Jupyter Notebook /visualization/cnn/visualization_boxplot_cnn.ipynb.

## Instructions for evaluating the impact of adversarial attacks on individual model predictions

LSTM model
1. Compute various information about the impact of a targeted adversarial attack on a single prediction of the model using the Jupyter notebook /robustness_evaluation/lstm/robustness_example_attack_lstm.ipynb.
2. Visualize the impact of the attack on the prediction and the input data using the Jupyter notebook /visualization/lstm/visualization_example-attack_lstm.ipynb.

CNN model
1. Compute various information about the impact of a targeted adversarial attack on a single prediction of the model using the Jupyter notebook /robustness_evaluation/cnn/robustness_example_attack_cnn.ipynb.
2. Visualize the impact of the attack on the prediction and the input data using the Jupyter notebook /visualization/cnn/visualization_example-attack_cnn.ipynb.

## License

[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt)
