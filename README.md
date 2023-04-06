# ConSent

ConSent lets you train, evaluate and use deep learning models to classify chat dialogues, i.e., sequences of text that contains meta information such as *group*, *username*, *text*, *timestamp*, and a *code* or set of labels you are interested in. The use case in which ConSent was designed consists of classifying student chats according to a coding scheme. The data must be coded with high reliability (*k*>0.7) before you use ConSent. To start training a model, we suggest at least 5k coded messages to (hopefully) achieve satisfactory reliability to your use case.

 > ConSent is currently compatible with Python 3.9, Keras-Tensorflow 2.5, and TensorFlow Hub.

Below is the neural network we are initially using. We separate contextual information and sentence encoding features in two branches. The contextual information are auxiliary features that might help describe the situation of chat message at hand. We chose to use only binary auxiliary features because we expect that the network weights will be more easily optimized when inputs oscilate in a similar way. Here, the sentence encoder can be any pretrained language model capable of extracting embeddings. This way, we benefit from the increasingly high potential of transfer learning to NLP tasks to produce more reliable estimates. This architecture might benefit from changes for better results.

<img src="https://drive.google.com/uc?id=1aGcFBylS-KJjyrVMPG0JxBOrjTGExQLF" width="500">

As mentioned in [our paper](https://doi.org/10.1016/j.caeai.2023.100123), a model-centric improvement to ConSent would involve, for example, adding more auxiliary features. But also, data-centric improvements could be valuable, by correcting some bad labels in your data.


## Installation

It is recommended to install it in a conda environment.
```
conda create --name consent python=3.9
conda activate consent
```

ConSent is not on PyPi. You can easily install it by cloning the repository to your working directory and install it locally from source.
```
git clone https://github.com/adaj/consent.git
cd consent
pip install --user -r requirements.txt 
pip install --user .
```

To use ConSent, create a new .py (or .ipynb) file.

You will need to load some data. Please refer to `tests/test_data/Chats-EN-ConSent_dummy_data` to some examples of dummy data to play with.

Also, you will need to setup the configuration file. Refer to `examples/configs` for some examples.


### How to use

You can train a ConSent model or load one from a file. Here is a simple example:
```python
import json
import pandas as pd
from consent import Config, ConSent
import consent.utils as utils

# Load the example config
with open("examples/configs/L1__train.json", 'r') as f:
  config = Config(**json.load(f))

# Initialize
consent = ConSent(config)

# Load some data
data_df = pd.read_csv("tests/test_data/Chats-EN-ConSent_dummy_data.csv", 
                      index_col=0)

# Rename the target column to 'code'
data_df = data_df.rename(columns={"L1": "code"})

# Split train and test sets (for the dummy_data, keep test_size=0.5)
train_data_df, test_data_df = \
  utils.train_test_split(data_df, test_size=0.5)

# Train
consent.train(train_data_df)

# Generate a prediction (remember to use only the consent_pred)
sent_pred, consent_pred = \
  consent.predict_proba(dialog_id='4935ab', username='Milhouse', 
                        text='hey Bart are you there?')

# Optionally: Convert the output to the label with highest probability 
label = consent.onehot_decode(consent_pred)

# Generate predictions from dataframe
preds = test_data_df\
          .groupby('dialog_id')\
          .apply(consent.predict_sequence)

# Concatenate predictions for all groups in dataframe
preds = pd.concat(preds.apply(pd.DataFrame).values)\
          .reset_index(drop=True)
```

> If you want to test our pretrained models and join forces, sign up to the [waitlist form](https://forms.gle/CrvMb1Qz4A34BZTy9). 

#### Search for the best configuration with `hyperparameter_tuning`

We implemented `Handler` as a facade component for model selection. It loads the data and the config from file paths. The results can be tracked with [W&B](https://wandb.ai/site), be sure to set your W&B space in the config file.

For hyperparameter tuning, you will need to define a special config file (as `examples/configs/hyperparameter_tuning.json`). The `Handler` will test `n_iter` combinations of hyperparameters and report output metrics about it. 

Check the `model_selection` module for more details. 

```python
from consent.model_selection import Handler

run = Handler(
  data_file_path = TRAIN_DATA_PATH,
  config_file_path = "examples/configs/hyperparameter_tuning.json",
  random_state = 1
)

run.hyperparameter_tuning(
  val_size = 0.2,
  n_iter = -1,
  output_file_path = "examples/outputs/hyperparameter_tuning.csv"
) 
```


#### Get reliable performance estimation with `cross_validation`

```python
from consent.model_selection import Handler

run = Handler(
  data_file_path = TRAIN_DATA_PATH,
  config_file_path = "examples/configs/train.json",,
  random_state = 1
)
run.cross_validation(
  folds = 5,
  output_file_path = "examples/outputs/cross_validation.csv"
)
```


## How to contribute

This repository still needs some work.

We are very happy to receive and merge your contributions into this repository!

To contribute via pull request, follow these steps:

1.  Check if there is an open issue describing the feature.
2.  Create an issue describing the feature you want to work on.
3.  Write your code, tests and documentation, and format them with `black`.
4.  Create a pull request describing your changes.

Your pull request will be reviewed by a maintainer, who will get back to you about any necessary changes or questions. By sending a pull request, you agree with this repository's license agreement.

## Citation

BibTex:
```
@article{consent,
  title={Automated coding of student chats, a trans-topic and language approach},
  author={de Araujo, Adelson and Papadopoulos, Pantelis M. and McKenney, Susan and de Jong, Ton},
  journal={Computers and Education: Artificial Intelligence},
  volume={100123},
  year={2023},
  publisher={Elsevier},
  doi={10.1016/j.caeai.2023.100123},
  url={https://doi.org/10.1016/j.caeai.2023.100123}
}
```

## License

Licensed under GNU General Public License v3.0. Copyright 2023 Adelson de Araujo. [Copy of the license](https://github.com/adaj/consent/blob/master/LICENSE.md).

