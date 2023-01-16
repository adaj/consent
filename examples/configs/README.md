# config

```
Attributes:
        architecture (str): A label to refer to the architecture in use.
        caller (str): A label to mention the context which the model is used.
        datasets (List[str]): Labels of datasets used to train the model.
        dataset_name (str): Labels of the dataset currently in use.
        code_name (str): Code label, i.e. the column in data_df with the labels.
        codes (List[str]): Codes/labels in the data to be learned.
        default_code (str): Default code label.
        wandb_project (str): Name of the W&B project to send results.
        load_weights (str): Load weights of layers from a pretrained model.
        language_featurizer (str): URL of TF-Hub featurizer to use.
        sent_hl_units (int): Number of units in the sent hidden layer.
        sent_dropout (float): Dropout rate for the sent hidden layer.
        consent_hl_units (int): Number of units in the consent hidden layer.
        lags (int): Number of previous codes to use.
        augmentation (dict): Data augmentation configuration.
        n_samples (int): Number of samples used during training.
        max_epochs (int): Max number of epochs to train the model.
        callback_patience (int): Epochs without improvement to stop train.
        learning_rate (float): Learning rate to train the model.
        batch_size (int): Batch size to train the model.
```

Here's a recommend config example to start with:

```
{
    "dataset_name": "Chats-<language,2 digits>-<topic>",
    "code_name": "L2C",
    "codes": ["IN", "AR", "AI", "AM", "NOS"],
    "default_code": "NOS",
    "wandb_project": "consent_tests",
    "language_featurizer": "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3",
    "sent_hl_units": 256,
    "sent_dropout": 0.5,
    "consent_hl_units": 32,
    "lags": 7,
    "augmentation": False,
    "max_epochs": 50,
    "callback_patience": 5,
    "learning_rate": 1e-3,
    "batch_size": 512
}
```

