# SPAM ML

SPAM ML Library and application for training, serializing/deserializing, and sharing ML models for sports prediction.

## Design

SPAM_ML generally contains "models" and feature_engineering utilities to generate "named models" that can easily be persisted, loaded to disk, or shared via zip file.

A model generally has a few methods it creates

* `train`
  * This is generally a copy paste of all the feature engineering and training calls from a jupyter notebook. this is responsible for grabbing data from `spam_data`, doing final tranformations and drops, and calling fit or running a training loop
* `save`
  * This persists a model and any other objects worth persisting (data, scalers, etc) to files in `self.path`
* `load`
  * This loads objects from saved files in `self.path`
* `predict`
  * Makes predictions given an input

With this, a model can easily fit itself to data, and persist or load itself. In time this will likely become a bit more complicated as we allow more parameterization of these functions, but for now making a new model class with some copy pasted code is probably okay. Parameterization in the initializer (such as supplying an SKLearn Learner or dataset) can allow for some easy code sharing (or drop common feature engineering code into utilities)

When a model is created or loaded it is given a randomly generated "name". This name allows you to reload the model using the name, and if you would like to find where these models are stored ping me or trace the code.

## Extending

Likely copy the file `spam_ml/models/james_simple_linear.py` and change the `train()` method (and maybe a few others). Initially working in a jupyter notebook is usually easier if you're trying out new Features or model types that require different data processing.

## Caveats

If you change `spam_data` you likely need to reinstall it into the `spam_ml` project using

* `pdm add ../spam_data`

