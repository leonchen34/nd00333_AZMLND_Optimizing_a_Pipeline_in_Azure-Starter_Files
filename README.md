# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary
**In 1-2 sentences, explain the problem statement: e.g "This dataset contains data about... we seek to predict..."**

This dataset contains data about a bank marketing campaign. We seek to predict if the client will subscribe to a term deposit with the bank, i.e. the feature column "y".

**In 1-2 sentences, explain the solution: e.g. "The best performing model was a ..."**

The best performing model was a VotingEnsemble picked by AutoML.

## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**

After data is downloaded from web, cleanup is performed. Basically the operation is to convert some features from string value to simple 0 or 1. 
<br>
To tune hyperparameter, RandomParameterSampling is used to sweep two parameters, inverse regularization strengh and max interation, which are used by logisticRegression classification alogorithm. 
<br>
I picked inverse regularization strength ranging from 0.5 to 1.5, and max iteration choice among 75, 100 and 125.

**What are the benefits of the parameter sampler you chose?**

With RandomParameterSampling, it will not exahust through all possible parameters. 
This will make the trainig more efficient.

**What are the benefits of the early stopping policy you chose?**

With BanditPolicy, it will drop the training if performance is 10% below best one. 
This will make training more efficient.

## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**

AutoML picked VotingEnsemble as the best model, which contains various algorithms with different weight.
Here I list two with the most weight.
<br>
MaxAbsScaler, LightGBM : weight 0.2 with the following hyper parameters:
<br>
        "boosting_type": "goss",
        "colsample_bytree": 0.5944444444444444,
        "learning_rate": 0.026323157894736843,
        "max_bin": 310,
        "max_depth": -1,
        "min_child_weight": 3,
        "min_data_in_leaf": 0.00001,
        "min_split_gain": 0.7894736842105263,
        "n_estimators": 50,
        "num_leaves": 131,
        "reg_alpha": 0.3684210526315789,
        "reg_lambda": 1,
        "subsample": 1
<br>
<br>
MaxAbsScaler, LightGBM: weight 0.467 with the following hyper parameters:
<br>
        "spec_class": "sklearn",
        "class_name": "LightGBMClassifier",
        "module": "automl.client.core.common.model_wrappers",
        "param_args": [],
        "param_kwargs": {
            "min_data_in_leaf": 20
        },
        "prepared_kwargs": {}


## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

The best accuracy for logisticRegression is 0.9102681 with max iteration at 75 and regularized strength at 1.144949.
<br>
The best accuracy for AutoML is 0.9150531 with VotingEnsemble.
<br>
With hyper parameter tuning, it trains a certain algorithm with different parameters.
<br>
With AutoML, for certain class of problems, it tries different algorithms and select not only parameters, but also the best algorithm.

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

One thing I noticed that train.py downloads and cleans data every time. I think it is more efficient to download and clean once at project file and save into datastore as x.csv and y.csv. Then from train.py, just read from data store.

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**

has code compute_target.delete()

