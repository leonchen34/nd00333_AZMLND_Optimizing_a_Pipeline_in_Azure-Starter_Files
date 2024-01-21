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

This dataset contains data about a bank marketing campaign. We seek to predict if the client will subscribe to a term deposit with the bank, i.e. the feature column "y".

The best performing model was a VotingEnsemble picked by AutoML.

## Scikit-learn Pipeline

After data is downloaded from web, cleanup is performed. Basically the operation is to convert some features from string value to simple 0 or 1. 
<br>
To tune hyperparameter, RandomParameterSampling is used to sweep two parameters, inverse regularization strengh and max interation, which are used by logisticRegression classification alogorithm. 
<br>
I picked inverse regularization strength ranging from 0.5 to 1.5, and max iteration choice among 75, 100 and 125.

With RandomParameterSampling, it will not exahust through all possible parameters. 
This will make the trainig more efficient.

With BanditPolicy, it will drop the training if performance is 10% below best one. 
This will make training more efficient.

## AutoML

AutoML picked VotingEnsemble as the best model, which contains various algorithms with different weight.
Here I list two with the most weight.
<br>
MaxAbsScaler, LightGBM : weight 0.267 with the following hyper parameters:
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
MaxAbsScaler, LightGBM: weight 0.4 with the following hyper parameters:
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

The best accuracy for logisticRegression is 0.91067 with max iteration at 75 and regularized strength at 1.1398.
<br>
The best accuracy for AutoML is 0.91469 with VotingEnsemble.
<br>
With hyper parameter tuning, it trains a certain algorithm with different hyper parameters.
<br>
With AutoML, for certain class of problems, it tries different algorithms and select not only parameters, but also the best algorithm.

## Future work

We could create several pipline steps for the experiments. One such pipeline step is to do data download and clean up. After this step runs once, its output data will be used for both hyper parameter tunning step runs and AutoML step runs.

## Proof of cluster clean up

At the end of the Code, did a compute_target.get() before and after compute_target.delete(). The after get() proved the cluster clean up. 
I also double checked from azureml studio Compute->Compute Clusters and make sure the cluster was gone after executing compute_target.delete().

