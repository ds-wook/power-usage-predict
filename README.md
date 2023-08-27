# power-usage-predict
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)  
2023 전력사용량 예측 AI 경진대회

## Setting
- CPU: i7-11799K core 8
- RAM: 32GB
- GPU: NVIDIA GeForce RTX 3090 Ti

## Cross Validation
+ Stratified Group KFold

## Ensemble Strategy
+ Ensemble Folds with Median

## Model Process
+ Boosting is All you need


## Requirements

By default, `hydra-core==1.1.0` was added to the requirements given by the competition.
For `pytorch`, refer to the link at https://pytorch.org/get-started/previous-versions/ and reinstall it with the right version of `pytorch` for your environment.

You can install a library where you can run the file by typing:

```sh
$ conda env create --file environment.yaml
```

## Run code

Code execution for the new model is as follows:

Running the learning code shell.

   ```sh
   $ sh scripts/run.sh
   ```

   Examples are as follows.

   ```sh
   python src/clustering.py

   for model in xgboost lightgbm catboost; do
    python src/train.py models=$model
    python src/predict.py models=$model
   done

   python src/teach.py
   python src/ensemble.py
   ```

## Benchmark
XGBoost-custom-loss: 5.5316
LightGBM-tweedie-loss: 5.8699
Categorical-Non-Catboost: 5.5252
Categorical-Catboost: 5.7216

The NN model has a significant performance difference compared to the boosting. Ensemble results also appeared to have a greater impact than other models.

## Submit

## Doesn't Work
+ meta feature: mean features
+ forcasting model: NBeat is not performance


## Reference

- [Model](https://www.sciencedirect.com/science/article/pii/S0169207021001874)
- [Loss](https://www.sciencedirect.com/science/article/pii/S0169207021001679)
- [Ensemble](https://www.kaggle.com/competitions/ventilator-pressure-prediction/discussion/276138)