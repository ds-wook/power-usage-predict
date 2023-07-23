# power-usage-predict
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)  
2023 전력사용량 예측 AI 경진대회

## Setting
- CPU: i7-11799K core 8
- RAM: 32GB
- GPU: NVIDIA GeForce RTX 3090 Ti

## Learner Architecture


## Model Architecture


## Ensemble Strategy

## Model Process

## Seed


## Requirements

By default, `hydra-core==1.2.0` was added to the requirements given by the competition.
For `pytorch`, refer to the link at https://pytorch.org/get-started/previous-versions/ and reinstall it with the right version of `pytorch` for your environment.

You can install a library where you can run the file by typing:

```sh
$ conda env create --file environment.yaml
```

## Run code

Code execution for the new model is as follows:

1. Put the basic data into the `input/upplus-recsys/` folder. When you execute the code that creates the data, the data for each `fold` star and `item_features`, `user_features` are stored in the `input/upplus-recsys/` folder.

   ```sh
   $ python scripts/make_dataset.py models=neucf
   ```

2. Running the learning code shell allows learning for each `fold`.

   ```sh
   $ sh scripts/train.sh
   ```

   Modifying the learning code shell will allow you to learn for each `fold`. You can also change the seed value. Examples are as follows.

   ```sh
   for seed in 22 94 95 96 99 3407
   do
       for fold in 0 1 2 3 4
       do
           python src/train.py models.fold=$fold models.seed=$seed
       done
   done
   ```

3. Running the prediction code shell saves the inferred values for each `fold` in the `output` folder.

   ```sh
   $ sh scripts/predict.sh
   ```

   Modifying the prediction code shell allows inference for each `fold`. And you need to set the seed value of the learned model. Examples are as follows.

   ```sh
   for seed in 22 94 95 96 99 3407
   do
       for fold in 0 1 2 3 4
       do
           python src/predict.py models.fold=$fold models.seed=94
       done
   done
   ```

## Benchmark

![Benchmark](https://user-images.githubusercontent.com/46340424/205429491-521460b6-1f0f-44f8-82c9-264f3f521be0.PNG)

The boosting model has a significant performance difference compared to the NN. Ensemble results also appeared to have a greater impact than other models.

## Submit

The file `submit` in the `output` folder is the file we finally submitted.
`best-lb-bootstrap-group-fold-enemble.csv` is an ensemble result of existing baseline models and models learned by 5Group KFold. In the case of `rank-nural-enemble.csv`, it is a result of adding existing baseline models and 5Group KFold models, and 5Fold models.

## Doesn't Work

- Boosting ranker model: Failed to process data to learn rank.
- Boosting binary model: binary learning took a long time. Not only did the training take a long time, but the model was not able to distinguish properly. This seems to be a problem caused by the lack of feature.
- Using the Boosting Ranker model after generation of candidates: It seems that it was not distinguished well because of the lack of features.
- As a result of using 4 layers, the difference in score between CV and LB seems to be overfitting.
- The Graph model took too long to learn.

## Reference

- [Model](https://www.sciencedirect.com/science/article/pii/S0169207021001874)