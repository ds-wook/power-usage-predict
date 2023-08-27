for model in xgboost lightgbm catboost; do
    python src/train.py models=$model
    python src/predict.py models=$model
done

python src/ensemble.py
