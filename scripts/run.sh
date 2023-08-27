python src/clustering.py

for model in xgboost lightgbm catboost; do
    python src/train.py models=$model
    python src/predict.py models=$model
done
python src/teach.py
python src/ensemble.py
