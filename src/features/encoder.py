import pandas as pd
from omegaconf import DictConfig
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


def categorize_tabnet_features(cfg: DictConfig, train: pd.DataFrame) -> tuple[list[int], list[int]]:
    """
    Categorical encoding
    Args:
        config: config
        train: dataframe
    Returns:
        dataframe
    """
    categorical_columns = []
    categorical_dims = {}

    label_encoder = LabelEncoder()

    for cat_feature in tqdm(cfg.features.categorical_features):
        train[cat_feature] = label_encoder.fit_transform(train[cat_feature].values)
        categorical_columns.append(cat_feature)
        categorical_dims[cat_feature] = len(label_encoder.classes_)

    features = [col for col in train.columns if col not in [cfg.data.target]]
    cat_idxs = [i for i, f in enumerate(features) if f in categorical_columns]
    cat_dims = [categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

    return cat_idxs, cat_dims
