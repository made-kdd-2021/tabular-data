import pandas as pd


def preprocess_cat_features_with_ohe(df: pd.DataFrame, cat_features: list) -> pd.DataFrame:
    """
    :param df: Датасет с необработанными категориальными признаками
    :param cat_features: Список категориальных признаков
    :return: Датасет с преобразованными категориальными признаками
    """
    df_copy = df.copy()
    for feature in cat_features:
        df_copy[feature] = df_copy[feature].astype(object)
    df_encoded = pd.get_dummies(df_copy)
    return df_encoded

