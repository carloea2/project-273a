import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler

def preprocess_data(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame, config):
    cfg = config.data

    # Handle missing values
    num_imputer = SimpleImputer(strategy=cfg.preprocessing.numeric_imputer)
    cat_imputer = SimpleImputer(strategy=cfg.preprocessing.categorical_imputer, fill_value=cfg.preprocessing.unknown_label)

    # numeric columns
    if cfg.columns.numeric:
        df_train[cfg.columns.numeric] = num_imputer.fit_transform(df_train[cfg.columns.numeric])
        df_val[cfg.columns.numeric] = num_imputer.transform(df_val[cfg.columns.numeric])
        df_test[cfg.columns.numeric] = num_imputer.transform(df_test[cfg.columns.numeric])

    # categorical columns (low cardinality)
    for col in cfg.columns.categorical_low_card:
        df_train[col].replace(['?', 'None', None], np.nan, inplace=True)
        df_val[col].replace(['?', 'None', None], np.nan, inplace=True)
        df_test[col].replace(['?', 'None', None], np.nan, inplace=True)

        df_train[col] = cat_imputer.fit_transform(df_train[[col]])
        df_val[col] = cat_imputer.transform(df_val[[col]])
        df_test[col] = cat_imputer.transform(df_test[[col]])

        #if using unknown category, ensure it exists
        if cfg.preprocessing.use_unknown_category:
            # Any category in val/test not seen in train -> set to UNKNOWN

            train_vals = set(df_train[col].astype(str).unique())
            df_val[col] = df_val[col].astype(str).apply(lambda x: x if x in train_vals else cfg.preprocessing.unknown_label)
            df_test[col] = df_test[col].astype(str).apply(lambda x: x if x in train_vals else cfg.preprocessing.unknown_label)

    # Scale numeric features
    scaler = None
    if cfg.preprocessing.scaler:
        scaler = StandardScaler() if cfg.preprocessing.scaler == "standard" else RobustScaler()
        if cfg.columns_numeric:
            df_train[cfg.columns.numeric] = scaler.fit_transform(df_train[cfg.columns.numeric])
            df_val[cfg.columns.numeric] = scaler.transform(df_val[cfg.columns.numeric])
            df_test[cfg.columns.numeric] = scaler.transform(df_test[cfg.columns.numeric])

    return df_train, df_val, df_test, scaler