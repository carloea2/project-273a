import pandas as pd


def build_vocab(df: pd.DataFrame, col: str, add_unknown: bool = True, unknown_label: str = "UNKNOWN"):
    """Build a vocabulary (unique values list) for a column"""

    values = df[col].astype(str).unique().tolist()
    if add_unknown:
        if unknown_label not in values:
            values = [unknown_label] + values
    return {val: idx for idx, val in enumerate(values)}

def build_entity_vocab_from_df(df: pd.DataFrame, col_list: list, truncate_icd = False):
    """Build vocab for codes like ICD or drug from multiple columns (e.g., diag_1, diag_2, diag_3)"""
    unique_vals = set()
    for col in col_list:
        if col in df.columns:
            vals = df[col].dropna().astype(str).unique()
            for v in vals:
                unique_vals.add(v)

    return {val:idx for idx, val in enumerate(sorted(unique_vals))}