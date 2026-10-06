"""Хуки предобработки данных, на которые ссылаются задания (``[data] preprocess_hook``)."""
import pandas as pd


def casco_business_type(df: pd.DataFrame) -> pd.DataFrame:
    """Как в casco/main_fit.py: BUSINESS_TYPE берётся из BUSINESS_TYPE_NEW в верхнем регистре."""
    if 'BUSINESS_TYPE_NEW' in df.columns:
        df['BUSINESS_TYPE'] = df['BUSINESS_TYPE_NEW'].str.upper()
    df['BUSINESS_TYPE'] = df['BUSINESS_TYPE'].fillna('NA')
    return df

