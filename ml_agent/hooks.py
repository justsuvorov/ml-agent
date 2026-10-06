"""Хуки предобработки данных, на которые ссылаются задания (``[data] preprocess_hook``)."""
import pandas as pd

SEGMENT_COLUMN = 'AGENT_BUSINESS_TYPE'


def casco_business_type(df: pd.DataFrame) -> pd.DataFrame:
    """Тип бизнеса для сегментации — по той же логике, что в main_fit / compare_models.

    main_fit перед сравнением моделей берёт ``BUSINESS_TYPE = BUSINESS_TYPE_NEW.str.upper()``
    (значения 'НОВОЕ ТС', 'ПРОЛОНГАЦИЯ', 'Б/У ТС ПЕРЕХОД', 'Б/У ТС ИНОЕ'). Результат пишем в отдельную
    колонку, чтобы не менять исходные данные, на которых обучаются модели.
    """
    source = df['BUSINESS_TYPE_NEW'] if 'BUSINESS_TYPE_NEW' in df.columns else df['BUSINESS_TYPE']
    df[SEGMENT_COLUMN] = source.astype('string').str.upper().fillna('NA')
    return df
