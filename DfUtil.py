from pandas import DataFrame


def scale_to_start_value(df: DataFrame):
    for column in df.columns:
        alpha = 1.0 / df[column].iloc[0]
        df[column] = alpha * df[column]
