def merge_start_end_year(df_):
    df_['Year'] = df_['startYear'].fillna(df_['endYear'])
    return df_