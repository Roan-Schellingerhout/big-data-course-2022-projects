def create_remake_column(df_):
    ''''Create column that indicates whether the movie is a remake''' 
    all_dups = df_.duplicated(subset='primaryTitle', keep=False)
    df_['hasRemake'] = False
    df_['hasRemake'][dup_bool] = True    
    
    return df_