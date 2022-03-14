def add_english_title_or_not(df_):
    '''Create a column indicating whether the title of the movie is english or not, simplifying the english column.'''
    
    assert 'title_language' in df_.columns, 'title of language has not yet been distilled'
    
    df_['isEN'] = False
    df_.loc[df_['title_language'] == 'en', 'isEN'] = True
    
    return df_