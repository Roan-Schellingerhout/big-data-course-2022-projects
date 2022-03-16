import tqdm
from langdetect import detect

def add_language_of_original_title(df_):
    '''Create column indicating the language of the available title.''' 
    
    # format the primary title so that it is plain but usable for language detection
    df_["temp_prim_title"] = df_["primaryTitle"].str.lower()\
                                       .str.normalize('NFKD')\
                                       .str.encode('ascii', errors='ignore')\
                                       .str.decode('utf-8')
    
    # make sure the original title is a string and not an object
    df_['temp_orig_title'] = df_['originalTitle'].astype(str)
    df_['title_language'] = 'unknown'

    # try to get the language for the original title for every row, otherwise get the language of the primary title
    for idx, row in tqdm.tqdm(df_.iterrows()):
        if df_.loc[idx, 'temp_orig_title'] != 'nan':
            try:
                df_.loc[idx, 'title_language'] = detect(df_.loc[idx]['temp_orig_title'])
            except:
                print(f"Could not detect language of original title: {df_.loc[idx]['temp_orig_title']}")
                df_.loc[idx, 'title_language'] = 'unknown'
        else:
            try:
                df_.loc[idx, 'title_language'] = detect(df_.loc[idx]['temp_prim_title'])
            except:
                print(f"Could not detect language of primary title: {df_.loc[idx]['temp_prim_title']}")
                df_.loc[idx, 'title_language'] = 'unknown'
                
    df_.drop(['temp_prim_title', 'temp_orig_title'], axis=1, inplace=True)
    
    return df_
                
        
    