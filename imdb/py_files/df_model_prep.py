from py_files.d2v_embed import d2v_embed
import pandas as pd
import math


def df_model_prep(df, filename):
    
    try:
        print("Looking for pre made file...")
        return pd.read_csv(f"{filename}_df_with_features_fully_processed_read_for_model.csv", index_col = 0)
    except:
        print("No file found, creating a new one")
    
    # just encode languages into ints for this column
    df['title_language'] = pd.factorize(df['title_language'])[0]
    
    # dealing with (some) nan values
    for index, row in df.iterrows():
        
        breakpoint()
        # For missing startYear and endYear entries, insert the other, if it exists.
        if math.isnan(row['startYear']):
            if not math.isnan(row['endYear']):
                df.at[index,'startYear']=df.at[index,'endYear']
        if math.isnan(row['endYear']):
            if not math.isnan(row['startYear']):
                df.at[index,'endYear']=df.at[index,'startYear']
                
        # For missing oscar_noms and oscar_wins, insert 0
        if math.isnan(row['oscar_noms']):
            df.at[index,'oscar_noms'] = 0
        if math.isnan(row['oscar_wins']):
            df.at[index,'oscar_wins'] = 0

    df['numVotes'] = df['numVotes'].fillna(df['numVotes'].mean(skipna=True))
    df['runtimeMinutes'] = df['runtimeMinutes'].fillna(df['runtimeMinutes'].mean(skipna=True))
    
    df['title_language'] = pd.factorize(df['title_language'])[0]
    
    prim_title_df = d2v_embed(df['primaryTitle'])
    orig_title_df = d2v_embed(df['originalTitle'])
    prim_title_formatted_df = d2v_embed(df['primaryTitleFormatted'])
    title_formatted_df = d2v_embed(df['titleFormatted'])
    genres_df = d2v_embed(df['genres'])
    
    df.drop(columns = df.select_dtypes(include='object').columns, inplace=True)
    
    df = df.join(prim_title_df)
    df = df.join(orig_title_df)
    df = df.join(prim_title_formatted_df)
    df = df.join(title_formatted_df)
    df = df.join(genres_df)
    
    df.to_csv(f"{filename}_df_with_features_fully_processed_read_for_model.csv")
    
    return df