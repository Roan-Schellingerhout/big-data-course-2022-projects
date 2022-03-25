from py_files.d2v_embed import d2v_embed
import numpy as np
import pandas as pd
import re
import math
from sklearn.preprocessing import MultiLabelBinarizer


def df_model_prep(df, filename):
    
    try:
        print("Looking for pre made file...")
        return pd.read_csv(f"{filename}_df_with_features_fully_processed_ready_for_model.csv", index_col = 0)
    except:
        print("No file found, creating a new one")
    
    df['oscar_noms'] = df['oscar_noms'].fillna(0.0)
    df['oscar_wins'] = df['oscar_noms'].fillna(0.0)
    df['razzie_noms'] = df['razzie_noms'].fillna(0.0)
    df['razzie_wins'] = df['razzie_wins'].fillna(0.0)
    
    
    # just encode languages into ints for this column
    df['title_language'] = pd.factorize(df['title_language'])[0]
    df['original_language'] = pd.factorize(df['original_language'])[0]
    
    lang_proc = df["language"]\
                .replace(np.nan, " ")\
                .apply(lambda x: re.sub("[^a-zA-Z]", " ", str(x)))\
                .str.split()
    
    mlb = MultiLabelBinarizer()
    lang_proc = pd.DataFrame(mlb.fit_transform(lang_proc),
                             columns=mlb.classes_,
                             index=df.index)
    
    df = df.drop(columns = ['language'])
    df = df.join(lang_proc)

    prod_proc = df["production_companies"]\
                .replace(np.nan, " ")\
                .apply(lambda x: re.sub("[^a-zA-Z]", " ", str(x)))\
                .str.split()
    
    mlb_prod = MultiLabelBinarizer()
    prod_proc = pd.DataFrame(mlb_prod.fit_transform(prod_proc),
                             columns=mlb_prod.classes_,
                             index=df.index)
    
    df = df.drop(columns = ["production_companies"])
    df = df.join(prod_proc, rsuffix='_prod')
    
#     df['numVotes'] = df['numVotes'].fillna(df['numVotes'].mean(skipna=True))
#     df['runtimeMinutes'] = df['runtimeMinutes'].fillna(df['runtimeMinutes'].mean(skipna=True))
    
    prim_title_df = d2v_embed(df['primaryTitle'], filename)
    orig_title_df = d2v_embed(df['originalTitle'], filename)
    title_formatted_df = d2v_embed(df['titleFormatted'], filename)
#     overview_df = d2v_embed(df['overview'])
    
    df.drop(columns = ['primaryTitle', 'originalTitle', 'titleFormatted'], inplace=True)
    
    df = df.join(prim_title_df)
    df = df.join(orig_title_df)
    df = df.join(title_formatted_df)
    
    df.to_csv(f"{filename}_df_with_features_fully_processed_ready_for_model.csv")
    
    return df