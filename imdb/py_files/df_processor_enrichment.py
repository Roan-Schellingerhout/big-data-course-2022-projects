import json
import numpy as np
import ast
import pandas as pd
from itertools import groupby

from py_files.writer_director_to_one_hot import writer_director_to_one_hot
from py_files.add_merge_begin_end_year import merge_start_end_year
from py_files.load_box_office_data import load_and_aggregate_box_office
from py_files.add_remake_feature import create_remake_column
from py_files.add_langoriginaltitle_feature import add_language_of_original_title
from py_files.add_ENvsNonEN_feature import add_english_title_or_not
from py_files.add_movie_genre_feature import add_movie_genre

from py_files.d2v_embed import d2v_embed
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import math

from py_files.load_original_data import load_original_data

def dict_to_list(dictionary):
    try:
        d = ast.literal_eval(dictionary)
    except ValueError:
        return []
    
    return [i["name"] for i in d]

def df_processor_enrichment(filename):
    
    try:
        print("Looking for pre made file...")
        return pd.read_csv(f"{filename}_df_with_features.csv", index_col = 0)
    except:
        print("File not found, creating a new one..")
    
    if filename == 'train':
        df_original = load_original_data()
        df_original.head()
    else:
        df_original = pd.read_csv(filename, index_col=0)
        # df_original.head()

    # start the preprocessing
    df_preprocessed = df_original.replace("\\N", np.nan)
    df_preprocessed["primaryTitleFormatted"] = df_preprocessed["primaryTitle"].str.lower()\
                                                                              .str.normalize('NFKD')\
                                                                              .str.encode('ascii', errors='ignore')\
                                                                              .str.decode('utf-8')\
                                                                              .str.replace(" ", "_", regex=True)\
                                                                              .str.replace("\W", "", regex=True)

    # merge endYear into beginYear when beginYear is not available --> rename Year
    df_preprocessed = merge_start_end_year(df_preprocessed)

    # set the datatypes of the dataframe correctly
    df_preprocessed['Year'] = df_preprocessed['Year'].astype(int)
    df_preprocessed['runtimeMinutes'] = df_preprocessed['runtimeMinutes'].astype(float)

    # df_preprocessed.info()


    oscars = pd.read_csv("additional_data/oscars.csv")

    oscars["film"] = oscars["film"].str.lower()\
                                   .str.normalize('NFKD')\
                                   .str.encode('ascii', errors='ignore')\
                                   .str.decode('utf-8')\
                                   .str.replace(" ", "_", regex=True)\
                                   .str.replace("\W", "", regex=True)

    # Counting oscar nominations and wins per movie
    oscar_noms = pd.merge(df_preprocessed, oscars, left_on = "primaryTitleFormatted", right_on = "film").groupby("tconst")["winner"].count()
    oscar_wins = pd.merge(df_preprocessed, oscars, left_on = "primaryTitleFormatted", right_on = "film").groupby("tconst")["winner"].sum()


    # Find writers and directors per movie and combine the two
    writers = writer_director_to_one_hot("writers")
    directors = writer_director_to_one_hot("directors")
    written_and_directed = writers.add(directors, fill_value=0).fillna(0).astype(int).loc[df_preprocessed["tconst"]]
    
    df_TMDB = pd.read_csv("additional_data/TMDB.csv")[["budget", "genres", "imdb_id", 
                                                       "original_language", "overview", 
                                                       "popularity", "production_companies", 
                                                       "tagline", "Keywords", "revenue"]]

    ### TODO: genres_TMDB and the other genres column should be merged
    df_TMDB["genres_TMDB"] = df_TMDB["genres"].apply(lambda x: dict_to_list(x))
    df_TMDB["Keywords"] = df_TMDB["Keywords"].apply(lambda x: dict_to_list(x))
    df_TMDB["production_companies"] = df_TMDB["production_companies"].apply(lambda x: dict_to_list(x))
    df_TMDB = df_TMDB.set_index("imdb_id")
    
    df_meta = pd.read_csv("additional_data/Metacritic.csv").drop("Unnamed: 0", axis=1).set_index("movie")
    df_meta["overview"] = df_meta["overview"].apply(lambda x: eval(x))
    df_meta["overview"] = df_meta["overview"].apply(lambda x: x[0] if x else str(x))
    
    # Combine for faster merge
    overviews = pd.merge(df_TMDB["overview"], df_meta["overview"], left_index=True, right_index=True, how="outer")
    overviews["overview"] = overviews["overview_x"].str.cat(overviews["overview_y"], na_rep="")
    overviews = overviews.drop(["overview_x", "overview_y"], axis=1)
    df_TMDB = df_TMDB.drop(["overview", "genres"], axis=1)    

    df_box_office_mojo = load_and_aggregate_box_office()

    # process the 'release group' (read movie title) in the same way as the formatted title
    df_box_office_mojo["Release Group"] = df_box_office_mojo["Release Group"].str.lower()\
                                           .str.normalize('NFKD')\
                                           .str.encode('ascii', errors='ignore')\
                                           .str.decode('utf-8')\
                                           .str.replace(" ", "_", regex=True)\
                                           .str.replace("\W", "", regex=True)
    df_box_office_mojo.drop(['%', '%.1'], axis=1, inplace=True)


    df_incl_exog = df_preprocessed.copy(deep=True)
    df_incl_exog = df_incl_exog.rename({"tconst" : "id"}, axis = 1).set_index("id")
    # df_incl_exog.info()


    df_incl_exog["oscar_noms"] = oscar_noms
    df_incl_exog["oscar_wins"] = oscar_wins
    
    df_incl_exog = pd.merge(df_incl_exog, df_TMDB, how="left", left_index=True, right_index=True)
    df_incl_exog = pd.merge(df_incl_exog, overviews, how="left", left_index=True, right_index=True)

    df_incl_exog = df_incl_exog.reset_index().merge(df_box_office_mojo, left_on=['primaryTitleFormatted', 'Year'], right_on=['Release Group', 'year'], how="left").set_index('id')
    df_incl_exog.drop(['Release Group', 'year'], axis=1, inplace=True)

    df_incl_exog.loc[df_incl_exog['Worldwide'] == '-', 'Worldwide'] = np.nan
    df_incl_exog.loc[df_incl_exog['Domestic'] == '-', 'Domestic'] = np.nan
    df_incl_exog.loc[df_incl_exog['Foreign'] == '-', 'Foreign'] = np.nan
    df_incl_exog.loc[df_incl_exog['Worldwide'].notnull(), 'Worldwide'] = df_incl_exog.loc[df_incl_exog['Worldwide'].notnull(), 'Worldwide'].apply(lambda x: float(x.replace('$', '').replace(',', '')))
    df_incl_exog.loc[df_incl_exog['Domestic'].notnull(), 'Domestic'] = df_incl_exog.loc[df_incl_exog['Domestic'].notnull(), 'Domestic'].apply(lambda x: float(x.replace('$', '').replace(',', '')))
    df_incl_exog.loc[df_incl_exog['Foreign'].notnull(), 'Foreign'] = df_incl_exog.loc[df_incl_exog['Foreign'].notnull(), 'Foreign'].apply(lambda x: float(x.replace('$', '').replace(',', '')))


    df_incl_exog = create_remake_column(df_incl_exog)

    # # add the language of the original title, currently commented for training data usage and not wait 15 min every time
    # df_incl_exog = add_language_of_original_title(df_incl_exog)

    df_added_lang = pd.read_csv('additional_data/df_added_lang.csv', index_col=0)
    df_added_lang = df_added_lang.rename({"tconst" : "id"}, axis = 1).set_index("id")
    df_incl_exog = df_incl_exog.join(df_added_lang['title_language'], how='left')

    df_incl_exog = add_english_title_or_not(df_incl_exog)
    df_incl_exog = add_movie_genre(df_incl_exog)
    df_incl_exog = pd.concat([df_incl_exog.T, written_and_directed.T]).T
    df_incl_exog.to_csv(f"{filename}_df_with_features.csv")
    
    return df_incl_exog