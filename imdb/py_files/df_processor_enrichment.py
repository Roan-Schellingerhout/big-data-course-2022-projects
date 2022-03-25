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
from py_files.impute_missing_function import impute_missing


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
        # df_original.head()
    else:
        df_original = pd.read_csv(filename, index_col=0)
        # df_original.head()

    # copy the dataframe so we leave the original untouched
    df_preprocessed = df_original.copy(deep=True)

    # start the preprocessing
    df_preprocessed = df_original.replace("\\N", np.nan)
    df_preprocessed["primaryTitleFormatted"] = df_preprocessed["primaryTitle"].str.lower()\
                                                                              .str.normalize('NFKD')\
                                                                              .str.encode('ascii', errors='ignore')\
                                                                              .str.decode('utf-8')\
                                                                              .str.replace("\W", " ", regex=True)
    
    # merge endYear into beginYear when beginYear is not available --> rename Year
    df_preprocessed = merge_start_end_year(df_preprocessed)

    # set the datatypes of the dataframe correctly
    df_preprocessed['Year'] = df_preprocessed['Year'].astype(int)
    df_preprocessed['runtimeMinutes'] = df_preprocessed['runtimeMinutes'].astype(float)
    df_preprocessed["startYear"] = df_preprocessed["startYear"].astype(float)
    df_preprocessed["endYear"] = df_preprocessed["endYear"].astype(float)
    
    ## preprocessing of exogenous data
    
    # Oscar data
    oscars = pd.read_csv("additional_data/oscars.csv")

    oscars["film"] = oscars["film"].str.lower()\
                                   .str.normalize('NFKD')\
                                   .str.encode('ascii', errors='ignore')\
                                   .str.decode('utf-8')\
                                   .str.replace("\W", " ", regex=True)

    # Counting oscar nominations and wins per movie
    oscar_noms = pd.merge(df_preprocessed, oscars, left_on = "primaryTitleFormatted", right_on = "film").groupby("tconst")["winner"].count()
    oscar_wins = pd.merge(df_preprocessed, oscars, left_on = "primaryTitleFormatted", right_on = "film").groupby("tconst")["winner"].sum()
    
    
    #Razzies data
    razzies = pd.read_csv("additional_data/Razzies.csv")

    razzies["moviename"] = razzies["moviename"].str.lower()\
                                   .str.normalize('NFKD')\
                                   .str.encode('ascii', errors='ignore')\
                                   .str.decode('utf-8')\
                                   .str.replace("\W", " ", regex=True)

    # Counting oscar nominations and wins per movie
    razzie_noms = pd.merge(df_preprocessed, razzies, left_on = "primaryTitleFormatted", right_on = "moviename").groupby("tconst")["Wins"].count()
    razzie_wins = pd.merge(df_preprocessed, razzies, left_on = "primaryTitleFormatted", right_on = "moviename").groupby("tconst")["Wins"].sum()
    
    
    # TMDB data
    df_TMDB = pd.read_csv("additional_data/TMDB.csv")[["budget", 
                                                       "genres", 
                                                       "imdb_id", 
                                                       "original_language", 
                                                       "overview", 
                                                       "popularity", 
                                                       "production_companies", 
                                                       "tagline", 
                                                       "Keywords", 
                                                       "revenue"]]
    
    
    df_TMDB["genres"] = df_TMDB["genres"].apply(lambda x: dict_to_list(x))
    df_TMDB["Keywords"] = df_TMDB["Keywords"].apply(lambda x: dict_to_list(x))
    df_TMDB["production_companies"] = df_TMDB["production_companies"].apply(lambda x: dict_to_list(x))
    df_TMDB = df_TMDB.set_index("imdb_id")

    
    # Metacritic data
    df_meta = pd.read_csv(f"additional_data/Metacritic_{filename}.csv").drop("Unnamed: 0", axis=1)\
                                                                       .set_index("movie")
    df_meta["overview"] = df_meta["overview"].apply(lambda x: eval(x))
    df_meta["overview"] = df_meta["overview"].apply(lambda x: x[0] if x else str(x))
    df_meta.rename({'genres': 'genres_meta'}, axis=1, inplace=True)
    
    # Combine overviews
    overviews = pd.merge(df_TMDB["overview"], df_meta["overview"], left_index=True, right_index=True, how="outer")
    overviews["overview"] = overviews["overview_x"].str.cat(overviews["overview_y"], na_rep="")
    overviews = overviews.drop(["overview_x", "overview_y"], axis=1)
    df_TMDB = df_TMDB.drop("overview", axis=1)

    df_TMDB.rename({'genres': 'genres_tmdb'}, axis=1, inplace=True)
    
    
    #Box Office data
    df_box_office_mojo = load_and_aggregate_box_office()

    # process the 'release group' (read movie title) in the same way as the formatted title
    df_box_office_mojo["Release Group"] = df_box_office_mojo["Release Group"].str.lower()\
                                           .str.normalize('NFKD')\
                                           .str.encode('ascii', errors='ignore')\
                                           .str.decode('utf-8')\
                                           .str.replace("\W", " ", regex=True)
    
    df_box_office_mojo.drop(['%', '%.1'], axis=1, inplace=True)

    df_incl_exog = df_preprocessed.copy(deep=True)
    df_incl_exog = df_incl_exog.rename({"tconst" : "id"}, axis = 1).set_index("id")
    
    #Add oscar data
    df_incl_exog["oscar_noms"] = oscar_noms
    df_incl_exog["oscar_wins"] = oscar_wins
    
    df_incl_exog["razzie_noms"] = razzie_noms
    df_incl_exog["razzie_wins"] = razzie_wins
    
    #Add mojo box office
    df_incl_exog = df_incl_exog.reset_index().merge(df_box_office_mojo, left_on=['primaryTitleFormatted', 'Year'], right_on=['Release Group', 'year'], how="left").set_index('id')
    df_incl_exog.drop(['Release Group', 'year'], axis=1, inplace=True)

    df_incl_exog.loc[df_incl_exog['Worldwide'] == '-', 'Worldwide'] = np.nan
    df_incl_exog.loc[df_incl_exog['Domestic'] == '-', 'Domestic'] = np.nan
    df_incl_exog.loc[df_incl_exog['Foreign'] == '-', 'Foreign'] = np.nan

    df_incl_exog.loc[df_incl_exog['Worldwide'].notnull(), 'Worldwide'] = \
        df_incl_exog.loc[df_incl_exog['Worldwide'].notnull(), 'Worldwide']\
                    .apply(lambda x: float(x.replace('$', '').replace(',', '')))
    
    df_incl_exog.loc[df_incl_exog['Domestic'].notnull(), 'Domestic'] = \
        df_incl_exog.loc[df_incl_exog['Domestic'].notnull(), 'Domestic']\
                    .apply(lambda x: float(x.replace('$', '').replace(',', '')))

    df_incl_exog.loc[df_incl_exog['Foreign'].notnull(), 'Foreign'] = \
        df_incl_exog.loc[df_incl_exog['Foreign'].notnull(), 'Foreign']\
                    .apply(lambda x: float(x.replace('$', '').replace(',', '')))
    
    
    #Add remake column
    df_incl_exog = create_remake_column(df_incl_exog)
    
    #Add title language
    df_added_lang = pd.read_csv('additional_data/df_added_lang.csv', index_col=0)
    df_added_lang = df_added_lang.rename({"tconst" : "id"}, axis = 1).set_index("id")
    df_incl_exog = df_incl_exog.join(df_added_lang['title_language'], how='left')
    
    #Add whether title is English or not
    df_incl_exog = add_english_title_or_not(df_incl_exog)
    
    #Add movie genres
    df_incl_exog = add_movie_genre(df_incl_exog)
    
    #Add TMDB & Metacritic overviews
    df_incl_exog = pd.merge(df_incl_exog, df_TMDB, how="left", left_index=True, right_index=True)
    df_incl_exog = pd.merge(df_incl_exog, df_meta, how="left", left_index=True, right_index=True)
    df_incl_exog = pd.merge(df_incl_exog, overviews, how="left", left_index=True, right_index=True)
    
    #Remove the last stuff we don't want
    df_incl_exog['genres_tmdb']= df_incl_exog['genres_tmdb'].apply(lambda d: d if isinstance(d, list) else [])
    df_incl_exog['genres_meta']= df_incl_exog['genres_meta'].apply(lambda d: d if isinstance(d, list) else [])
    df_incl_exog['genres_movielens']= df_incl_exog['genres_movielens'].apply(lambda d: d if isinstance(d, list) else [])
    
    #Merge genre columns together
    df_incl_exog['genres_combined'] = df_incl_exog['genres_meta'] + \
                                      df_incl_exog['genres_tmdb'] + \
                                      df_incl_exog['genres_movielens']
    
    df_incl_exog['genres_combined'] = df_incl_exog['genres_combined'].apply(lambda x: list(set(x)))
    df_incl_exog.drop(['genres_meta', 'genres_tmdb', 'genres_movielens'], axis=1, inplace=True, errors='ignore')
    
    s = df_incl_exog['genres_combined'].explode()
    df_incl_exog = df_incl_exog.join(pd.crosstab(s.index, s), how='left', lsuffix='x')

    #Coalesce revenue and box office values
    df_incl_exog['revenue'] = df_incl_exog[['revenue', 'Worldwide']].max(axis=1)
    df_incl_exog[df_incl_exog['revenue'] < 100000] = np.nan
    
    # drop some unnecessary columns
    df_incl_exog.drop(['startYear', 
                       'endYear', 
                       'primaryTitleFormatted', 
                       'Domestic', 
                       'Foreignx', 
                       'Keywords', 
                       'genres_combined', 
                       'overview_x', 
                       'overview_y', 
                       'tagline', 
                       'Worldwide'], axis=1, inplace=True, errors='ignore')
    
    
    #Set columns with correct datatype
    df_incl_exog['isEN'] = df_incl_exog['isEN'].astype('float')
    df_incl_exog['title_language'] = df_incl_exog['title_language'].astype('string')
    
    #Impute missing
    imputed_df = impute_missing(df_incl_exog, strategy = 'MICE')
    
    df_incl_exog = df_incl_exog.drop(columns = imputed_df.columns.values)\
                               .join(imputed_df)
    
    df_incl_exog.to_csv(f"{filename}_df_with_features.csv")
    
    return df_incl_exog