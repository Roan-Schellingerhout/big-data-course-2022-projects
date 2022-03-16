import re
import pandas as pd


def retrieve_year(string):
    try:
        return int(re.search('\((.*?)\)', string).group()[1:-1])
    except:
        return pd.NA

def remove_year(string):
    try:
        return re.sub('\((.*?)\)', '', string)[:-1]
    except:
        return str

def add_movie_genre(df_):
    ''''Create onehot encoded features of genres'''
    
    # load movies with genre data
    movie_genres = pd.read_csv(r'additional_data/movie_genres.csv', index_col=0)

    # remove movies in data set that don't have genres
    movie_genres = movie_genres[movie_genres['genres'] != '(no genres listed)']
    
    # get date for each movie from title column
    movie_genres['year'] = movie_genres['title'].apply(lambda x: retrieve_year(x))
    movie_genres = movie_genres.dropna(subset='year')

    # remove year from title column and set title data type correctly
    movie_genres['year'] = movie_genres['year'].astype(int)
    movie_genres['title'] = movie_genres['title'].apply(lambda x: remove_year(x)).astype('string')
    movie_genres['genres'] = movie_genres['genres'].apply(lambda x: x.split('|'))
    
    # format title in same way as original dataset
    movie_genres["titleFormatted"] = movie_genres["title"].str.lower()\
                                       .str.normalize('NFKD')\
                                       .str.encode('ascii', errors='ignore')\
                                       .str.decode('utf-8')\
                                       .str.replace(" ", "_", regex=True)\
                                       .str.replace("\W", "", regex=True)
    
    movie_genres.drop_duplicates(subset=['titleFormatted', 'year'], inplace=True)
    
    df_ = df_.reset_index().merge(movie_genres[['year', 'titleFormatted', 'genres']], left_on=['primaryTitleFormatted', 'Year'], right_on=['titleFormatted', 'year'], how='left').set_index('id')
    s = df_['genres'].explode()
    df_ = df_.join(pd.crosstab(s.index, s), how='left')
    
    return df_
    
    