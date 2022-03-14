import glob 

def add_box_office(df_):
    '''Add the box office values for a variety of movies.'''
    
    # concatenate all box office files
    all_files = glob.glob("box_office_mojo/*.csv")
    li = []
    for filename in all_files:
        _df_ = pd.read_csv(filename, index_col=None, header=0)
        _df_['year'] = int(filename[-8:-4])
        li.append(_df_)
    df_box_office = pd.concat(li, axis=0, ignore_index=True)
    
    # process the box office movie names in the same way as the given dataset
    df_box_office["Release Group"] = df_box_office["Release Group"].str.lower()\
                                       .str.normalize('NFKD')\
                                       .str.encode('ascii', errors='ignore')\
                                       .str.decode('utf-8')\
                                       .str.replace(" ", "_", regex=True)\
                                       .str.replace("\W", "", regex=True)
    
    # add the box office values ot the dataframe
    df_ = pd.merge(df_, df_box_office, left_on=['primaryTitle', 'startYear'], right_on=['Release Group', 'year'], how='left')
    
    # drop the uneccessary columns
    df_.drop(['Release Group', '%', '%.1', 'year', 'endYear'], axis=1, inplace=True)
    
    return df_