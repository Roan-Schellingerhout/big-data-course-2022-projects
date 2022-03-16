import json

import pandas as pd

from itertools import groupby
from sklearn.preprocessing import MultiLabelBinarizer

def open_writer_director_file(data):
    if data == "writers":
        with open("writing.json") as f:
            writers = f.read()
        
        return json.loads(writers)
    elif data == "directors":
        with open("directing.json") as f:
            directors = f.read()
        
        return json.loads(directors)    
    else:
        return NotImplemented    

def writer_director_to_one_hot(kind = "writers"):

    if kind in ["writers", "directors"]:
        json_file = open_writer_director_file(kind)
    else:
        return NotImplemented
  
    if kind == "writers":
         # Group writers/directors by movie
        groups = groupby([(i["movie"], i["writer"]) for i in json_file], key = lambda x : x[0])

        # Turn groupby object into a json-like dict
        grouped = {writer: [i[1] for i in movies] for writer, movies in groups}

        # Convert to Series
        df = pd.DataFrame.from_records(list(grouped.items())).set_index(0).squeeze()
    else:
        df = pd.DataFrame(json_file).groupby("movie")["director"].apply(lambda x: x.values)
    
    # Create one-hot encoded DataFrame
    mlb = MultiLabelBinarizer()
    res = pd.DataFrame(mlb.fit_transform(df),
                       columns=mlb.classes_,
                       index=df.index)
    
    return res.drop("\\N", axis = 1)