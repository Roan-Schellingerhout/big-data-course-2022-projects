import glob
import pandas as pd

def load_and_aggregate_box_office():
    """This function loads all the box office values scraped from the box office mojo.""" 
    
    # concatenate all box office files
    all_files = glob.glob("box_office_mojo/*.csv")

    print(f"Found files: {', '.join(all_files)}")

    li = []

    for filename in all_files:
        df_ = pd.read_csv(filename, index_col=None, header=0)
        df_['year'] = int(filename[-8:-4])
        li.append(df_)

    df_box_office = pd.concat(li, axis=0, ignore_index=True)
    
    return df_box_office