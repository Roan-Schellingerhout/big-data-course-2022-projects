import pandas as pd
import glob

def load_original_data():
    """Load original data into dataframe."""
    
    all_files = glob.glob("train*.csv")

    print(f"Found files: {', '.join(all_files)}")

    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    df = pd.concat(li, axis=0, ignore_index=True).drop("Unnamed: 0", axis=1)
    
    return df