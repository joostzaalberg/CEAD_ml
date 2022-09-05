import pandas as pd

def import_csv(data_path):
    df = pd.read_csv(data_path)
    print(df)

    return df