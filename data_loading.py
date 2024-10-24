import pandas as pd
from sqlalchemy import create_engine
import json 

with open('config.json') as config_file: CONFIG = json.load(config_file)
conn_string = f"postgresql://{CONFIG['user']}:{CONFIG['password']}@{CONFIG['host']}:{CONFIG['port']}/{CONFIG['dbname']}"
engine = create_engine(conn_string)

def read_table_to_df(table_name):
    try:
        df = pd.read_sql(f'SELECT * FROM "{table_name}";', engine)
        print(f"Data from {table_name} loaded successfully!")
        return df
    except Exception as e:
        print(f"Error while reading from table {table_name}: {e}")
        return None
    
def write_df_to_table(df, table_name):
    try:
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        print(f"DataFrame written to {table_name} successfully!")
    except Exception as e:
        print(f"Error while writing DataFrame to table {table_name}: {e}")