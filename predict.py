import pickle
import pandas as pd

def predict(df_orig, X, path, columns):
    with open(path, 'rb') as file:
        model = pickle.load(file)

    probabilities = model.predict_proba(X)
    
    prob_df = pd.DataFrame(probabilities, columns=columns)

    for col in columns:
        df_orig[col]=(prob_df[col] * 10).round() / 10 

    return df_orig

