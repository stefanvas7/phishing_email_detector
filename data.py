import pandas as pd
import numpy as np

df = pd.read_csv("CEAS_08.csv", usecols = ['body', 'label'])

# train on body and label columns only
df = df.dropna(subset=["body", "label"])

df.info()

# split
train, val, test = np.split(df.sample(frac=1), [int(0.8*len(df)), int(0.9*len(df))])


