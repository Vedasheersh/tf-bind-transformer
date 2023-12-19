import pandas as pd

datafile = '../data/natmet/final_data/kcat_merged_wdups.csv'
df = pd.read_csv(datafile)

cols = ['sequence','reaction_smiles','log10_value']
df = df[cols]

df.to_csv('../data/kcat_merged_wdups_simpleformat.csv')
