import pandas as pd

data = pd.read_csv('../CMaps/RUL_FD001.txt', sep=" ", header=None)
data.drop(columns=[1], inplace=True)
data.columns = ['RUL']
print(len(data))
for i in range(len(data)):
    print(data.loc[i]['RUL'])
