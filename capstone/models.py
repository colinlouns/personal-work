import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from math import sqrt
import random
import pickle

def squaring(x):
    return x*x

ss = StandardScaler()
model = LinearRegression()

df = pd.read_csv('player_stats.csv')

df
data = df
data.rename(columns={'name': 'player_id'}, inplace=True)

players = data.groupby('player_id')['Year'].agg({'min': min, 'max': max, 'count': 'size'})
players = players[((players['min'] > 2000) | (players['max'] > 2015)) & (players['count'] > 6)]
players = players[players['max'] - players['min'] < 20]
players = players.index.values

#-------------------------------------------------------------------------------

ba_model = LinearRegression()

df1 = data[data['player_id'].isin(players)]

numeric_columns = ['Year', 'Age', 'BA', 'PA']

for column in list(numeric_columns):
    df1[column] = pd.to_numeric(df1[column], errors = 'coerce')

df1 = df1[df1['PA'] > 80]
df1 = df1[['player_id', 'Year', 'Age', 'BA']]
df1['Age^2'] = squaring(df1['Age'])
df1['BA_1'] = df1.groupby(['player_id'])['BA'].shift(1)
df1['BA_2'] = df1.groupby(['player_id'])['BA'].shift(2)
df1['BA_3'] = df1.groupby(['player_id'])['BA'].shift(3)
df1['BA_4'] = df1.groupby(['player_id'])['BA'].shift(4)
df1 = df1.dropna()

df1.to_csv('./csv_extras/batting_average.csv')


len(player_id_list)
player_id_list = list(np.unique([df1['player_id']]))
random.shuffle(player_id_list)
player_id_train = player_id_list[:300]
player_id_test = player_id_list[300:]

train = df1[df1['player_id'].isin(player_id_train)]
test = df1[df1['player_id'].isin(player_id_test)]

X_train = train[['Age', 'Age^2', 'BA_1', 'BA_2', 'BA_3', 'BA_4']]
X_test = test[['Age', 'Age^2', 'BA_1', 'BA_2', 'BA_3', 'BA_4']]
y_train = train['BA']
y_test = test['BA']

ss.fit(X_train)
X_train_scaled = ss.transform(X_train)
ss.fit(X_test)
X_test_scaled = ss.transform(X_test)

ba_model.fit(X_train_scaled, y_train)
train_score = ba_model.score(X_train_scaled, y_train)
test_score = ba_model.score(X_test_scaled, y_test)
print(train_score, test_score)

predictions = ba_model.predict(X_test_scaled)

rmse = sqrt(mean_squared_error(y_test, predictions))
print(rmse)

pickle.dump(ba_model, open('ba_model.pkl', 'wb'))
#-------------------------------------------------------------------------------

hr_model = LinearRegression()

df2 = data[data['player_id'].isin(players)]

numeric_columns_2 = ['Year', 'Age', 'HR', 'PA']

for column in list(numeric_columns_2):
    df2[column] = pd.to_numeric(df2[column], errors = 'coerce')

df2 = df2[df2['PA'] > 80]

df2 = df2[['player_id', 'Year', 'Age', 'HR']]
df2['Age^2'] = squaring(df2['Age'])
df2['HR_1'] = df2.groupby(['player_id'])['HR'].shift(1)
df2['HR_2'] = df2.groupby(['player_id'])['HR'].shift(2)
df2['HR_3'] = df2.groupby(['player_id'])['HR'].shift(3)
df2['HR_4'] = df2.groupby(['player_id'])['HR'].shift(4)

df2 = df2.dropna()

df2.to_csv('./csv_extras/home_runs.csv')

player_id_list = list(np.unique([df2['player_id']]))
random.shuffle(player_id_list)
len(player_id_list)
player_id_train = player_id_list[:310]
player_id_test = player_id_list[310:]

train = df2[df2['player_id'].isin(player_id_train)]
test = df2[df2['player_id'].isin(player_id_test)]

X_train = train[['Age', 'Age^2', 'HR_1', 'HR_2', 'HR_3', 'HR_4']]
X_test = test[['Age', 'Age^2', 'HR_1', 'HR_2', 'HR_3', 'HR_4']]
y_train = train['HR']
y_test = test['HR']

ss.fit(X_train)
X_train_scaled = ss.transform(X_train)
ss.fit(X_test)
X_test_scaled = ss.transform(X_test)

hr_model.fit(X_train_scaled, y_train)
hr_model.score(X_train_scaled, y_train)
hr_model.score(X_test_scaled, y_test)

predictions = hr_model.predict(X_test_scaled)

rmse = sqrt(mean_squared_error(y_test, predictions))
rmse

compare = pd.DataFrame([])

compare['predictions'] = predictions
compare['trues'] = y_test.values

compare['true_error'] = compare['predictions'] - compare['trues']

compare.sort_values(by = 'true_error')

np.mean(np.abs(compare['true_error']))

kf = KFold(n_splits=5, random_state=42, shuffle=True)

cross_val_score(model, X_train_scaled, y_train, cv=kf)

pickle.dump(hr_model, open('hr_model.pkl', 'wb'))

#-------------------------------------------------------------------------------

hits_model = LinearRegression()

df3 = data[data['player_id'].isin(players)]

numeric_columns_3 = ['Year', 'Age', 'H', 'PA']

for column in list(numeric_columns_3):
    df3[column] = pd.to_numeric(df3[column], errors = 'coerce')

df3 = df3[df3['PA'] > 80]

df3 = df3[['player_id', 'Year', 'Age', 'H']]
df3['Age^2'] = squaring(df3['Age'])
df3['H_1'] = df3.groupby(['player_id'])['H'].shift(1)
df3['H_2'] = df3.groupby(['player_id'])['H'].shift(2)
df3['H_3'] = df3.groupby(['player_id'])['H'].shift(3)
df3['H_4'] = df3.groupby(['player_id'])['H'].shift(4)

df3 = df3.dropna()

df3.to_csv('./csv_extras/hits.csv')

player_id_list = list(np.unique([df3['player_id']]))
random.shuffle(player_id_list)
len(player_id_list)
player_id_train = player_id_list[:310]
player_id_test = player_id_list[310:]

train = df3[df3['player_id'].isin(player_id_train)]
test = df3[df3['player_id'].isin(player_id_test)]

X_train = train[['Age', 'Age^2', 'H_1', 'H_2', 'H_3', 'H_4']]
X_test = test[['Age', 'Age^2', 'H_1', 'H_2', 'H_3', 'H_4']]
y_train = train['H']
y_test = test['H']

ss.fit(X_train)
X_train_scaled = ss.transform(X_train)
ss.fit(X_test)
X_test_scaled = ss.transform(X_test)

hits_model.fit(X_train_scaled, y_train)
hits_model.score(X_train_scaled, y_train)
hits_model.score(X_test_scaled, y_test)

predictions = hits_model.predict(X_test_scaled)

rmse = sqrt(mean_squared_error(y_test, predictions))
rmse

pickle.dump(hits_model, open('hits_model.pkl', 'wb'))

#-------------------------------------------------------------------------------

sb_model = LinearRegression()

df4 = data[data['player_id'].isin(players)]

numeric_columns_4 = ['Year', 'Age', 'SB', 'PA']

for column in list(numeric_columns_4):
    df4[column] = pd.to_numeric(df4[column], errors = 'coerce')

df4 = df4[df4['PA'] > 80]

df4 = df4[['player_id', 'Year', 'Age', 'SB']]
df4['Age^2'] = squaring(df4['Age'])
df4['SB_1'] = df4.groupby(['player_id'])['SB'].shift(1)
df4['SB_2'] = df4.groupby(['player_id'])['SB'].shift(2)
df4['SB_3'] = df4.groupby(['player_id'])['SB'].shift(3)
df4['SB_4'] = df4.groupby(['player_id'])['SB'].shift(4)

df4 = df4.dropna()

df4.to_csv('./csv_extras/stolen_bases.csv')

player_id_list = list(np.unique([df4['player_id']]))
random.shuffle(player_id_list)
len(player_id_list)
player_id_train = player_id_list[:310]
player_id_test = player_id_list[310:]

train = df4[df4['player_id'].isin(player_id_train)]
test = df4[df4['player_id'].isin(player_id_test)]

X_train = train[['Age', 'Age^2', 'SB_1', 'SB_2', 'SB_3', 'SB_4']]
X_test = test[['Age', 'Age^2', 'SB_1', 'SB_2', 'SB_3', 'SB_4']]
y_train = train['SB']
y_test = test['SB']

ss.fit(X_train)
X_train_scaled = ss.transform(X_train)
ss.fit(X_test)
X_test_scaled = ss.transform(X_test)

sb_model.fit(X_train_scaled, y_train)
sb_model.score(X_train_scaled, y_train)
sb_model.score(X_test_scaled, y_test)

predictions = sb_model.predict(X_test_scaled)

rmse = sqrt(mean_squared_error(y_test, predictions))
rmse

pickle.dump(sb_model, open('sb_model.pkl', 'wb'))

#-------------------------------------------------------------------------------

rbi_model = LinearRegression()

df5 = data[data['player_id'].isin(players)]

numeric_columns_5 = ['Year', 'Age', 'RBI', 'PA']

for column in list(numeric_columns_5):
    df5[column] = pd.to_numeric(df5[column], errors = 'coerce')

df5 = df5[df5['PA'] > 80]

df5 = df5[['player_id', 'Year', 'Age', 'RBI']]
df5['Age^2'] = squaring(df5['Age'])
df5['RBI_1'] = df5.groupby(['player_id'])['RBI'].shift(1)
df5['RBI_2'] = df5.groupby(['player_id'])['RBI'].shift(2)
df5['RBI_3'] = df5.groupby(['player_id'])['RBI'].shift(3)
df5['RBI_4'] = df5.groupby(['player_id'])['RBI'].shift(4)

df5 = df5.dropna()

df5.to_csv('./csv_extras/runs_batted_in.csv')

player_id_list = list(np.unique([df5['player_id']]))
random.shuffle(player_id_list)
len(player_id_list)
player_id_train = player_id_list[:310]
player_id_test = player_id_list[310:]

train = df5[df5['player_id'].isin(player_id_train)]
test = df5[df5['player_id'].isin(player_id_test)]

X_train = train[['Age', 'Age^2', 'RBI_1', 'RBI_2', 'RBI_3', 'RBI_4']]
X_test = test[['Age', 'Age^2', 'RBI_1', 'RBI_2', 'RBI_3', 'RBI_4']]
y_train = train['RBI']
y_test = test['RBI']

ss.fit(X_train)
X_train_scaled = ss.transform(X_train)
ss.fit(X_test)
X_test_scaled = ss.transform(X_test)

rbi_model.fit(X_train_scaled, y_train)
rbi_model.score(X_train_scaled, y_train)
rbi_model.score(X_test_scaled, y_test)

predictions = rbi_model.predict(X_test_scaled)

rmse = sqrt(mean_squared_error(y_test, predictions))
rmse

pickle.dump(rbi_model, open('rbi_model.pkl', 'wb'))

#-------------------------------------------------------------------------------

runs_model = LinearRegression()

df6 = data[data['player_id'].isin(players)]

numeric_columns_6 = ['Year', 'Age', 'R', 'PA']

for column in list(numeric_columns_6):
    df6[column] = pd.to_numeric(df6[column], errors = 'coerce')

df6 = df6[df6['PA'] > 80]

df6 = df6[['player_id', 'Year', 'Age', 'R']]
df6['Age^2'] = squaring(df6['Age'])
df6['R_1'] = df6.groupby(['player_id'])['R'].shift(1)
df6['R_2'] = df6.groupby(['player_id'])['R'].shift(2)
df6['R_3'] = df6.groupby(['player_id'])['R'].shift(3)
df6['R_4'] = df6.groupby(['player_id'])['R'].shift(4)

df6 = df6.dropna()

df6.to_csv('./csv_extras/runs_batted_in.csv')

player_id_list = list(np.unique([df6['player_id']]))
random.shuffle(player_id_list)
len(player_id_list)
player_id_train = player_id_list[:310]
player_id_test = player_id_list[310:]

train = df6[df6['player_id'].isin(player_id_train)]
test = df6[df6['player_id'].isin(player_id_test)]

X_train = train[['Age', 'Age^2', 'R_1', 'R_2', 'R_3', 'R_4']]
X_test = test[['Age', 'Age^2', 'R_1', 'R_2', 'R_3', 'R_4']]
y_train = train['R']
y_test = test['R']

ss.fit(X_train)
X_train_scaled = ss.transform(X_train)
ss.fit(X_test)
X_test_scaled = ss.transform(X_test)

runs_model.fit(X_train_scaled, y_train)
runs_model.score(X_train_scaled, y_train)
runs_model.score(X_test_scaled, y_test)

predictions = runs_model.predict(X_test_scaled)

rmse = sqrt(mean_squared_error(y_test, predictions))
rmse

pickle.dump(runs_model, open('runs_model.pkl', 'wb'))
