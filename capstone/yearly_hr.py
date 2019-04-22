twenty_sixteen_hr = final_data[final_data['Year'] == 2016]
twenty_seventeen_hr = final_data[final_data['Year'] == 2017]
twenty_eighteen_hr = final_data[final_data['Year'] == 2018]


twenty_sixteen_hr_shortened = twenty_sixteen_hr[['player_id', 'HR']]
twenty_seventeen_hr_shortened = twenty_seventeen_hr[['player_id', 'HR']]
twenty_eighteen_hr_shortened = twenty_eighteen_hr[['player_id', 'HR']]


last_two_years = twenty_eighteen_hr_shortened.merge(twenty_seventeen_hr_shortened, how = 'outer', on = 'player_id', suffixes=('_2018', '_2017'))
last_three_years = last_two_years.merge(twenty_sixteen_hr_shortened, how = 'outer', on = 'player_id')

last_two_years

last_three_years.columns = ['player_id', 'HR_2018', 'HR_2017', 'HR_2016']
last_three_years['HR_1'] = last_three_years['HR_2018'] - last_three_years['HR_2017']
last_three_years['HR_2'] = last_three_years['HR_2018'] - last_three_years['HR_2016']

last_three_years

mini_data = last_three_years.dropna()

model = LinearRegression()

X = mini_data[['HR_2017', 'HR_2016']]
y = mini_data['HR_2018']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)

model.fit(X_train, y_train)
model.score(X_train, y_train)
model.score(X_test, y_test)


# old ID creator
# i = 0
# player_id = 0
# data['player_id'] = None
# for i in range(len(data)-1):
#     this_year = int(data.iloc[i]['Year'])
#     next_year = int(data.iloc[i+1]['Year'])
#     if this_year == next_year - 1:
#         data.at[i, 'player_id'] = player_id
#     else:
#         data.at[i, 'player_id'] = player_id
#         player_id = player_id + 1
