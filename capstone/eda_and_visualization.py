import pandas as pd
import altair as alt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import svm, linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from math import sqrt

alt.renderers.enable('nteract')


data = pd.read_csv('./player_stats.csv')
data.columns

data.rename(columns={'name': 'player_id'}, inplace=True)


# need to add/remove player_id when using to_numeric further down
important_columns = ['2B', '3B', 'AB', 'Age', 'BA', 'BB', 'H', 'HR', 'OBP',
                     'OPS', 'OPS+', 'PA', 'R', 'RBI', 'SB', 'SLG', 'SO', 'TB',
                     'Year']
data

data_tester2 = data.head(5)



data[important_columns].info()
cleaned_data = data[important_columns].dropna()

for column in important_columns:
    cleaned_data[column] = pd.to_numeric(cleaned_data[column], errors = 'coerce')

# chose data from the more "modern" era of baseball
finalized_data = cleaned_data[cleaned_data['Year'] > 2000]
# chose data from players who had at least 100 plate appearences to ensure significance
final_data = finalized_data[finalized_data['PA'] > 99]
final_data[final_data['Year'] == 2017]

target_stats = ['H', 'HR', 'R', 'RBI']
final_data.info()
final_data.describe().transpose()

hits_df = stat_df_generator(input_stat = 'H', player_stats_df = final_data)
home_runs_df = stat_df_generator(input_stat = 'HR', player_stats_df = final_data)

# make diffence from final stat columns for the target stats

def stat_df_generator(input_stat, player_stats_df):

    unique_ids = list(np.unique([player_stats_df['player_id']]))
    desired_stat_df = pd.DataFrame([])

    for i in list(unique_ids):

        temp_player_frame = player_stats_df[player_stats_df['player_id'] == i]

        try:
            temp_stat_df = pd.DataFrame([])
            temp_stat_df['player_id'] = [i]
            final_year = temp_player_frame.iloc[-1]['Year']
            temp_stat_df[f'{input_stat}_0'] = temp_player_frame.iloc[-1][f'{input_stat}']

            for x in range(1, len(temp_frame)-1):
                current_year = temp_player_frame.iloc[(-1 - x)]['Year']
                current_year_stat = temp_player_frame.iloc[(-1 - x)][f'{input_stat}']
                temp_stat_df[f'{input_stat}_{final_year - current_year}'] = [current_year_hr]

            desired_stat_df = pd.concat([desired_stat_df, temp_stat_df], sort = False )

        except:
            continue

    return desired_stat_df

# ------------------------------------------------------------------------------

alt.Chart(final_data.sample(4999)).mark_circle(size = 80, opacity = 0.1).encode(
    alt.X('Age', title = "Age"),
    alt.Y('HR', title = "Home Runs"),
).properties(title= "Age vs. Home Runs"
).interactive(
)

alt.Chart(final_data.sample(4999)).mark_circle(size = 80, opacity = 0.1).encode(
    alt.X('H', title = "Hits"),
    alt.Y('HR', title = "Home Runs"),
).properties(title= "Hits vs. Home Runs"
).interactive(
)

alt.Chart(final_data.sample(4999)).mark_circle(size = 80, opacity = 0.1).encode(
    alt.X('H', title = "Hits"),
    alt.Y('BA', title = "Batting Average"),
).properties(title= "Hits vs. Batting Average"
).interactive(
)
