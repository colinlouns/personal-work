import requests
from bs4 import BeautifulSoup
import string
import pandas as pd
import time
import numpy as np

# create list of player names and slugs for scraping
def make_player_dict():

    r = requests.get(f'https://www.baseball-reference.com/players/{i}/')
    soup = BeautifulSoup(r.text)
    players_by_lastname = soup.find( 'div', { 'class': "section_content" } )

    player_list = []

    for lastname in players_by_lastname.find_all('a'):
        player = {}
        player['name'] = lastname.text
        player['slug'] = lastname['href'].split('/')[-1]

        player_list.append(player)

    return player_list

# scraper to pull soup for the players
def player_scraper(player_dict):

    print('Scraping {}'.format(player_dict["name"]))
    player_req = requests.get(f'https://www.baseball-reference.com/players/{i}/{player_dict["slug"]}')
    player_soup = BeautifulSoup(player_req.text)
    return player_soup

# pull player stats and remove extra rows from the reference site via the year column.  Also used OPS+ to remove minor
# league stats as they are not wanted, and OPS+ isn't tracked in the minors
def grab_and_clean(table_data, player):

    stats = pd.read_html(str(table_data))[0]
    stats['Year'] = pd.to_numeric(stats['Year'], errors = 'coerce')
    important_stats = stats.dropna(subset = ['Year', 'OPS+'])
    important_stats = important_stats.groupby('Year').max()
    important_stats['player_id'] = player['slug'].split('.')[0]
    return important_stats

# ------------------------------------------------------------------------------

player_stats = pd.DataFrame([])

for i in list(string.ascii_lowercase):

    print(f'Scraping the {i}\'s')

    player_list = make_player_dict()

    for player in player_list:

        player_soup = player_scraper(player)
        table = player_soup.find('table', {'id': 'batting_standard'})

        try:
            important_stats = grab_and_clean(table, player)
            print("has stats")
            player_stats = player_stats.append(important_stats)
            player_stats.to_csv('player_stats.csv')
        except:
            print('errored')
            continue

        time.sleep(np.random.uniform(high=3.1))

    # time.sleep(3 + np.random.uniform())
