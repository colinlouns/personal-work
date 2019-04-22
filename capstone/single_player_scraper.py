import requests
from bs4 import BeautifulSoup
import string
import pandas as pd
import time
import numpy as np

#-------------------------------------------------------------------------------

# create list of player names and slugs for scraping

def make_player_dict(initial, input_name):

    r = requests.get(f'https://www.baseball-reference.com/players/{initial}/')
    soup = BeautifulSoup(r.text)
    player_name_and_link = soup.find('a', text = f'{input_name}')
    print(player_name_and_link)
    player = {}
    player['name'] = player_name_and_link.text
    player['slug'] = player_name_and_link['href'].split('/')[-1]

    return player

# scraper to pull soup for the players
def player_table_scraper(initial, player_dict):

    print('Scraping {}'.format(player_dict["name"]))
    player_req = requests.get(f'https://www.baseball-reference.com/players/{initial}/{player_dict["slug"]}')
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

def player_scraper(input_name):

    print(f'Scraping the stats for {input_name}')
    lastname_initial = input_name.split(' ')[-1][0].lower()
    input_name_camel = input_name.lower().replace(' ', '_')

    player_dict = make_player_dict(initial = lastname_initial, input_name = input_name)

    player_soup = player_table_scraper(initial = lastname_initial, player_dict = player_dict)

    table = player_soup.find('table', {'id': 'batting_standard'})

    try:
        player_stats = grab_and_clean(table_data = table, player = player_dict)
        print("has stats")
        player_stats.to_csv(f'./individual_player_stats/{input_name_camel}.csv')
    except:
        print('errored')

    time.sleep(3 + np.random.uniform())

    return player_stats
