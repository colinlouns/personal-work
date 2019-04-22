import requests
from bs4 import BeautifulSoup
import string
import pandas as pd
import time
import numpy as np


for i in list(string.ascii_lowercase)[0:4]:

    print(f'Scraping the {i}\'s')

    r = requests.get(f'https://www.baseball-reference.com/players/{i}/')
    soup = BeautifulSoup(r.text)
    players_by_lastname = soup.find( 'div', { 'class': "section_content" } )

    player_list = []

    for i in players_by_lastname.find_all('a'):
        player = {}
        player['name'] = i.text
        player['slug'] = i['href'].split('/')[-1]

        player_list.append(player)

player_res = requests.get('https://www.baseball-reference.com/players/a/aaronha01.shtml')
player_soup = BeautifulSoup(player_res.text)

player_soup
table = player_soup.find('table', {'id': 'batting_standard'})
table
try:
    stats = pd.read_html(str(table))[0]


stats['Year'] = pd.to_numeric(stats['Year'], errors = 'coerce')
important_stats = stats.dropna(subset = ['Year'])
important_stats

player_stats = pd.DataFrame([])

for player in player_list[0:10]:

        print('Scraping {}'.format(player["name"]))

        player_res = requests.get('https://www.baseball-reference.com/players/d/{}'.format(player["slug"]))
        player_soup = BeautifulSoup(player_res.text)
        table = player_soup.find('table', {'id': 'batting_standard'})
        try:
            stats = pd.read_html(str(table))[0]
            stats['Year'] = pd.to_numeric(stats['Year'], errors = 'coerce')
            important_stats = stats.dropna(subset = ['Year'])

            player_stats = player_stats.append(important_stats)
            print("has stats")
        except:
            print('errored')
            continue

for player in player_list[0:2]:
    print('Scraping {}'.format(player["name"]))

    player_res = requests.get(f'https://www.baseball-reference.com/players/a/{player["slug"]}')
    player_soup = BeautifulSoup(player_res.text)
    table = player_soup.find('table', {'id': 'batting_standard'})
    print('{}'.format(player["name"], player_soup.select('caption'))

    try:
        stats = pd.read_html(str(table))[0]
        stats['Year'] = pd.to_numeric(stats['Year'], errors = 'coerce')
        important_stats = stats.dropna(subset = ['Year'])

        player_stats = player_stats.append(important_stats)
        print("has stats")
    except:
        print('errored')
        continue



            player_stats
