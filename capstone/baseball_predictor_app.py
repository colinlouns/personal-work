import pickle
import pandas as pd
from flask import Flask, request, render_template, jsonify
from flask_functions import data_to_model
from single_player_scraper import player_scraper

app = Flask(__name__)

#-------- MODEL GOES HERE -----------#
hr_model = pickle.load(open("hr_model.pkl","rb"))
hits_model = pickle.load(open("hits_model.pkl","rb"))
rbi_model = pickle.load(open("rbi_model.pkl","rb"))
runs_model = pickle.load(open("runs_model.pkl","rb"))
sb_model = pickle.load(open("sb_model.pkl","rb"))
ba_model = pickle.load(open("ba_model.pkl","rb"))

#-------- ROUTES GO HERE -----------#

@app.route('/')
def my_form():
    return render_template('entry_form.html')

@app.route('/', methods=['POST'])
def entry_form_post():
    input_name = request.form['name']
    stat_requested = request.form['stat']

    player_df = player_scraper(input_name = input_name)

    last_stat = list(player_df[f'{stat_requested}'])[-1]

    if stat_requested.lower() == 'hr':
        model = hr_model
    if stat_requested.lower() == 'h':
        model = hits_model
    if stat_requested.lower() == 'rbi':
        model = rbi_model
    if stat_requested.lower() == 'r':
        model = runs_model
    if stat_requested.lower() == 'sb':
        model = sb_model
    if stat_requested.lower() == 'ba':
        model = ba_model

    prediction = data_to_model(player_df, stat_requested, model)
    print(prediction)
    results = {f'Predicted {stat_requested}:' : abs(round(prediction[0], 1)), f'Last Year\'s {stat_requested}' : last_stat}
    return jsonify(results)

if __name__ == '__main__':
    '''Connects to the server'''

    HOST = '127.0.0.1'
    PORT = 4000

    app.run(HOST, PORT)
