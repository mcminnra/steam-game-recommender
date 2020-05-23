#!/usr/bin/env python3

import json
import requests
import time
import xml.etree.ElementTree as ET
import warnings

from colored import fg, attr
from lxml import html
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from tqdm import tqdm
from xgboost import XGBRegressor

# Globals
WAIT_FOR_RESP_DOWNLOAD = 0.10


def get_steam_app_info(appid):
    """
    Gets information and description about a steam app

    Parameters
    ----------
    appid : int
        Steam app's appid (i.e. every game has one. It's in the address bar of a game's store page)

    Returns
    -------
    json
        appdetails JSON response
    """
    r = requests.get(f'https://store.steampowered.com/api/appdetails?appids={appid}')
    time.sleep(WAIT_FOR_RESP_DOWNLOAD)
    return json.loads(r.text)


def get_steam_store_html(appid):
    """
    Gets raw Steam store page HTML for a appid

    Parameters
    ----------
    appid : int
        Steam app's appid (i.e. every game has one. It's in the address bar of a game's store page)

    Returns
    -------
    html
        lxml html tree
    """
    r = requests.get(f'https://store.steampowered.com/app/{appid}')
    time.sleep(WAIT_FOR_RESP_DOWNLOAD)
    return html.fromstring(r.text)


def get_appid_tags(appid):
    """
    Gets the 10 app tags from a Steam Store Page

    Parameters
    ----------
    appid : int
        Steam app's appid (i.e. every game has one. It's in the address bar of a game's store page)

    Returns
    -------
    list
        list of tags in order of relevance
    """
    tree = get_steam_store_html(appid)
    tags = [tag.strip() for tag in tree.xpath('//a[@class="app_tag"]/text()')]
    return tags


def get_appid_reviews(appid):
    """
    Gets reviews (count and like%) from recent and all reviews on a steam store page

    Parameters
    ----------
    appid : int
        Steam app's appid (i.e. every game has one. It's in the address bar of a game's store page)

    Returns
    -------
    list
        [recent%, recent_count, all%, all_count]
    """
    tree = get_steam_store_html(appid)
    reviews = [review.strip() for review in tree.xpath('//span[@class="nonresponsive_hidden responsive_reviewdesc"]/text()') if '%' in review]
    
    # Remove some chars
    reviews = [r.replace(',', '').replace('%', '') for r in reviews]
    
    # Grab only numbers from reviews
    if len(reviews) == 1:
        #if no recent reviews, make recent the same as all
        recent_r = [int(s) for s in reviews[0].split() if s.isdigit()]
        all_r = [int(s) for s in reviews[0].split() if s.isdigit()]
    elif len(reviews) == 0:
        #if no reviews, set to 0
        recent_r = [0, 0]
        all_r = [0, 0]
    else: 
        recent_r = [int(s) for s in reviews[0].split() if s.isdigit()][:2]
        all_r = [int(s) for s in reviews[1].split() if s.isdigit()]
    
    return recent_r+all_r
    
    
def is_dlc(appid):
    """
    Checks to see if an appid is a DLC item

    Parameters
    ----------
    appid : int
        Steam app's appid (i.e. every game has one. It's in the address bar of a game's store page)

    Returns
    -------
    bool
    """
    r = requests.get(f'https://store.steampowered.com/api/appdetails/?appids={appid}')
    time.sleep(WAIT_FOR_RESP_DOWNLOAD)
    store_data = json.loads(r.text)
    
    try:
        app_type = store_data[str(appid)]['data']['type']
    except Exception:
        return False
    
    if app_type == 'dlc':
        return True
    else:
        return False
    
    
def get_wishlist_df():
    """
    Gets a user's wishlist and combines it with metadata

    Parameters
    ----------

    Returns
    -------
    Pandas DataFrame
    """
    r = requests.get('https://store.steampowered.com/wishlist/profiles/76561198053753111/wishlistdata/?p=0')
    time.sleep(WAIT_FOR_RESP_DOWNLOAD)
    
    wishlist = json.loads(r.text)
    appids = list(wishlist.keys())
    games = [wishlist[appid]['name'] for appid in appids]
    
    data = {
        'Game': games,
        'AppID': appids,
        'Is Owned': False,
        'Recent Percent': 0,
        'All Percent': 0,
        'Is DLC': None
    }
    df = pd.DataFrame(data).set_index('AppID')
    for index in tqdm(df.index, desc=' Getting wishlist metadata'):
        recent_p, _, all_p, _ = get_appid_reviews(index)
        df.at[index, 'Recent Percent'] = recent_p
        df.at[index, 'All Percent'] = all_p
        df.at[index, 'Is DLC'] = is_dlc(index)
        
    return df


def get_library_df():
    """
    Gets a user's current library and combines it with metadata
    Parameters
    ----------

    Returns
    -------
    Pandas DataFrame
    """
    r = requests.get('https://steamcommunity.com/id/ryder___/games?tab=all&xml=1')
    time.sleep(WAIT_FOR_RESP_DOWNLOAD)
    
    root = ET.fromstring(r.text)[2]
  
    games = []
    appids = []
    for game in root.findall('game'):
        games.append(game.find('name').text)
        appids.append(game.find('appID').text)
        
    data = {
        'Game': games,
        'AppID': appids,
        'Is Owned': True,
        'Recent Percent': 0,
        'All Percent': 0,
        'Is DLC': None
    }
    df = pd.DataFrame(data).set_index('AppID')
    for index in tqdm(df.index, desc=' Getting library metadata'):
        recent_p, _, all_p, _ = get_appid_reviews(index)
        df.at[index, 'Recent Percent'] = recent_p
        df.at[index, 'All Percent'] = all_p
        df.at[index, 'Is DLC'] = is_dlc(index)
        
    return df
    

if __name__ == '__main__':
    #print(json.loads(requests.get(f'https://store.steampowered.com/api/appdetails/?appids=774461').text))
    #import sys
    #sys.exit()

    # === Get User's Review Data ===
    print(f'{fg("cyan")}::{attr("reset")}  Getting Steam user review data and gathering metadata...')
    # Get Data
    df = pd.read_excel('~/gdrive/video_games/reviews/reviews_and_wishlist.xlsx', skiprows=2).set_index('Steam AppID')
    
    # Keep Relevant Cols from review excel
    df = df[['Game', 'Score']]
    
    # Add Reviews metadata cols
    df['Recent Percent'] = 0
    df['All Percent'] = 0
    for index in tqdm(df.index, desc=' Getting train reviews'):
        recent_p, _, all_p, _ = get_appid_reviews(index)
        df.at[index, 'Recent Percent'] = recent_p
        df.at[index, 'All Percent'] = all_p
        
    # Get Tags
    tags_dict = {}
    for index in tqdm(df.index.values, desc=' Getting train tags'):
        tags = get_appid_tags(index)
        tags_dict[index] = tags
            
    UNIQUE_TAGS= sorted(list(set().union(*list(tags_dict.values()))))
            
    # Create tag columns
    for tag in UNIQUE_TAGS:
        df[tag] = 0
                
    # Map tag rank to df
    for index, row in tqdm(df.iterrows(), desc=' Mapping tags to ranked training columns'):
        tags = tags_dict[index]
                    
        for tag, rank in zip(tags, np.arange(len(tags), 0, -1)):
            df.at[index, tag] = int(rank)

    # == Tags Pivot Summary ==
    # Init tags dict
    tags_score = {}
    for tag in df.columns[4:]:
        tags_score[tag] = {}
        tags_score[tag]['Score'] = []
        tags_score[tag]['All Percent'] = []

    # Get tag scores
    for i, row in df.iterrows():
        # Get Top 5 tags for each game
        tag_cols = row.index[4:]
        top_5 = [tag for tag in tag_cols if row[tag] >= 15]

        # Map reviews to top 5 tags
        for tag in top_5:
            tags_score[tag]['Score'] += [row['Score']]
            tags_score[tag]['All Percent'] += [row['All Percent']]

    # Abusing DataFrames here to take the mean of values in a column
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        df_tags = pd.DataFrame(tags_score).T
        df_tags['Score'] = [np.round(np.array(x).mean(), 2) for x in df_tags['Score'].values]
        df_tags['All Percent'] = [np.round(np.array(x).mean(), 2) for x in df_tags['All Percent'].values]
        df_tags = df_tags.dropna().sort_values(by='Score', ascending=False)

    print(f' Your Tags Sorted By Score')
    print(df_tags)
    
    # === Creating training dataframes ===
    print(f'{fg("cyan")}::{attr("reset")}  Creating training dataframes...')
    
    ids = df['Game']
    y = df['Score']
    X = df.drop(['Game', 'Score'], axis=1)

    # === Auto Hyperparam Tuning and Model training ===
    print(f'{fg("cyan")}::{attr("reset")}  Tuning hyperparameters and training recommender...')

    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    model = XGBRegressor(objective='reg:squarederror')
    rf_random = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=100, cv=3, verbose=0, random_state=42, n_jobs=4)
    rf_random.fit(X, y)

    # Fit Model
    model = XGBRegressor(objective='reg:squarederror', **rf_random.best_params_, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)

    print(f' MSE: {mean_squared_error(y, y_pred)}')

    # === Getting user library and wishlist games to generate recommendations ===
    print(f'{fg("cyan")}::{attr("reset")}  Getting Steam user library and wishlist games to generate recommendations...')
    # Get Test Data
    df_test = pd.concat([get_library_df(), get_wishlist_df()])
    df_test = df_test.drop([str(x) for x in df.index.values])

    # Grab model tag input
    for tag in UNIQUE_TAGS:
        df_test[tag] = 0

    # Get tag ranks for tags that exist in model input
    for index in tqdm(df_test.index.values, desc=' Getting test tags and mapping to train tag inputs'):
        tags = get_appid_tags(index)
    
        for tag, rank in zip(tags, np.arange(len(tags), 0, -1)):
            if tag in UNIQUE_TAGS:
                df_test.at[index, tag] = int(rank)
            else:
                #print(f'tag "{tag}" not in input -- ignoring')
                pass

    # === Predicting test rank ===
    print(f'{fg("cyan")}::{attr("reset")}  Getting recommendations...')
    df_test = df_test[df_test['Is DLC'] == False]  # Filter to games

    test_names = df_test['Game']
    test_owned = df_test['Is Owned']
    X_test = df_test.drop(['Game', 'Is Owned', 'Is DLC'], axis=1)
    test_preds = model.predict(X_test)

    output_data = {
        'Game': test_names,
        'Is Owned': test_owned,
        'Predicted Score': test_preds
    }
    output_df = pd.DataFrame(output_data).sort_values('Predicted Score', ascending=False)

    print('\n== Top 25 ==')
    print(output_df.head(25))
    print('\n== Bottom 25 ==')
    print(output_df.tail(25))
