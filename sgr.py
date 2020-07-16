#!/usr/bin/env python3

import json
import requests
import time
import xml.etree.ElementTree as ET
import warnings

from colored import fg, attr
from lxml import html
import mord
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from tqdm import tqdm
from xgboost import XGBRegressor

# Globals
WAIT_FOR_RESP_DOWNLOAD = 0.10
NUM_OF_TAGS = 20
TAGS_MAP = {
    # Dark
    'Dark Fantasy': 'Dark',
    'Dark Humor': 'Dark',
    # Golf
    'Mini Golf': 'Golf',
    # Turn-Based
    'Turn-Based Combat': 'Turn-Based',
    'Turn-Based Strategy': 'Turn-Based',
    'Turn-Based Tactics': 'Turn-Based',
    # Platformer
    '3D Platformer': 'Platformer',
    #Puzzle
    'Puzzle Platformer': 'Puzzle',
}

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
    Gets the app tags from a Steam Store Page

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
    

def recommend_games():
    # === Get User's Review Data ===
    print(f'{fg("cyan")}::{attr("reset")}  Getting Steam user review data and gathering game metadata...')
    # Get Data
    df = pd.read_excel('~/gdrive/video_games/reviews/reviews_and_wishlist.xlsx', skiprows=2)
    df = df[df['Steam AppID'].notnull()]
    df['Steam AppID'] = df['Steam AppID'].astype(int)
    df = df.set_index('Steam AppID')
    print(df)
    
    # Keep Relevant Cols from review excel
    df = df[['Game', 'Score']]
    
    # Add Reviews metadata cols
    df['Recent Percent'] = 0
    df['All Percent'] = 0
    for index in tqdm(df.index, desc=' Getting train Steam Game recent and all review percentages'):
        recent_p, _, all_p, _ = get_appid_reviews(index)
        df.at[index, 'Recent Percent'] = recent_p
        df.at[index, 'All Percent'] = all_p
        
    # Get Tags
    tags_dict = {}
    for index in tqdm(df.index.values, desc=' Getting train Steam Game tags'):
        tags = get_appid_tags(index)
        tags = list(dict.fromkeys([TAGS_MAP[tag] if tag in TAGS_MAP.keys() else tag for tag in tags]))  # Map specific tags to more general ones
        tags_dict[index] = tags[:NUM_OF_TAGS]
            
    UNIQUE_TAGS= sorted(list(set().union(*list(tags_dict.values()))))
    print('-- Unique Training Tags --')
    print(UNIQUE_TAGS)
    
    # Create tag columns
    for tag in UNIQUE_TAGS:
        df[tag] = 0
                
    # Map tag rank to df
    for index, row in tqdm(df.iterrows(), desc=' Mapping tags to columns and ranking based on importance'):
        tags = tags_dict[index]
                    
        for tag, rank in zip(tags, np.arange(len(tags), 0, -1)):
            df.at[index, tag] = int(rank)  # For Importance Ranking
            #df.at[index, tag] = 1  # Binary Has/Not Has Flag

    # === Creating training dataframes ===
    print(f'{fg("cyan")}::{attr("reset")}  Creating training set dataframes...')
    
    ids = df['Game']
    y = df['Score']
    X = df.drop(['Game', 'Score'], axis=1)

    # === Auto Hyperparam Tuning and Model training ===
    print(f'{fg("cyan")}::{attr("reset")}  Tuning hyperparameters and training recommender...')

    #n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    #max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    #min_samples_split = [2, 5, 10]
    #min_samples_leaf = [1, 2, 4]
    #bootstrap = [True, False]
    #random_grid = {'n_estimators': n_estimators,
    #               'max_depth': max_depth,
    #               'min_samples_split': min_samples_split,
    #               'min_samples_leaf': min_samples_leaf,
    #               'bootstrap': bootstrap}
    #model = XGBRegressor(objective='reg:squarederror')
    #rf_random = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=100, cv=3, verbose=0, random_state=42, n_jobs=4)
    #rf_random.fit(X, y)

    # Fit Model
    #model = XGBRegressor(objective='reg:squarederror', **rf_random.best_params_, iid=True, random_state=42, verbose=0)
    model = mord.OrdinalRidge()
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
        tags = list(dict.fromkeys([TAGS_MAP[tag] if tag in TAGS_MAP.keys() else tag for tag in tags]))  # Map specific tags to more general ones
        tags = tags[:NUM_OF_TAGS]
    
        for tag, rank in zip(tags, np.arange(len(tags), 0, -1)):
            if tag in UNIQUE_TAGS:
                df_test.at[index, tag] = int(rank)  # For Importance Ranking
                #df_test.at[index, tag] = 1  # Binary Has/Not Has Flag
            else:
                #print(f'tag "{tag}" not in input -- ignoring')
                pass

    # === Analysis and Recommendations ===
    print(f'{fg("cyan")}::{attr("reset")}  Getting analysis and recommendations...')
    # Get X Test df
    df_test = df_test[df_test['Is DLC'] == False]  # Filter to games
    test_names = df_test['Game']
    test_owned = df_test['Is Owned']
    X_test = df_test.drop(['Game', 'Is Owned', 'Is DLC'], axis=1)

   # use shap explainer to pull most important features influencing preds
    explainer = shap.LinearExplainer(model, X, feature_dependence="independent")
    shap_values = explainer.shap_values(X_test)
    df_shap = pd.DataFrame(shap_values, columns=X_test.columns)
    df_shap.index = X_test.index
    shap_mean_values = df_shap.abs().mean().sort_values(ascending=False).round(3)
    shap_mean_values = shap_mean_values[shap_mean_values!=0]

    print('\n== Most Impactful Features ==')
    print(shap_mean_values)        
    
    # Get predictions
    test_preds = model.predict(X_test)

    # Get Top features
    top_features_dict = {}
    for i, row in df_shap.iterrows():
        top_features = row.abs().sort_values(ascending=False)
        top_features = top_features[top_features >= 0.01].index.values
        top_features_dict[test_names[i]] = row[top_features]
    
    output_data = {
        'Game': test_names,
        'Is Owned': test_owned,
        'Predicted Score': test_preds
    }
    output_df = pd.DataFrame(output_data).sort_values('Predicted Score', ascending=False)

    print(f'Expected Value: {explainer.expected_value:0.2f}')

    print('\n== Top 10 Recommended Games ==')
    print()
    for game_index, row in output_df.head(10).iterrows():
        print(row['Game'])
        print(f'Owned: {row["Is Owned"]}')
        print(f'Predicted Rating: {row["Predicted Score"]:0.2f}')
        print()

        # Get most impactful tags
        shap_features = top_features_dict[row['Game']]
        tags_attr = []
        for tag_index, value in zip(shap_features.index, shap_features):
            tags_attr.append([tag_index, X_test.loc[game_index, tag_index], value])
        df_tags = pd.DataFrame(tags_attr, columns=['Tag', 'Tag Value', 'Tag Impact']).set_index('Tag')
        print(df_tags)
        print()
    
    print('\n== Bottom 10 Not-Recommended Games ==')
    print()
    for game_index, row in output_df.tail(10).iterrows():
        print(row['Game'])
        print(f'Owned: {row["Is Owned"]}')
        print(f'Predicted Rating: {row["Predicted Score"]:0.2f}')
        print()

        # Get most impactful tags
        shap_features = top_features_dict[row['Game']]
        tags_attr = []
        for tag_index, value in zip(shap_features.index, shap_features):
            tags_attr.append([tag_index, X_test.loc[game_index, tag_index], value])
        df_tags = pd.DataFrame(tags_attr, columns=['Tag', 'Tag Value', 'Tag Impact']).set_index('Tag')
        print(df_tags)
        print()

    return model, output_df, shap_values, X_test


if __name__ == '__main__':
    _, _, _, _ = recommend_games()
