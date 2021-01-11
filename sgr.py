#!/usr/bin/env python3

import json
import requests
import time
import xml.etree.ElementTree as ET
import warnings

from lxml import html
import numpy as np
import pandas as pd
from rich import print
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table
import shap
from sklearn.model_selection import GridSearchCV, cross_val_score
from xgboost import XGBRegressor

# Globals
WAIT_FOR_RESP_DOWNLOAD = 0.10
NUM_OF_TAGS = 20
TAGS_MAP = {
    # Co-op
    'Online Co-Op': 'Co-op',
    'Local Co-Op': 'Co-op',
    'Co-op Campaign': 'Co-op',
    # Dark
    'Dark Fantasy': 'Dark',
    'Dark Humor': 'Dark',
    # Golf
    'Mini Golf': 'Golf',
    # Local Multiplayer
    '4 Player Local': 'Local Multiplayer',
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
    for index in track(df.index, description='Getting Steam Wishlist Games and Metadata'):
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
    for index in track(df.index, description='Getting Steam Library Games and Metadata'):
        recent_p, _, all_p, _ = get_appid_reviews(index)
        df.at[index, 'Recent Percent'] = recent_p
        df.at[index, 'All Percent'] = all_p
        df.at[index, 'Is DLC'] = is_dlc(index)
        
    return df
    

def recommend_games():
    # get print console
    console = Console()
    
    # === Get User's Review Data ===
    print(f'::  Getting Steam user review data and gathering game metadata...')
    # Get Data
    df = pd.read_excel('~/gdrive/video_games/reviews/reviews_and_wishlist.xlsx', skiprows=2)
    df = df[df['Steam AppID'].notnull()]
    df['Steam AppID'] = df['Steam AppID'].astype(int)
    df = df.set_index('Steam AppID')

    # Table - Rated Games
    rated_games_table = Table(title="Rated Games", show_header=True, header_style="bold purple")
    rated_games_table.add_column("Steam AppId", style="cyan")
    rated_games_table.add_column("Game")
    rated_games_table.add_column("Score", justify="right", style="bold")
    for i, row in df.iterrows():
        # Color according to score value
        score_string = str(row.Score)
        if row.Score >= 7:
            score_string = "[green]" + score_string
        elif row.Score < 4:
            score_string = "[red]" + score_string
        rated_games_table.add_row(str(row.name), row.Game, score_string)
    console.print(rated_games_table)
    
    # Keep Relevant Cols from review excel
    df = df[['Game', 'Score']]
    
    # Add Reviews metadata cols
    df['Recent Percent'] = 0
    df['All Percent'] = 0
    for index in track(df.index, description='Getting Steam Recent and All Review Percentages for Rated Games'):
        recent_p, _, all_p, _ = get_appid_reviews(index)
        df.at[index, 'Recent Percent'] = recent_p
        df.at[index, 'All Percent'] = all_p
        
    # Get Tags
    tags_dict = {}
    for index in track(df.index.values, description='Getting Steam Tags for Rated Games'):
        tags = get_appid_tags(index)
        tags = list(dict.fromkeys([TAGS_MAP[tag] if tag in TAGS_MAP.keys() else tag for tag in tags]))  # Map specific tags to more general ones
        tags_dict[index] = tags[:NUM_OF_TAGS]

    UNIQUE_TAGS= sorted(list(set().union(*list(tags_dict.values()))))

    # Panel - Unique Tags
    print(Panel(Columns(UNIQUE_TAGS, equal=True), title="Unique Training Tags"))
    
    # Create tag columns
    for tag in UNIQUE_TAGS:
        df[tag] = 0
                
    # Map tag rank to df
    for index, row in track(df.iterrows(), description='Creating Tag columns and ranking based on importance', total=len(df)):
        tags = tags_dict[index]
                    
        for tag, rank in zip(tags, np.arange(len(tags), 0, -1)):
            df.at[index, tag] = int(rank)  # For Importance Ranking
            #df.at[index, tag] = 1  # Binary Has/Not Has Flag

    # === Creating training dataframes ===
    print(f'::  Creating training set dataframes...')
    
    ids = df['Game']
    y = df['Score']
    X = df.drop(['Game', 'Score'], axis=1)

    # == Add Content Similarity Score ==
    X_arr = X.drop(['Recent Percent', 'All Percent'], axis=1)
    tag_columns = X_arr.columns
    X_arr = X_arr.astype(float).values
    y_arr = y.astype(float).values

    # Get Pseudo Weighted Average by rating for items
    for i in range(0, len(y_arr)):
        X_arr[i] = X_arr[i] * y_arr[i]
    u_profile = X_arr.sum(axis=0)
    u_profile = (u_profile / np.sqrt(np.sum(u_profile**2)))*NUM_OF_TAGS
    u_profile = pd.Series(u_profile, index=tag_columns)

    # Get cosine sim between weighted average profile and X items
    cosine_scores = []
    for i, row in X.iterrows():
        x = row[2:].values
        sim = np.dot(u_profile.values, x)/(np.linalg.norm(u_profile.values)*np.linalg.norm(x))
        sim = np.round(sim, 4)
        cosine_scores.append(sim)
    X['User Profile Similarity'] = cosine_scores

    # === Auto Hyperparam Tuning and Model training ===
    print(f'::  Tuning hyperparameters and training recommender...')
    objective = ['reg:squarederror']
    max_depth = [int(x) for x in np.linspace(0, 15, num = 16) if x != 0]
    n_estimators = [int(x) for x in np.linspace(start = 0, stop = 500, num = 501) if x != 0]
    random_grid = {
        'objective': objective,
        'max_depth': max_depth,
        'n_estimators': n_estimators,
    }
    model = XGBRegressor(random_state=42, verbosity=0, n_jobs=-1)
    search = GridSearchCV(estimator=model, param_grid=random_grid, scoring='neg_mean_squared_error', cv=3, verbose=1, n_jobs=-1)
    search.fit(X, y)
    print(search.best_params_)

    # Fit Model
    model = XGBRegressor(**search.best_params_, random_state=42, verbosity=0, n_jobs=-1)
    model.fit(X, y)

    # Get Cross Val Score
    scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)
    print(f' Avg. MSE: {scores.mean():0.4f} (+/- {scores.std():0.4f})')

    # === Getting user library and wishlist games to generate recommendations ===
    print(f'::  Getting Steam user library and wishlist games to generate recommendations...')
    # Get Test Data
    df_test = pd.concat([get_library_df(), get_wishlist_df()])
    df_test = df_test.drop([str(x) for x in df.index.values], errors='ignore')

    # Grab model tag input
    for tag in UNIQUE_TAGS:
        df_test[tag] = 0

    # Get tag ranks for tags that exist in model input
    for index in track(df_test.index.values, description='Getting test tags and mapping to train tag inputs'):
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

    # === Recommendations ===
    print(f'::  Getting recommendations...')
    # Get X Test df
    df_test = df_test[df_test['Is DLC'] == False]  # Filter to games
    test_ids = df_test.index.values
    test_names = df_test['Game']
    test_owned = df_test['Is Owned']
    X_test = df_test.drop(['Game', 'Is Owned', 'Is DLC'], axis=1)

    # Calculate User Profile Sim Scores
    cosine_scores = []
    for i, row in X_test.iterrows():
        x = row[2:].astype(float).values
        sim = np.dot(u_profile.values, x)/(np.linalg.norm(u_profile.values)*np.linalg.norm(x))
        sim = np.round(sim, 4)
        cosine_scores.append(sim)
    X_test['User Profile Similarity'] = cosine_scores
    
    # Get predictions
    test_preds = model.predict(X_test)

    # Formulate Output
    output_data = {
        'Steam AppId': test_ids,
        'Game': test_names,
        'Is Owned': test_owned,
        'User Profile Similarity': X_test['User Profile Similarity'].values,
        'Predicted Score': test_preds
    }
    output_df = pd.DataFrame(output_data).sort_values('Predicted Score', ascending=False)

    # Table - UProfile Tags
    u_profile = u_profile.sort_values(ascending=False)
    u_profile_table = Table(title="User Tags Profile", show_header=True, header_style="bold purple")
    u_profile_table.add_column("Tag")
    u_profile_table.add_column("Estimated Value", justify="right", style="bold")
    for index, value in u_profile.items():
        u_profile_table.add_row(index, f'{value:0.2f}')
    console.print(u_profile_table)

    # Table - All Games Predicted Score
    sim_mean = output_df['User Profile Similarity'].mean()
    sim_std = output_df['User Profile Similarity'].std()
    predicted_games_table = Table(title="Predicted Score for Steam Library and Wishlist Games", show_header=True, header_style="bold purple")
    predicted_games_table.add_column("Steam AppId", style="cyan")
    predicted_games_table.add_column("Game")
    predicted_games_table.add_column("User Profile Similarity", justify="right", style="bold")
    predicted_games_table.add_column("Predicted Score", justify="right", style="bold")
    for i, row in output_df.iterrows():
        # Color score value
        score_string = str(np.round(row['Predicted Score'], 2))
        if row['Predicted Score'] >= 7:
            score_string = "[green]" + score_string
        elif row['Predicted Score'] < 4:
            score_string = "[red]" + score_string

        # Color Similarity
        sim_string = f'{row["User Profile Similarity"]*100:0.2f}%'
        if row['User Profile Similarity'] >= (sim_mean + sim_std):
            sim_string = "[green]" + sim_string
        elif row['User Profile Similarity'] <= (sim_mean - sim_std):
            sim_string = "[red]" + sim_string
        
        predicted_games_table.add_row(row['Steam AppId'], row.Game, sim_string, score_string)
    console.print(predicted_games_table)
        
    return model, output_df, X_test


if __name__ == '__main__':
    _, _, _ = recommend_games()
