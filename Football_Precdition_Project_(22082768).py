#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime as dt
from zipfile import ZipFile


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


folder = '/content/Datasets/Datasets'


# In[ ]:


folder='/content/Datasets/Datasets'
raw_data_1 = pd.read_csv(r'/content/Datasets/2000-01.csv')
raw_data_2 = pd.read_csv(r'/content/Datasets/2001-02.csv')
raw_data_3 = pd.read_csv(r'/content/Datasets/2002-03.csv')
raw_data_4 = pd.read_csv(r'/content/Datasets/2003-04.csv')
raw_data_5 = pd.read_csv(r'/content/Datasets/2004-05.csv')
raw_data_6 = pd.read_csv(r'/content/Datasets/2005-06.csv')
raw_data_7 = pd.read_csv(r'/content/Datasets/2006-07.csv')
raw_data_8 = pd.read_csv(r'/content/Datasets/2007-08.csv')
raw_data_9 = pd.read_csv(r'/content/Datasets/2008-09.csv')
raw_data_10 = pd.read_csv(r'/content/Datasets/2009-10.csv')
raw_data_11 = pd.read_csv(r'/content/Datasets/2010-11.csv')
raw_data_12 = pd.read_csv(r'/content/Datasets/2011-12.csv')
raw_data_13 = pd.read_csv(r'/content/Datasets/2012-13.csv')
raw_data_14 = pd.read_csv(r'/content/Datasets/2013-14.csv')
raw_data_15 = pd.read_csv(r'/content/Datasets/2014-15.csv')
raw_data_16 = pd.read_csv(r'/content/Datasets/2015-16.csv')
raw_data_17 = pd.read_csv(r'/content/Datasets/2016-17.csv')
raw_data_18 = pd.read_csv(r'/content/Datasets/2017-18.csv')


# In[ ]:


# Combine the manually loaded datasets into a list
datasets = [raw_data_1, raw_data_2, raw_data_3, raw_data_4, raw_data_5,
            raw_data_6, raw_data_7, raw_data_8, raw_data_9, raw_data_10,
            raw_data_11, raw_data_12, raw_data_13, raw_data_14, raw_data_15,
            raw_data_16, raw_data_17, raw_data_18]


# In[ ]:


seasons = [f'{year}-{str(year + 1)[-2:]}' for year in range(2000, 2018)]


# In[ ]:


for i, season in enumerate(seasons):
    print(f"Data for {season}:")
    display(datasets[i].head())


# EDA

# In[ ]:


# Displaying data types and null values for all datasets
for i, season in enumerate(seasons):
    print(f"Data Types and Null Values for {season}:")
    print(datasets[i].dtypes)
    print("\nMissing values:\n", datasets[i].isnull().sum())
    print("\n")


# In[ ]:


# Checking unique values and duplicates
for i, season in enumerate(seasons):
    print(f"Unique Values and Duplicates for {season}:")
    print("Unique values:")
    print(datasets[i].nunique())
    print("Number of duplicates:", datasets[i].duplicated().sum())
    print("\n")


# In[ ]:


# Summary statistics for numeric and categorical features
for i, season in enumerate(seasons):
    print(f"Summary Statistics for {season}:")
    print(datasets[i].describe(include='all'))
    print("\n")


# In[ ]:


# Distribution of Goals
for i, season in enumerate(seasons):
    plt.figure(figsize=(12, 6))
    sns.histplot(datasets[i]['FTHG'], bins=10, kde=True, label='Home Goals', color='blue')
    sns.histplot(datasets[i]['FTAG'], bins=10, kde=True, label='Away Goals', color='red')
    plt.title(f'Distribution of Goals Scored - {season}')
    plt.xlabel('Goals')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


# In[ ]:


# Boxplots for Home and Away Goals
for i, season in enumerate(seasons):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=datasets[i][['FTHG', 'FTAG']])
    plt.title(f'Boxplot of Home and Away Goals - {season}')
    plt.xlabel('Team')
    plt.ylabel('Goals')
    plt.xticks([0, 1], ['Home', 'Away'])
    plt.show()


# In[ ]:


# Count plots for Full Time Result (FTR)
for i, season in enumerate(seasons):
    plt.figure(figsize=(8, 4))
    sns.countplot(datasets[i]['FTR'], palette='coolwarm')
    plt.title(f'Distribution of Full Time Result (FTR) - {season}')
    plt.xlabel('Result')
    plt.ylabel('Count')
    plt.show()


# In[ ]:


# Correlation Matrix
for i, season in enumerate(seasons):
    plt.figure(figsize=(14, 10))
    numeric_data = datasets[i].select_dtypes(include=[np.number])
    correlation_matrix = numeric_data.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
    plt.title(f'Correlation Matrix - {season}')
    plt.show()


# In[ ]:


# Average goals scored by Home and Away teams
for i, season in enumerate(seasons):
    home_goals = datasets[i]['FTHG'].mean()
    away_goals = datasets[i]['FTAG'].mean()
    print(f"Average Goals Scored - {season}:")
    print(f"Home: {home_goals:.2f}, Away: {away_goals:.2f}\n")


# In[ ]:


# Calculating win rates for home, away, and draws
for i, season in enumerate(seasons):
    total_matches = datasets[i].shape[0]
    home_wins = datasets[i][datasets[i]['FTR'] == 'H'].shape[0]
    away_wins = datasets[i][datasets[i]['FTR'] == 'A'].shape[0]
    draws = datasets[i][datasets[i]['FTR'] == 'D'].shape[0]

    print(f"Win Rates for {season}:")
    print(f"Home Win Rate: {home_wins / total_matches * 100:.2f}%")
    print(f"Away Win Rate: {away_wins / total_matches * 100:.2f}%")
    print(f"Draw Rate: {draws / total_matches * 100:.2f}%\n")


# In[ ]:


# Calculating goal differences for each match
for i, season in enumerate(seasons):
    datasets[i]['GoalDifference'] = datasets[i]['FTHG'] - datasets[i]['FTAG']
    plt.figure(figsize=(8, 4))
    sns.histplot(datasets[i]['GoalDifference'], bins=20, kde=True, color='purple')
    plt.title(f'Distribution of Goal Differences - {season}')
    plt.xlabel('Goal Difference')
    plt.ylabel('Frequency')
    plt.show()


# In[ ]:


# Analysis of total goals scored over the season
total_goals = []
seasons = [f'{year}-{str(year + 1)[-2:]}' for year in range(2000, 2018)]
for i, season in enumerate(seasons):
    total_goals.append(datasets[i]['FTHG'].sum() + datasets[i]['FTAG'].sum())

plt.figure(figsize=(12, 6))
sns.lineplot(x=seasons, y=total_goals, marker='o')
plt.title('Total Goals Scored per Season')
plt.xlabel('Season')
plt.ylabel('Total Goals')
plt.xticks(rotation=45)
plt.show()


# In[ ]:


# Combine all season datasets into a single DataFrame
combined_data = pd.concat(datasets, ignore_index=True)

# Display the first few rows of the combined dataset
print("Combined DataFrame:")
display(combined_data.head())


# In[ ]:


# Compute the correlation matrix for the combined dataset
numeric_data_combined = combined_data.select_dtypes(include=[np.number])  # Select only numeric columns
correlation_matrix_combined = numeric_data_combined.corr()

# Display the correlation matrix
print("Combined Correlation Matrix:")
display(correlation_matrix_combined)


# In[ ]:


# Select required columns
columns_req = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
playing_statistics_1 = raw_data_1[columns_req]
playing_statistics_2 = raw_data_2[columns_req]
playing_statistics_3 = raw_data_3[columns_req]
playing_statistics_4 = raw_data_4[columns_req]
playing_statistics_5 = raw_data_5[columns_req]
playing_statistics_6 = raw_data_6[columns_req]
playing_statistics_7 = raw_data_7[columns_req]
playing_statistics_8 = raw_data_8[columns_req]
playing_statistics_9 = raw_data_9[columns_req]
playing_statistics_10 = raw_data_10[columns_req]
playing_statistics_11 = raw_data_11[columns_req]
playing_statistics_12 = raw_data_12[columns_req]
playing_statistics_13 = raw_data_13[columns_req]
playing_statistics_14 = raw_data_14[columns_req]
playing_statistics_15 = raw_data_15[columns_req]
playing_statistics_16 = raw_data_16[columns_req]
playing_statistics_17 = raw_data_17[columns_req]
playing_statistics_18 = raw_data_18[columns_req]

# Concatenate all seasons' data
all_data = pd.concat([playing_statistics_1, playing_statistics_2, playing_statistics_3,
                      playing_statistics_4, playing_statistics_5, playing_statistics_6,
                      playing_statistics_7, playing_statistics_8, playing_statistics_9,
                      playing_statistics_10, playing_statistics_11, playing_statistics_12,
                      playing_statistics_13, playing_statistics_14, playing_statistics_15,
                      playing_statistics_16, playing_statistics_17, playing_statistics_18],
                     ignore_index=True)

# Display the first few rows
print(all_data.head())

# Additional EDA Steps
# 1. Distribution of Full Time Results (FTR)
sns.countplot(x='FTR', data=all_data)
plt.title('Distribution of Full Time Results')
plt.show()

# 2. Goals scored by Home vs Away teams
sns.histplot(all_data['FTHG'], bins=20, color='blue', label='Home Goals', kde=True)
sns.histplot(all_data['FTAG'], bins=20, color='red', label='Away Goals', kde=True)
plt.legend()
plt.title('Distribution of Goals Scored')
plt.show()




# In[ ]:


# Select only numeric columns for correlation matrix calculation
numeric_cols = all_data.select_dtypes(include=[np.number]).columns
corr = all_data[numeric_cols].corr()

# Plot the heatmap
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# Feature Engineering

# In[ ]:


def get_goals_scored(playing_stat):
    teams = {team: [] for team in playing_stat['HomeTeam'].unique()}
    for i in range(len(playing_stat)):
        HTGS = playing_stat.iloc[i]['FTHG']
        ATGS = playing_stat.iloc[i]['FTAG']
        teams[playing_stat.iloc[i].HomeTeam].append(HTGS)
        teams[playing_stat.iloc[i].AwayTeam].append(ATGS)
    num_matchweeks = max(len(v) for v in teams.values())
    GoalsScored = pd.DataFrame(data=teams, index=range(1, num_matchweeks + 1)).T
    GoalsScored[0] = 0
    for i in range(2, num_matchweeks + 1):
        GoalsScored[i] = GoalsScored[i] + GoalsScored[i-1]
    return GoalsScored

def get_goals_conceded(playing_stat):
    teams = {team: [] for team in playing_stat['HomeTeam'].unique()}
    for i in range(len(playing_stat)):
        ATGC = playing_stat.iloc[i]['FTHG']
        HTGC = playing_stat.iloc[i]['FTAG']
        teams[playing_stat.iloc[i].HomeTeam].append(HTGC)
        teams[playing_stat.iloc[i].AwayTeam].append(ATGC)
    num_matchweeks = max(len(v) for v in teams.values())
    GoalsConceded = pd.DataFrame(data=teams, index=range(1, num_matchweeks + 1)).T
    GoalsConceded[0] = 0
    for i in range(2, num_matchweeks + 1):
        GoalsConceded[i] = GoalsConceded[i] + GoalsConceded[i-1]
    return GoalsConceded

def get_gss(playing_stat):
    GC = get_goals_conceded(playing_stat)
    GS = get_goals_scored(playing_stat)
    j = 0
    HTGS = []
    ATGS = []
    HTGC = []
    ATGC = []
    num_games = len(playing_stat)
    num_matchweeks = min(GS.shape[1], GC.shape[1])
    for i in range(num_games):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        HTGS.append(GS.loc[ht].iloc[j])
        ATGS.append(GS.loc[at].iloc[j])
        HTGC.append(GC.loc[ht].iloc[j])
        ATGC.append(GC.loc[at].iloc[j])
        if (i + 1) % 10 == 0 and j < num_matchweeks - 1:
            j += 1
    playing_stat['HTGS'] = HTGS
    playing_stat['ATGS'] = ATGS
    playing_stat['HTGC'] = HTGC
    playing_stat['ATGC'] = ATGC
    return playing_stat

# Apply the feature engineering functions to each dataset
playing_statistics_1 = get_gss(playing_statistics_1)
playing_statistics_2 = get_gss(playing_statistics_2)
playing_statistics_3 = get_gss(playing_statistics_3)
playing_statistics_4 = get_gss(playing_statistics_4)
playing_statistics_5 = get_gss(playing_statistics_5)
playing_statistics_6 = get_gss(playing_statistics_6)
playing_statistics_7 = get_gss(playing_statistics_7)
playing_statistics_8 = get_gss(playing_statistics_8)
playing_statistics_9 = get_gss(playing_statistics_9)
playing_statistics_10 = get_gss(playing_statistics_10)
playing_statistics_11 = get_gss(playing_statistics_11)
playing_statistics_12 = get_gss(playing_statistics_12)
playing_statistics_13 = get_gss(playing_statistics_13)
playing_statistics_14 = get_gss(playing_statistics_14)
playing_statistics_15 = get_gss(playing_statistics_15)
playing_statistics_16 = get_gss(playing_statistics_16)
playing_statistics_17 = get_gss(playing_statistics_17)
playing_statistics_18 = get_gss(playing_statistics_18)


# In[ ]:


import pandas as pd

# Assuming 'playing_stat' is the DataFrame for the match statistics

def get_matchres(playing_stat):
    teams = {team: [] for team in playing_stat['HomeTeam'].unique()}
    for i in range(len(playing_stat)):
        if playing_stat.iloc[i].FTR == 'H':
            teams[playing_stat.iloc[i].HomeTeam].append('W')
            teams[playing_stat.iloc[i].AwayTeam].append('L')
        elif playing_stat.iloc[i].FTR == 'A':
            teams[playing_stat.iloc[i].AwayTeam].append('W')
            teams[playing_stat.iloc[i].HomeTeam].append('L')
        else:
            teams[playing_stat.iloc[i].AwayTeam].append('D')
            teams[playing_stat.iloc[i].HomeTeam].append('D')
    return pd.DataFrame(data=teams, index=range(1, 39)).T

def get_points(result):
    if result == 'W':
        return 3
    elif result == 'D':
        return 1
    else:
        return 0

def get_cuml_points(matchres):
    matchres_points = matchres.applymap(get_points)
    num_matchweeks = matchres_points.shape[1]
    matchres_points.insert(0, 0, 0)
    for i in range(2, num_matchweeks + 1):
        matchres_points[i] = matchres_points[i] + matchres_points[i - 1]
    return matchres_points

def get_agg_points(playing_stat):
    matchres = get_matchres(playing_stat)
    cum_pts = get_cuml_points(matchres)
    HTP = []
    ATP = []
    j = 0
    num_games = len(playing_stat)
    num_matchweeks = cum_pts.shape[1]
    for i in range(num_games):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        if j < num_matchweeks:
            HTP.append(cum_pts.loc[ht].iloc[j])
            ATP.append(cum_pts.loc[at].iloc[j])
        if (i + 1) % 10 == 0 and j < num_matchweeks - 1:
            j += 1
    playing_stat['HTP'] = HTP
    playing_stat['ATP'] = ATP
    return playing_stat

def get_form(playing_stat, num):
    matchres = get_matchres(playing_stat)
    form_final = matchres.copy()
    for i in range(num, 39):
        form_final[i] = ''
        for j in range(num):
            form_final[i] += matchres[i-j]
    return form_final

def add_form(playing_stat, num):
    form = get_form(playing_stat, num)
    h = ['M' for _ in range(num * 10)]
    a = ['M' for _ in range(num * 10)]
    j = num
    for i in range(num * 10, len(playing_stat)):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        past = form.loc[ht][j]
        h.append(past[num-1])
        past = form.loc[at][j]
        a.append(past[num-1])
        if ((i + 1) % 10) == 0:
            j += 1
    playing_stat[f'HM{num}'] = h
    playing_stat[f'AM{num}'] = a
    return playing_stat

def add_form_df(playing_statistics):
    for num in range(1, 6):
        playing_statistics = [add_form(df, num) for df in playing_statistics]
    return playing_statistics

# Apply cumulative points calculation
playing_statistics_1 = get_agg_points(playing_statistics_1)
playing_statistics_2 = get_agg_points(playing_statistics_2)
playing_statistics_3 = get_agg_points(playing_statistics_3)
playing_statistics_4 = get_agg_points(playing_statistics_4)
playing_statistics_5 = get_agg_points(playing_statistics_5)
playing_statistics_6 = get_agg_points(playing_statistics_6)
playing_statistics_7 = get_agg_points(playing_statistics_7)
playing_statistics_8 = get_agg_points(playing_statistics_8)
playing_statistics_9 = get_agg_points(playing_statistics_9)
playing_statistics_10 = get_agg_points(playing_statistics_10)
playing_statistics_11 = get_agg_points(playing_statistics_11)
playing_statistics_12 = get_agg_points(playing_statistics_12)
playing_statistics_13 = get_agg_points(playing_statistics_13)
playing_statistics_14 = get_agg_points(playing_statistics_14)
playing_statistics_15 = get_agg_points(playing_statistics_15)
playing_statistics_16 = get_agg_points(playing_statistics_16)
playing_statistics_17 = get_agg_points(playing_statistics_17)
playing_statistics_18 = get_agg_points(playing_statistics_18)

# Adding form (last n matches) data
playing_statistics_1 = add_form_df([playing_statistics_1])[0]
playing_statistics_2 = add_form_df([playing_statistics_2])[0]
playing_statistics_3 = add_form_df([playing_statistics_3])[0]
playing_statistics_4 = add_form_df([playing_statistics_4])[0]
playing_statistics_5 = add_form_df([playing_statistics_5])[0]
playing_statistics_6 = add_form_df([playing_statistics_6])[0]
playing_statistics_7 = add_form_df([playing_statistics_7])[0]
playing_statistics_8 = add_form_df([playing_statistics_8])[0]
playing_statistics_9 = add_form_df([playing_statistics_9])[0]
playing_statistics_10 = add_form_df([playing_statistics_10])[0]
playing_statistics_11 = add_form_df([playing_statistics_11])[0]
playing_statistics_12 = add_form_df([playing_statistics_12])[0]
playing_statistics_13 = add_form_df([playing_statistics_13])[0]
playing_statistics_14 = add_form_df([playing_statistics_14])[0]
playing_statistics_15 = add_form_df([playing_statistics_15])[0]
playing_statistics_16 = add_form_df([playing_statistics_16])[0]
playing_statistics_17 = add_form_df([playing_statistics_17])[0]
playing_statistics_18 = add_form_df([playing_statistics_18])[0]


# In[ ]:


# Preparing the final dataset by concatenating all seasons' data
playing_stat = pd.concat([
    playing_statistics_1, playing_statistics_2, playing_statistics_3,
    playing_statistics_4, playing_statistics_5, playing_statistics_6,
    playing_statistics_7, playing_statistics_8, playing_statistics_9,
    playing_statistics_10, playing_statistics_11, playing_statistics_12,
    playing_statistics_13, playing_statistics_14, playing_statistics_15,
    playing_statistics_16, playing_statistics_17, playing_statistics_18
], ignore_index=True)

# Check if the columns HTFormPts and ATFormPts exist, if not, calculate them
if 'HTFormPts' not in playing_stat.columns or 'ATFormPts' not in playing_stat.columns:
    # Assuming HTFormPtsStr and ATFormPtsStr columns exist
    playing_stat['HTFormPtsStr'] = (
        playing_stat['HM1'] + playing_stat['HM2'] + playing_stat['HM3'] + playing_stat['HM4'] + playing_stat['HM5']
    )
    playing_stat['ATFormPtsStr'] = (
        playing_stat['AM1'] + playing_stat['AM2'] + playing_stat['AM3'] + playing_stat['AM4'] + playing_stat['AM5']
    )

    def get_form_points(string):
        sum_points = 0
        for letter in string:
            sum_points += get_points(letter)
        return sum_points

    playing_stat['HTFormPts'] = playing_stat['HTFormPtsStr'].apply(get_form_points)
    playing_stat['ATFormPts'] = playing_stat['ATFormPtsStr'].apply(get_form_points)

# Calculating additional features
playing_stat['HTGD'] = playing_stat['HTGS'] - playing_stat['HTGC']
playing_stat['ATGD'] = playing_stat['ATGS'] - playing_stat['ATGC']
playing_stat['DiffPts'] = playing_stat['HTP'] - playing_stat['ATP']
playing_stat['DiffFormPts'] = playing_stat['HTFormPts'] - playing_stat['ATFormPts']

# Ensure that MW column is correctly calculated and exists
if 'MW' not in playing_stat.columns:
    def get_mw(df):
        j = 1
        MatchWeek = []
        for i in range(len(df)):
            MatchWeek.append(j)
            if ((i + 1) % 10) == 0:
                j += 1
        df['MW'] = MatchWeek
        return df

    playing_stat = get_mw(playing_stat)

# Scaling certain features by Matchweek
playing_stat['MW'] = playing_stat['MW'].astype(float)
cols_to_scale = ['HTGD', 'ATGD', 'DiffPts', 'DiffFormPts', 'HTP', 'ATP']
for col in cols_to_scale:
    playing_stat[col] = playing_stat[col] / playing_stat['MW']

# Transform the target variable (FTR) to binary classification ('H' or 'NH')
playing_stat['FTR'] = playing_stat['FTR'].apply(lambda x: 'H' if x == 'H' else 'NH')

# Saving the final dataset
playing_stat.to_csv('/content/final_dataset.csv', index=False)

# Loading the dataset for further use
dataset = pd.read_csv('/content/final_dataset.csv')




# In[ ]:


columns_req = ['Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR']

playing_statistics_1 = raw_data_1[columns_req]
playing_statistics_2 = raw_data_2[columns_req]
playing_statistics_3 = raw_data_3[columns_req]
playing_statistics_4 = raw_data_4[columns_req]
playing_statistics_5 = raw_data_5[columns_req]
playing_statistics_6 = raw_data_6[columns_req]
playing_statistics_7 = raw_data_7[columns_req]
playing_statistics_8 = raw_data_8[columns_req]
playing_statistics_9 = raw_data_9[columns_req]
playing_statistics_10 = raw_data_10[columns_req]
playing_statistics_11 = raw_data_11[columns_req]
playing_statistics_12 = raw_data_12[columns_req]
playing_statistics_13 = raw_data_13[columns_req]
playing_statistics_14 = raw_data_14[columns_req]
playing_statistics_15 = raw_data_15[columns_req]
playing_statistics_16 = raw_data_16[columns_req]
playing_statistics_17 = raw_data_17[columns_req]
playing_statistics_18 = raw_data_18[columns_req]


# In[ ]:


import pandas as pd

def get_goals_scored(playing_stat):
    # Create a dictionary with team names as keys
    teams = {team: [] for team in playing_stat['HomeTeam'].unique()}

    # Fill in the goals scored for each team
    for i in range(len(playing_stat)):
        HTGS = playing_stat.iloc[i]['FTHG']
        ATGS = playing_stat.iloc[i]['FTAG']
        teams[playing_stat.iloc[i].HomeTeam].append(HTGS)
        teams[playing_stat.iloc[i].AwayTeam].append(ATGS)

    # Determine the number of matchweeks
    num_matchweeks = max(len(v) for v in teams.values())

    # Create a dataframe for goals scored where rows are teams and columns are matchweek.
    GoalsScored = pd.DataFrame(data=teams, index=range(1, num_matchweeks + 1)).T
    GoalsScored[0] = 0  # Initialize the cumulative sum

    # Aggregate to get cumulative goals up to that point
    for i in range(2, num_matchweeks + 1):
        GoalsScored[i] = GoalsScored[i] + GoalsScored[i-1]

    return GoalsScored


def get_goals_conceded(playing_stat):
    # Create a dictionary with team names as keys
    teams = {team: [] for team in playing_stat['HomeTeam'].unique()}

    # Fill in the goals conceded for each team
    for i in range(len(playing_stat)):
        ATGC = playing_stat.iloc[i]['FTHG']
        HTGC = playing_stat.iloc[i]['FTAG']
        teams[playing_stat.iloc[i].HomeTeam].append(HTGC)
        teams[playing_stat.iloc[i].AwayTeam].append(ATGC)

    # Determine the number of matchweeks
    num_matchweeks = max(len(v) for v in teams.values())

    # Create a dataframe for goals conceded where rows are teams and columns are matchweek.
    GoalsConceded = pd.DataFrame(data=teams, index=range(1, num_matchweeks + 1)).T
    GoalsConceded[0] = 0  # Initialize the cumulative sum

    # Aggregate to get cumulative goals conceded up to that point
    for i in range(2, num_matchweeks + 1):
        GoalsConceded[i] = GoalsConceded[i] + GoalsConceded[i-1]

    return GoalsConceded

def get_gss(playing_stat):
    GC = get_goals_conceded(playing_stat)
    GS = get_goals_scored(playing_stat)

    j = 0
    HTGS = []
    ATGS = []
    HTGC = []
    ATGC = []

    num_games = len(playing_stat)
    num_matchweeks = min(GS.shape[1], GC.shape[1])

    for i in range(num_games):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam

        HTGS.append(GS.loc[ht].iloc[j])
        ATGS.append(GS.loc[at].iloc[j])
        HTGC.append(GC.loc[ht].iloc[j])
        ATGC.append(GC.loc[at].iloc[j])

        if (i + 1) % 10 == 0 and j < num_matchweeks - 1:
            j += 1

    playing_stat['HTGS'] = HTGS
    playing_stat['ATGS'] = ATGS
    playing_stat['HTGC'] = HTGC
    playing_stat['ATGC'] = ATGC

    return playing_stat

# Apply to each dataset
playing_statistics_1 = get_gss(playing_statistics_1)
playing_statistics_2 = get_gss(playing_statistics_2)
playing_statistics_3 = get_gss(playing_statistics_3)
playing_statistics_4 = get_gss(playing_statistics_4)
playing_statistics_5 = get_gss(playing_statistics_5)
playing_statistics_6 = get_gss(playing_statistics_6)
playing_statistics_7 = get_gss(playing_statistics_7)
playing_statistics_8 = get_gss(playing_statistics_8)
playing_statistics_9 = get_gss(playing_statistics_9)
playing_statistics_10 = get_gss(playing_statistics_10)
playing_statistics_11 = get_gss(playing_statistics_11)
playing_statistics_12 = get_gss(playing_statistics_12)
playing_statistics_13 = get_gss(playing_statistics_13)
playing_statistics_14 = get_gss(playing_statistics_14)
playing_statistics_15 = get_gss(playing_statistics_15)
playing_statistics_16 = get_gss(playing_statistics_16)
playing_statistics_17 = get_gss(playing_statistics_17)
playing_statistics_18 = get_gss(playing_statistics_18)


# In[ ]:


import pandas as pd

def get_points(result):
    if result == 'W':
        return 3
    elif result == 'D':
        return 1
    else:
        return 0


def get_cuml_points(matchres):
    # Convert match results to points
    matchres_points = matchres.applymap(get_points)

    # Ensure there's no issue with index when initializing the cumulative points
    num_matchweeks = matchres_points.shape[1]
    num_teams = matchres_points.shape[0]

    matchres_points[0] = 0  # Initialize the cumulative sum with zero points
    for i in range(2, num_matchweeks + 1):
        matchres_points[i] = matchres_points[i] + matchres_points[i - 1]

    return matchres_points


def get_matchres(playing_stat):
    # Create a dictionary with team names as keys
    teams = {team: [] for team in playing_stat['HomeTeam'].unique()}

    # the value corresponding to keys is a list containing the match result
    for i in range(len(playing_stat)):
        if playing_stat.iloc[i].FTR == 'H':
            teams[playing_stat.iloc[i].HomeTeam].append('W')
            teams[playing_stat.iloc[i].AwayTeam].append('L')
        elif playing_stat.iloc[i].FTR == 'A':
            teams[playing_stat.iloc[i].AwayTeam].append('W')
            teams[playing_stat.iloc[i].HomeTeam].append('L')
        else:
            teams[playing_stat.iloc[i].AwayTeam].append('D')
            teams[playing_stat.iloc[i].HomeTeam].append('D')

    # Determine the number of matchweeks based on the length of the values
    num_matchweeks = max(len(results) for results in teams.values())

    # Create a DataFrame with the results and ensure correct indexing
    return pd.DataFrame(data=teams, index=range(1, num_matchweeks + 1)).T

def get_agg_points(playing_stat):
    matchres = get_matchres(playing_stat)
    cum_pts = get_cuml_points(matchres)
    HTP = []
    ATP = []
    j = 0

    num_games = len(playing_stat)
    num_matchweeks = cum_pts.shape[1]

    for i in range(num_games):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam

        if j < num_matchweeks:
            HTP.append(cum_pts.loc[ht].iloc[j])
            ATP.append(cum_pts.loc[at].iloc[j])

        if (i + 1) % 10 == 0 and j < num_matchweeks - 1:
            j += 1

    playing_stat['HTP'] = HTP
    playing_stat['ATP'] = ATP
    return playing_stat

# Apply to each dataset
playing_statistics_1 = get_gss(playing_statistics_1)
playing_statistics_2 = get_gss(playing_statistics_2)
playing_statistics_3 = get_gss(playing_statistics_3)
playing_statistics_4 = get_gss(playing_statistics_4)
playing_statistics_5 = get_gss(playing_statistics_5)
playing_statistics_6 = get_gss(playing_statistics_6)
playing_statistics_7 = get_gss(playing_statistics_7)
playing_statistics_8 = get_gss(playing_statistics_8)
playing_statistics_9 = get_gss(playing_statistics_9)
playing_statistics_10 = get_gss(playing_statistics_10)
playing_statistics_11 = get_gss(playing_statistics_11)
playing_statistics_12 = get_gss(playing_statistics_12)
playing_statistics_13 = get_gss(playing_statistics_13)
playing_statistics_14 = get_gss(playing_statistics_14)
playing_statistics_15 = get_gss(playing_statistics_15)
playing_statistics_16 = get_gss(playing_statistics_16)
playing_statistics_17 = get_gss(playing_statistics_17)
playing_statistics_18 = get_gss(playing_statistics_18)


# In[ ]:


def get_form(playing_stat,num):
    form = get_matchres(playing_stat)
    form_final = form.copy()
    for i in range(num,39):
        form_final[i] = ''
        j = 0
        while j < num:
            form_final[i] += form[i-j]
            j += 1
    return form_final

def add_form(playing_stat,num):
    form = get_form(playing_stat,num)
    h = ['M' for i in range(num * 10)]  # since form is not available for n MW (n*10)
    a = ['M' for i in range(num * 10)]

    j = num
    for i in range((num*10),380):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam

        past = form.loc[ht][j]               # get past n results
        h.append(past[num-1])                    # 0 index is most recent

        past = form.loc[at][j]               # get past n results.
        a.append(past[num-1])                   # 0 index is most recent

        if ((i + 1)% 10) == 0:
            j = j + 1

    playing_stat['HM' + str(num)] = h
    playing_stat['AM' + str(num)] = a


    return playing_stat


def add_form_df(playing_statistics):
    playing_statistics = add_form(playing_statistics,1)
    playing_statistics = add_form(playing_statistics,2)
    playing_statistics = add_form(playing_statistics,3)
    playing_statistics = add_form(playing_statistics,4)
    playing_statistics = add_form(playing_statistics,5)
    return playing_statistics

# Make changes to df
playing_statistics_1 = add_form_df(playing_statistics_1)
playing_statistics_2 = add_form_df(playing_statistics_2)
playing_statistics_3 = add_form_df(playing_statistics_3)
playing_statistics_4 = add_form_df(playing_statistics_4)
playing_statistics_5 = add_form_df(playing_statistics_5)
playing_statistics_6 = add_form_df(playing_statistics_6)
playing_statistics_7 = add_form_df(playing_statistics_7)
playing_statistics_8 = add_form_df(playing_statistics_8)
playing_statistics_9 = add_form_df(playing_statistics_9)
playing_statistics_10 = add_form_df(playing_statistics_10)
playing_statistics_11 = add_form_df(playing_statistics_11)
playing_statistics_12 = add_form_df(playing_statistics_12)
playing_statistics_13 = add_form_df(playing_statistics_13)
playing_statistics_14 = add_form_df(playing_statistics_14)
playing_statistics_15 = add_form_df(playing_statistics_15)
playing_statistics_16 = add_form_df(playing_statistics_16)
playing_statistics_17 = add_form_df(playing_statistics_17)
playing_statistics_18 = add_form_df(playing_statistics_18)


# In[ ]:


def get_form(playing_stat,num):
    form = get_matchres(playing_stat)
    form_final = form.copy()
    for i in range(num,39):
        form_final[i] = ''
        j = 0
        while j < num:
            form_final[i] += form[i-j]
            j += 1
    return form_final

def add_form(playing_stat,num):
    form = get_form(playing_stat,num)
    h = ['M' for i in range(num * 10)]  # since form is not available for n MW (n*10)
    a = ['M' for i in range(num * 10)]

    j = num
    for i in range((num*10),380):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam

        past = form.loc[ht][j]               # get past n results
        h.append(past[num-1])                    # 0 index is most recent

        past = form.loc[at][j]               # get past n results.
        a.append(past[num-1])                   # 0 index is most recent

        if ((i + 1)% 10) == 0:
            j = j + 1

    playing_stat['HM' + str(num)] = h
    playing_stat['AM' + str(num)] = a


    return playing_stat


def add_form_df(playing_statistics):
    playing_statistics = add_form(playing_statistics,1)
    playing_statistics = add_form(playing_statistics,2)
    playing_statistics = add_form(playing_statistics,3)
    playing_statistics = add_form(playing_statistics,4)
    playing_statistics = add_form(playing_statistics,5)
    return playing_statistics

# Make changes to df
playing_statistics_1 = add_form_df(playing_statistics_1)
playing_statistics_2 = add_form_df(playing_statistics_2)
playing_statistics_3 = add_form_df(playing_statistics_3)
playing_statistics_4 = add_form_df(playing_statistics_4)
playing_statistics_5 = add_form_df(playing_statistics_5)
playing_statistics_6 = add_form_df(playing_statistics_6)
playing_statistics_7 = add_form_df(playing_statistics_7)
playing_statistics_8 = add_form_df(playing_statistics_8)
playing_statistics_9 = add_form_df(playing_statistics_9)
playing_statistics_10 = add_form_df(playing_statistics_10)
playing_statistics_11 = add_form_df(playing_statistics_11)
playing_statistics_12 = add_form_df(playing_statistics_12)
playing_statistics_13 = add_form_df(playing_statistics_13)
playing_statistics_14 = add_form_df(playing_statistics_14)
playing_statistics_15 = add_form_df(playing_statistics_15)
playing_statistics_16 = add_form_df(playing_statistics_16)
playing_statistics_17 = add_form_df(playing_statistics_17)
playing_statistics_18 = add_form_df(playing_statistics_18)


# In[ ]:


def get_form(playing_stat,num):
    form = get_matchres(playing_stat)
    form_final = form.copy()
    for i in range(num,39):
        form_final[i] = ''
        j = 0
        while j < num:
            form_final[i] += form[i-j]
            j += 1
    return form_final

def add_form(playing_stat,num):
    form = get_form(playing_stat,num)
    h = ['M' for i in range(num * 10)]  # since form is not available for n MW (n*10)
    a = ['M' for i in range(num * 10)]

    j = num
    for i in range((num*10),380):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam

        past = form.loc[ht][j]               # get past n results
        h.append(past[num-1])                    # 0 index is most recent

        past = form.loc[at][j]               # get past n results.
        a.append(past[num-1])                   # 0 index is most recent

        if ((i + 1)% 10) == 0:
            j = j + 1

    playing_stat['HM' + str(num)] = h
    playing_stat['AM' + str(num)] = a


    return playing_stat


def add_form_df(playing_statistics):
    playing_statistics = add_form(playing_statistics,1)
    playing_statistics = add_form(playing_statistics,2)
    playing_statistics = add_form(playing_statistics,3)
    playing_statistics = add_form(playing_statistics,4)
    playing_statistics = add_form(playing_statistics,5)
    return playing_statistics

# Make changes to df
playing_statistics_1 = add_form_df(playing_statistics_1)
playing_statistics_2 = add_form_df(playing_statistics_2)
playing_statistics_3 = add_form_df(playing_statistics_3)
playing_statistics_4 = add_form_df(playing_statistics_4)
playing_statistics_5 = add_form_df(playing_statistics_5)
playing_statistics_6 = add_form_df(playing_statistics_6)
playing_statistics_7 = add_form_df(playing_statistics_7)
playing_statistics_8 = add_form_df(playing_statistics_8)
playing_statistics_9 = add_form_df(playing_statistics_9)
playing_statistics_10 = add_form_df(playing_statistics_10)
playing_statistics_11 = add_form_df(playing_statistics_11)
playing_statistics_12 = add_form_df(playing_statistics_12)
playing_statistics_13 = add_form_df(playing_statistics_13)
playing_statistics_14 = add_form_df(playing_statistics_14)
playing_statistics_15 = add_form_df(playing_statistics_15)
playing_statistics_16 = add_form_df(playing_statistics_16)
playing_statistics_17 = add_form_df(playing_statistics_17)
playing_statistics_18 = add_form_df(playing_statistics_18)


# In[ ]:


import pandas as pd

def rearrange_columns(df, cols):
    # Check if all required columns exist in the DataFrame
    missing_cols = [col for col in cols if col not in df.columns]
    if not missing_cols:
        return df[cols]
    else:
        print(f"Warning: Missing columns {missing_cols} in the DataFrame.")
        return df

# List of columns to rearrange
cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTGS', 'ATGS', 'HTGC', 'ATGC', 'HTP', 'ATP', 'HM1', 'HM2', 'HM3',
        'HM4', 'HM5', 'AM1', 'AM2', 'AM3', 'AM4', 'AM5' ]

# Apply rearrangement to each DataFrame
playing_statistics_1 = rearrange_columns(playing_statistics_1, cols)
playing_statistics_2 = rearrange_columns(playing_statistics_2, cols)
playing_statistics_3 = rearrange_columns(playing_statistics_3, cols)
playing_statistics_4 = rearrange_columns(playing_statistics_4, cols)
playing_statistics_5 = rearrange_columns(playing_statistics_5, cols)
playing_statistics_6 = rearrange_columns(playing_statistics_6, cols)
playing_statistics_7 = rearrange_columns(playing_statistics_7, cols)
playing_statistics_8 = rearrange_columns(playing_statistics_8, cols)
playing_statistics_9 = rearrange_columns(playing_statistics_9, cols)
playing_statistics_10 = rearrange_columns(playing_statistics_10, cols)
playing_statistics_11 = rearrange_columns(playing_statistics_11, cols)
playing_statistics_12 = rearrange_columns(playing_statistics_12, cols)
playing_statistics_13 = rearrange_columns(playing_statistics_13, cols)
playing_statistics_14 = rearrange_columns(playing_statistics_14, cols)
playing_statistics_15 = rearrange_columns(playing_statistics_15, cols)
playing_statistics_16 = rearrange_columns(playing_statistics_16, cols)
playing_statistics_17 = rearrange_columns(playing_statistics_17, cols)
playing_statistics_18 = rearrange_columns(playing_statistics_18, cols)


# In[ ]:


def get_mw(playing_stat):
    j = 1
    MatchWeek = []
    for i in range(380):
        MatchWeek.append(j)
        if ((i + 1)% 10) == 0:
            j = j + 1
    playing_stat['MW'] = MatchWeek
    return playing_stat

playing_statistics_1 = get_mw(playing_statistics_1)
playing_statistics_2 = get_mw(playing_statistics_2)
playing_statistics_3 = get_mw(playing_statistics_3)
playing_statistics_4 = get_mw(playing_statistics_4)
playing_statistics_5 = get_mw(playing_statistics_5)
playing_statistics_6 = get_mw(playing_statistics_6)
playing_statistics_7 = get_mw(playing_statistics_7)
playing_statistics_8 = get_mw(playing_statistics_8)
playing_statistics_9 = get_mw(playing_statistics_9)
playing_statistics_10 = get_mw(playing_statistics_10)
playing_statistics_11 = get_mw(playing_statistics_11)
playing_statistics_12 = get_mw(playing_statistics_12)
playing_statistics_13 = get_mw(playing_statistics_13)
playing_statistics_14 = get_mw(playing_statistics_14)
playing_statistics_15 = get_mw(playing_statistics_15)
playing_statistics_16 = get_mw(playing_statistics_16)
playing_statistics_17 = get_mw(playing_statistics_17)
playing_statistics_18 = get_mw(playing_statistics_18)


# In[ ]:


playing_stat = pd.concat([playing_statistics_1,
                          playing_statistics_2,
                          playing_statistics_3,
                          playing_statistics_4,
                          playing_statistics_5,
                          playing_statistics_6,
                          playing_statistics_7,
                          playing_statistics_8,
                          playing_statistics_9,
                          playing_statistics_10,
                          playing_statistics_11,
                          playing_statistics_12,
                          playing_statistics_13,
                          playing_statistics_14,
                          playing_statistics_15,
                          playing_statistics_16,
                          playing_statistics_17,
                          playing_statistics_18
                          ], ignore_index=True)


# Gets the form points.
def get_form_points(string):
    sum = 0
    for letter in string:
        sum += get_points(letter)
    return sum

playing_stat['HTFormPtsStr'] = playing_stat['HM1'] + playing_stat['HM2'] + playing_stat['HM3'] + playing_stat['HM4'] + playing_stat['HM5']
playing_stat['ATFormPtsStr'] = playing_stat['AM1'] + playing_stat['AM2'] + playing_stat['AM3'] + playing_stat['AM4'] + playing_stat['AM5']

playing_stat['HTFormPts'] = playing_stat['HTFormPtsStr'].apply(get_form_points)
playing_stat['ATFormPts'] = playing_stat['ATFormPtsStr'].apply(get_form_points)

# Identify Win/Loss Streaks if any.
def get_3game_ws(string):
    if string[-3:] == 'WWW':
        return 1
    else:
        return 0

def get_5game_ws(string):
    if string == 'WWWWW':
        return 1
    else:
        return 0

def get_3game_ls(string):
    if string[-3:] == 'LLL':
        return 1
    else:
        return 0

def get_5game_ls(string):
    if string == 'LLLLL':
        return 1
    else:
        return 0

playing_stat['HTWinStreak3'] = playing_stat['HTFormPtsStr'].apply(get_3game_ws)
playing_stat['HTWinStreak5'] = playing_stat['HTFormPtsStr'].apply(get_5game_ws)
playing_stat['HTLossStreak3'] = playing_stat['HTFormPtsStr'].apply(get_3game_ls)
playing_stat['HTLossStreak5'] = playing_stat['HTFormPtsStr'].apply(get_5game_ls)

playing_stat['ATWinStreak3'] = playing_stat['ATFormPtsStr'].apply(get_3game_ws)
playing_stat['ATWinStreak5'] = playing_stat['ATFormPtsStr'].apply(get_5game_ws)
playing_stat['ATLossStreak3'] = playing_stat['ATFormPtsStr'].apply(get_3game_ls)
playing_stat['ATLossStreak5'] = playing_stat['ATFormPtsStr'].apply(get_5game_ls)

playing_stat.keys()


# In[ ]:


def only_hw(string):
    if string == 'H':
        return 'H'
    else:
        return 'NH'

playing_stat['FTR'] = playing_stat.FTR.apply(only_hw)

# Testing set (2015-16 season)
playing_stat_test = playing_stat[5700:]

#saving the final dataset
playing_stat.to_csv('/content/Datasets/final_dataset.csv')

#saving the test set
playing_stat_test.to_csv("/content/Datasets/test.csv")



# In[ ]:


#loading the final dataset
dataset = pd.read_csv('/content/Datasets/final_dataset.csv')

dataset.head()



# In[ ]:


dataset.keys()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Assuming 'dataset' is your DataFrame

# Select only numeric columns for correlation matrix calculation
numeric_dataset = dataset.select_dtypes(include=[float, int])

# Alternatively, if you want to drop non-numeric columns
# numeric_dataset = dataset.drop(columns=['Date', 'HomeTeam', 'AwayTeam', 'FTR'])

# Compute the correlation matrix
correlation_matrix = numeric_dataset.corr()

# Set up the matplotlib figure
plt.figure(figsize=(20,10))

# Draw the heatmap
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)

# Show the plot
plt.show()


# In[ ]:


# Print all column names in the dataset to verify their existence
print(dataset.columns)

# Now safely drop columns if they exist
columns_to_drop = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG',
                   'HTGS', 'ATGS', 'HTGC', 'ATGC',
                   'HM4', 'HM5', 'AM4', 'AM5', 'MW', 'HTFormPtsStr',
                   'ATFormPtsStr', 'HTFormPts', 'ATFormPts', 'HTWinStreak3',
                   'HTWinStreak5', 'HTLossStreak3', 'HTLossStreak5', 'ATWinStreak3',
                   'ATWinStreak5', 'ATLossStreak3', 'ATLossStreak5', 'DiffPts']

# Drop only columns that exist in the dataset
columns_to_drop = [col for col in columns_to_drop if col in dataset.columns]
dataset2 = dataset.copy().drop(columns=columns_to_drop)

# Check the resulting dataframe
print(dataset2.head())


# In[ ]:


dataset2.keys()



# In[ ]:


dataset2.head(10)


# In[ ]:


#what is the win rate for the home team?

# Total number of matches.
n_matches = dataset2.shape[0]

# Calculate number of features. -1 because we are saving one as the target variable (win/lose/draw)
n_features = dataset2.shape[1] - 1

# Calculate matches won by home team.
n_homewins = len(dataset2[dataset2.FTR == 'H'])

# Calculate win rate for home team.
win_rate = (float(n_homewins) / (n_matches)) * 100

# Print the results
print("Total number of matches: {}".format(n_matches))
print ("Number of features: {}".format(n_features))
print( "Number of matches won by home team: {}".format(n_homewins))
print ("Win rate of home team: {:.2f}%".format(win_rate))


# In[ ]:


import pandas as pd
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split

# Assuming dataset2 is already defined and contains the data

# Split the dataset into features and target variable
X_all = dataset2.drop(['FTR'], axis=1)
y_all = dataset2['FTR']

# Check available columns in X_all
print("Columns in X_all:", X_all.columns)

# Define the columns you want to standardize
cols_to_scale = ['HTGD', 'ATGD', 'HTP', 'ATP']

# Verify if the columns are present
missing_cols = [col for col in cols_to_scale if col not in X_all.columns]
if missing_cols:
    print(f"The following columns are missing: {missing_cols}")
else:
    # Standardizing numerical features
    X_all[cols_to_scale] = scale(X_all[cols_to_scale])

    # Convert categorical features to dummy variables
    X_all = pd.get_dummies(X_all, columns=['HM1', 'HM2', 'HM3', 'AM1', 'AM2', 'AM3'], drop_first=True)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3, random_state=2, stratify=y_all)

    print("Shape of X_train:", X_train.shape)
    print("Shape of X_test:", X_test.shape)




# # Adjust according to actual column names found in your dataset
# if 'HTGD' not in dataset2.columns:
#     dataset2['HTGD'] = dataset2['GoalsFor_Home'] - dataset2['GoalsAgainst_Home']  # Home team goal difference
# if 'ATGD' not in dataset2.columns:
#     dataset2['ATGD'] = dataset2['GoalsFor_Away'] - dataset2['GoalsAgainst_Away']  # Away team goal difference
# if 'HTP' not in dataset2.columns:
#     dataset2['HTP'] = dataset2['Points_Home']  # Adjust based on the actual column name
# if 'ATP' not in dataset2.columns:
#     dataset2['ATP'] = dataset2['Points_Away']  # Adjust based on the actual column name
# 

# In[ ]:


import pandas as pd

# Check available columns in X_all before attempting to convert types
print("Columns in X_all:", X_all.columns)

# Check if 'HM1', 'HM2', 'HM3', 'AM1', 'AM2', 'AM3' exist in X_all
expected_cols = ['HM1', 'HM2', 'HM3', 'AM1', 'AM2', 'AM3']
missing_cols = [col for col in expected_cols if col not in X_all.columns]
if missing_cols:
    print(f"The following columns are missing: {missing_cols}")
else:
    # Convert last 3 wins columns to string type
    X_all['HM1'] = X_all['HM1'].astype('str')
    X_all['HM2'] = X_all['HM2'].astype('str')
    X_all['HM3'] = X_all['HM3'].astype('str')
    X_all['AM1'] = X_all['AM1'].astype('str')
    X_all['AM2'] = X_all['AM2'].astype('str')
    X_all['AM3'] = X_all['AM3'].astype('str')

    # Function to preprocess features
    def preprocess_features(X):
        ''' Preprocesses the football data and converts categorical variables into dummy variables. '''
        output = pd.DataFrame(index=X.index)
        for col, col_data in X.items():
            if col_data.dtype == object:
                col_data = pd.get_dummies(col_data, prefix=col)
            output = output.join(col_data)
        return output

    # Apply preprocessing
    X_all = preprocess_features(X_all)
    print("Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns)))


# In[ ]:


X_all.head(10)


# In[ ]:


from sklearn.model_selection import train_test_split

# Shuffle and split the dataset into training and testing set.
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all,
                                                    test_size = 0.3,
                                                    random_state = 2,
                                                    stratify = y_all)


# In[ ]:


from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Create an imputer object with a strategy (mean, median, most_frequent, etc.)
imputer = SimpleImputer(strategy='mean')

# Create a pipeline that first imputes missing values, then applies Logistic Regression
pipeline = Pipeline([
    ('imputer', imputer),
    ('classifier', LogisticRegression(random_state=0))
])

# Fit the pipeline on the training set
pipeline.fit(X_train, y_train)


# In[ ]:


from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Define the columns with missing values
missing_columns = ['HTP', 'ATP']

# Create a pipeline with an imputer and a logistic regression classifier
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with the mean
    ('classifier', LogisticRegression(random_state=0))
])

# Fit the pipeline on the training set
pipeline.fit(X_train, y_train)

# Optional: You can also transform the data to check the imputed values
X_train_imputed = pipeline.named_steps['imputer'].transform(X_train)
X_test_imputed = pipeline.named_steps['imputer'].transform(X_test)


# In[ ]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import scale
from sklearn.metrics import f1_score, make_scorer


# In[ ]:


# Extract relevant columns and concatenate data
columns_req = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
playing_statistics = [df[columns_req] for df in datasets]

# Function to calculate goals scored
def get_goals_scored(playing_stat):
    teams = {team: [] for team in playing_stat['HomeTeam'].unique()}
    for i in range(len(playing_stat)):
        HTGS = playing_stat.iloc[i]['FTHG']
        ATGS = playing_stat.iloc[i]['FTAG']
        teams[playing_stat.iloc[i].HomeTeam].append(HTGS)
        teams[playing_stat.iloc[i].AwayTeam].append(ATGS)
    GoalsScored = pd.DataFrame(data=teams, index=[i for i in range(1, 39)]).T
    GoalsScored[0] = 0
    for i in range(2, 39):
        GoalsScored[i] = GoalsScored[i] + GoalsScored[i-1]
    return GoalsScored

# Function to calculate goals conceded
def get_goals_conceded(playing_stat):
    teams = {team: [] for team in playing_stat['HomeTeam'].unique()}
    for i in range(len(playing_stat)):
        ATGC = playing_stat.iloc[i]['FTHG']
        HTGC = playing_stat.iloc[i]['FTAG']
        teams[playing_stat.iloc[i].HomeTeam].append(HTGC)
        teams[playing_stat.iloc[i].AwayTeam].append(ATGC)
    GoalsConceded = pd.DataFrame(data=teams, index=[i for i in range(1, 39)]).T
    GoalsConceded[0] = 0
    for i in range(2, 39):
        GoalsConceded[i] = GoalsConceded[i] + GoalsConceded[i-1]
    return GoalsConceded

# Function to get goals scored and conceded statistics
def get_gss(playing_stat):
    GC = get_goals_conceded(playing_stat)
    GS = get_goals_scored(playing_stat)
    HTGS, ATGS, HTGC, ATGC = [], [], [], []
    j = 0
    for i in range(len(playing_stat)):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        HTGS.append(GS.loc[ht][j])
        ATGS.append(GS.loc[at][j])
        HTGC.append(GC.loc[ht][j])
        ATGC.append(GC.loc[at][j])
        if ((i + 1) % 10) == 0:
            j += 1
    playing_stat['HTGS'] = HTGS
    playing_stat['ATGS'] = ATGS
    playing_stat['HTGC'] = HTGC
    playing_stat['ATGC'] = ATGC
    return playing_stat

# Apply the function to each dataset
playing_statistics = [get_gss(df) for df in playing_statistics]

# Function to calculate cumulative points
def get_points(result):
    if result == 'W':
        return 3
    elif result == 'D':
        return 1
    else:
        return 0

def get_cuml_points(matchres):
    matchres_points = matchres.applymap(get_points)
    for i in range(2, 39):
        matchres_points[i] = matchres_points[i] + matchres_points[i-1]
    matchres_points.insert(column=0, loc=0, value=[0] * len(matchres_points))
    return matchres_points

def get_matchres(playing_stat):
    teams = {team: [] for team in playing_stat['HomeTeam'].unique()}
    for i in range(len(playing_stat)):
        if playing_stat.iloc[i].FTR == 'H':
            teams[playing_stat.iloc[i].HomeTeam].append('W')
            teams[playing_stat.iloc[i].AwayTeam].append('L')
        elif playing_stat.iloc[i].FTR == 'A':
            teams[playing_stat.iloc[i].AwayTeam].append('W')
            teams[playing_stat.iloc[i].HomeTeam].append('L')
        else:
            teams[playing_stat.iloc[i].AwayTeam].append('D')
            teams[playing_stat.iloc[i].HomeTeam].append('D')
    return pd.DataFrame(data=teams, index=[i for i in range(1, 39)]).T

def get_agg_points(playing_stat):
    matchres = get_matchres(playing_stat)
    cum_pts = get_cuml_points(matchres)
    HTP, ATP = [], []
    j = 0
    for i in range(len(playing_stat)):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        HTP.append(cum_pts.loc[ht][j])
        ATP.append(cum_pts.loc[at][j])
        if ((i + 1) % 10) == 0:
            j += 1
    playing_stat['HTP'] = HTP
    playing_stat['ATP'] = ATP
    return playing_stat

# Apply the function to each dataset
playing_statistics = [get_agg_points(df) for df in playing_statistics]

# Function to get team form
def get_form(playing_stat, num):
    form = get_matchres(playing_stat)
    form_final = form.copy()
    for i in range(num, 39):
        form_final[i] = ''
        j = 0
        while j < num:
            form_final[i] += form[i-j]
            j += 1
    return form_final

def add_form(playing_stat, num):
    form = get_form(playing_stat, num)
    h = ['M' for _ in range(num * 10)]
    a = ['M' for _ in range(num * 10)]
    j = num
    for i in range(num * 10, len(playing_stat)):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        h.append(form.loc[ht][j][num-1])
        a.append(form.loc[at][j][num-1])
        if ((i + 1) % 10) == 0:
            j += 1
    playing_stat[f'HM{num}'] = h
    playing_stat[f'AM{num}'] = a
    return playing_stat

def add_form_df(playing_statistics):
    for num in range(1, 6):
        playing_statistics = [add_form(df, num) for df in playing_statistics]
    return playing_statistics

# Add form data to datasets
playing_statistics = add_form_df(playing_statistics)

# Rearrange columns
cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTGS', 'ATGS', 'HTGC', 'ATGC', 'HTP', 'ATP',
        'HM1', 'HM2', 'HM3', 'HM4', 'HM5', 'AM1', 'AM2', 'AM3', 'AM4', 'AM5']
playing_statistics = [df[cols] for df in playing_statistics]

# Function to add matchweek
def get_mw(playing_stat):
    j = 1
    MatchWeek = []
    for i in range(len(playing_stat)):
        MatchWeek.append(j)
        if ((i + 1) % 10) == 0:
            j += 1
    playing_stat['MW'] = MatchWeek
    return playing_stat

# Add matchweek to datasets
playing_statistics = [get_mw(df) for df in playing_statistics]

# Combine all data into a single DataFrame
playing_stat = pd.concat(playing_statistics, ignore_index=True)

# Add form points
playing_stat['HTFormPtsStr'] = playing_stat['HM1'] + playing_stat['HM2'] + playing_stat['HM3'] + playing_stat['HM4'] + playing_stat['HM5']
playing_stat['ATFormPtsStr'] = playing_stat['AM1'] + playing_stat['AM2'] + playing_stat['AM3'] + playing_stat['AM4'] + playing_stat['AM5']
playing_stat['HTFormPts'] = playing_stat['HTFormPtsStr'].apply(get_form_points)
playing_stat['ATFormPts'] = playing_stat['ATFormPtsStr'].apply(get_form_points)

# Identify Win/Loss Streaks
playing_stat['HTWinStreak3'] = playing_stat['HTFormPtsStr'].apply(get_3game_ws)
playing_stat['HTWinStreak5'] = playing_stat['HTFormPtsStr'].apply(get_5game_ws)
playing_stat['HTLossStreak3'] = playing_stat['HTFormPtsStr'].apply(get_3game_ls)
playing_stat['HTLossStreak5'] = playing_stat['HTFormPtsStr'].apply(get_5game_ls)
playing_stat['ATWinStreak3'] = playing_stat['ATFormPtsStr'].apply(get_3game_ws)
playing_stat['ATWinStreak5'] = playing_stat['ATFormPtsStr'].apply(get_5game_ws)
playing_stat['ATLossStreak3'] = playing_stat['ATFormPtsStr'].apply(get_3game_ls)
playing_stat['ATLossStreak5'] = playing_stat['ATFormPtsStr'].apply(get_5game_ls)

# Calculate Goal Difference
playing_stat['HTGD'] = playing_stat['HTGS'] - playing_stat['HTGC']
playing_stat['ATGD'] = playing_stat['ATGS'] - playing_stat['ATGC']

# Calculate DiffPts and DiffFormPts
playing_stat['DiffPts'] = playing_stat['HTP'] - playing_stat['ATP']
playing_stat['DiffFormPts'] = playing_stat['HTFormPts'] - playing_stat['ATFormPts']

# Scale features by Matchweek
playing_stat.MW = playing_stat.MW.astype(float)
cols = ['HTGD', 'ATGD', 'DiffPts', 'DiffFormPts', 'HTP', 'ATP']
for col in cols:
    playing_stat[col] = playing_stat[col] / playing_stat.MW

# Transform target variable
def only_hw(string):
    if string == 'H':
        return 'H'
    else:
        return 'NH'

playing_stat['FTR'] = playing_stat.FTR.apply(only_hw)

# Save final dataset
playing_stat.to_csv('/content/final_dataset.csv', index=False)

# Load the dataset
dataset = pd.read_csv('/content/final_dataset.csv')

# Drop columns to prevent multicollinearity
dataset2 = dataset.copy().drop(columns=['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HTGS', 'ATGS', 'HTGC', 'ATGC',
                                        'HM4', 'HM5', 'AM4', 'AM5', 'MW', 'HTFormPtsStr', 'ATFormPtsStr',
                                        'HTWinStreak3', 'HTWinStreak5', 'HTLossStreak3', 'HTLossStreak5',
                                        'ATWinStreak3', 'ATWinStreak5', 'ATLossStreak3', 'ATLossStreak5',
                                        'DiffPts'])

# Split the dataset into features and target variable
X_all = dataset2.drop(['FTR'], axis=1)
y_all = dataset2['FTR']

# Standardizing numerical features
cols = [['HTGD', 'ATGD', 'HTP', 'ATP']]
for col in cols:
    X_all[col] = scale(X_all[col])

# Convert categorical features to dummy variables
X_all = pd.get_dummies(X_all, columns=['HM1', 'HM2', 'HM3', 'AM1', 'AM2', 'AM3'], drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3, random_state=2, stratify=y_all)

# Function to evaluate model performance
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.show()
    print(classification_report(y_test, y_pred))


# In[ ]:


from sklearn.preprocessing import LabelEncoder

# Encode target labels with value between 0 and n_classes-1
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Logistic Regression
lr_model = LogisticRegression(random_state=0)
lr_model.fit(X_train, y_train_encoded)
evaluate_model(lr_model, X_train, y_train_encoded)

# SVM
svm_model = SVC(kernel='rbf', random_state=0)
svm_model.fit(X_train, y_train_encoded)
evaluate_model(svm_model, X_train, y_train_encoded)

# Random Forest
rf_model = RandomForestClassifier(criterion='gini', n_estimators=700, min_samples_split=10, min_samples_leaf=1,
                                  max_features='sqrt', oob_score=True, random_state=1, n_jobs=-1)
rf_model.fit(X_train, y_train_encoded)
evaluate_model(rf_model, X_train, y_train_encoded)

# XGBoost
xgb_model = XGBClassifier(seed=82)
xgb_model.fit(X_train, y_train_encoded)
evaluate_model(xgb_model,X_train, y_train_encoded)

# Hyperparameter Tuning for XGBoost
parameters = {
    'learning_rate': [0.1],
    'n_estimators': [40],
    'max_depth': [3],
    'min_child_weight': [3],
    'gamma': [0.4],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
    'scale_pos_weight': [1],
    'reg_alpha': [1e-5]
}

f1_scorer = make_scorer(f1_score, pos_label=1)
grid_obj = GridSearchCV(XGBClassifier(seed=2), param_grid=parameters, scoring=f1_scorer, cv=5)
grid_obj = grid_obj.fit(X_train, y_train_encoded)
best_xgb = grid_obj.best_estimator_
print("Best XGBoost model:", best_xgb)

# Evaluate the best model
evaluate_model(best_xgb, X_test, y_test_encoded)


# In[ ]:


# Extract relevant columns and concatenate data
columns_req = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
playing_statistics = [df[columns_req] for df in datasets]

# Function to calculate goals scored
def get_goals_scored(playing_stat):
    teams = {team: [] for team in playing_stat['HomeTeam'].unique()}
    for i in range(len(playing_stat)):
        HTGS = playing_stat.iloc[i]['FTHG']
        ATGS = playing_stat.iloc[i]['FTAG']
        teams[playing_stat.iloc[i].HomeTeam].append(HTGS)
        teams[playing_stat.iloc[i].AwayTeam].append(ATGS)
    GoalsScored = pd.DataFrame(data=teams, index=[i for i in range(1, 39)]).T
    GoalsScored[0] = 0
    for i in range(2, 39):
        GoalsScored[i] = GoalsScored[i] + GoalsScored[i-1]
    return GoalsScored

# Function to calculate goals conceded
def get_goals_conceded(playing_stat):
    teams = {team: [] for team in playing_stat['HomeTeam'].unique()}
    for i in range(len(playing_stat)):
        ATGC = playing_stat.iloc[i]['FTHG']
        HTGC = playing_stat.iloc[i]['FTAG']
        teams[playing_stat.iloc[i].HomeTeam].append(HTGC)
        teams[playing_stat.iloc[i].AwayTeam].append(ATGC)
    GoalsConceded = pd.DataFrame(data=teams, index=[i for i in range(1, 39)]).T
    GoalsConceded[0] = 0
    for i in range(2, 39):
        GoalsConceded[i] = GoalsConceded[i] + GoalsConceded[i-1]
    return GoalsConceded

# Function to get goals scored and conceded statistics
def get_gss(playing_stat):
    GC = get_goals_conceded(playing_stat)
    GS = get_goals_scored(playing_stat)
    HTGS, ATGS, HTGC, ATGC = [], [], [], []
    j = 0
    for i in range(len(playing_stat)):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        HTGS.append(GS.loc[ht][j])
        ATGS.append(GS.loc[at][j])
        HTGC.append(GC.loc[ht][j])
        ATGC.append(GC.loc[at][j])
        if ((i + 1) % 10) == 0:
            j += 1
    playing_stat['HTGS'] = HTGS
    playing_stat['ATGS'] = ATGS
    playing_stat['HTGC'] = HTGC
    playing_stat['ATGC'] = ATGC
    return playing_stat

# Apply the function to each dataset
playing_statistics = [get_gss(df) for df in playing_statistics]

# Function to calculate cumulative points
def get_points(result):
    if result == 'W':
        return 3
    elif result == 'D':
        return 1
    else:
        return 0

def get_cuml_points(matchres):
    matchres_points = matchres.applymap(get_points)
    for i in range(2, 39):
        matchres_points[i] = matchres_points[i] + matchres_points[i-1]
    matchres_points.insert(column=0, loc=0, value=[0] * len(matchres_points))
    return matchres_points

def get_matchres(playing_stat):
    teams = {team: [] for team in playing_stat['HomeTeam'].unique()}
    for i in range(len(playing_stat)):
        if playing_stat.iloc[i].FTR == 'H':
            teams[playing_stat.iloc[i].HomeTeam].append('W')
            teams[playing_stat.iloc[i].AwayTeam].append('L')
        elif playing_stat.iloc[i].FTR == 'A':
            teams[playing_stat.iloc[i].AwayTeam].append('W')
            teams[playing_stat.iloc[i].HomeTeam].append('L')
        else:
            teams[playing_stat.iloc[i].AwayTeam].append('D')
            teams[playing_stat.iloc[i].HomeTeam].append('D')
    return pd.DataFrame(data=teams, index=[i for i in range(1, 39)]).T

def get_agg_points(playing_stat):
    matchres = get_matchres(playing_stat)
    cum_pts = get_cuml_points(matchres)
    HTP, ATP = [], []
    j = 0
    for i in range(len(playing_stat)):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        HTP.append(cum_pts.loc[ht][j])
        ATP.append(cum_pts.loc[at][j])
        if ((i + 1) % 10) == 0:
            j += 1
    playing_stat['HTP'] = HTP
    playing_stat['ATP'] = ATP
    return playing_stat

# Apply the function to each dataset
playing_statistics = [get_agg_points(df) for df in playing_statistics]

# Function to get team form
def get_form(playing_stat, num):
    form = get_matchres(playing_stat)
    form_final = form.copy()
    for i in range(num, 39):
        form_final[i] = ''
        j = 0
        while j < num:
            form_final[i] += form[i-j]
            j += 1
    return form_final

def add_form(playing_stat, num):
    form = get_form(playing_stat, num)
    h = ['M' for _ in range(num * 10)]
    a = ['M' for _ in range(num * 10)]
    j = num
    for i in range(num * 10, len(playing_stat)):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        h.append(form.loc[ht][j][num-1])
        a.append(form.loc[at][j][num-1])
        if ((i + 1) % 10) == 0:
            j += 1
    playing_stat[f'HM{num}'] = h
    playing_stat[f'AM{num}'] = a
    return playing_stat

def add_form_df(playing_statistics):
    for num in range(1, 6):
        playing_statistics = [add_form(df, num) for df in playing_statistics]
    return playing_statistics

# Add form data to datasets
playing_statistics = add_form_df(playing_statistics)

# Rearrange columns
cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTGS', 'ATGS', 'HTGC', 'ATGC', 'HTP', 'ATP',
        'HM1', 'HM2', 'HM3', 'HM4', 'HM5', 'AM1', 'AM2', 'AM3', 'AM4', 'AM5']
playing_statistics = [df[cols] for df in playing_statistics]

# Function to add matchweek
def get_mw(playing_stat):
    j = 1
    MatchWeek = []
    for i in range(len(playing_stat)):
        MatchWeek.append(j)
        if ((i + 1) % 10) == 0:
            j += 1
    playing_stat['MW'] = MatchWeek
    return playing_stat

# Add matchweek to datasets
playing_statistics = [get_mw(df) for df in playing_statistics]

# Combine all data into a single DataFrame
playing_stat = pd.concat(playing_statistics, ignore_index=True)

# Add form points
playing_stat['HTFormPtsStr'] = playing_stat['HM1'] + playing_stat['HM2'] + playing_stat['HM3'] + playing_stat['HM4'] + playing_stat['HM5']
playing_stat['ATFormPtsStr'] = playing_stat['AM1'] + playing_stat['AM2'] + playing_stat['AM3'] + playing_stat['AM4'] + playing_stat['AM5']
playing_stat['HTFormPts'] = playing_stat['HTFormPtsStr'].apply(get_form_points)
playing_stat['ATFormPts'] = playing_stat['ATFormPtsStr'].apply(get_form_points)

# Identify Win/Loss Streaks
playing_stat['HTWinStreak3'] = playing_stat['HTFormPtsStr'].apply(get_3game_ws)
playing_stat['HTWinStreak5'] = playing_stat['HTFormPtsStr'].apply(get_5game_ws)
playing_stat['HTLossStreak3'] = playing_stat['HTFormPtsStr'].apply(get_3game_ls)
playing_stat['HTLossStreak5'] = playing_stat['HTFormPtsStr'].apply(get_5game_ls)
playing_stat['ATWinStreak3'] = playing_stat['ATFormPtsStr'].apply(get_3game_ws)
playing_stat['ATWinStreak5'] = playing_stat['ATFormPtsStr'].apply(get_5game_ws)
playing_stat['ATLossStreak3'] = playing_stat['ATFormPtsStr'].apply(get_3game_ls)
playing_stat['ATLossStreak5'] = playing_stat['ATFormPtsStr'].apply(get_5game_ls)

# Calculate Goal Difference
playing_stat['HTGD'] = playing_stat['HTGS'] - playing_stat['HTGC']
playing_stat['ATGD'] = playing_stat['ATGS'] - playing_stat['ATGC']

# Calculate DiffPts and DiffFormPts
playing_stat['DiffPts'] = playing_stat['HTP'] - playing_stat['ATP']
playing_stat['DiffFormPts'] = playing_stat['HTFormPts'] - playing_stat['ATFormPts']

# Scale features by Matchweek
playing_stat.MW = playing_stat.MW.astype(float)
cols = ['HTGD', 'ATGD', 'DiffPts', 'DiffFormPts', 'HTP', 'ATP']
for col in cols:
    playing_stat[col] = playing_stat[col] / playing_stat.MW

# Transform target variable
def only_hw(string):
    if string == 'H':
        return 'H'
    else:
        return 'NH'

playing_stat['FTR'] = playing_stat.FTR.apply(only_hw)

# Save final dataset
playing_stat.to_csv('/content/final_dataset.csv', index=False)

# Load the dataset
dataset = pd.read_csv('/content/final_dataset.csv')

# Drop columns to prevent multicollinearity
dataset2 = dataset.copy().drop(columns=['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HTGS', 'ATGS', 'HTGC', 'ATGC',
                                        'HM4', 'HM5', 'AM4', 'AM5', 'MW', 'HTFormPtsStr', 'ATFormPtsStr',
                                        'HTWinStreak3', 'HTWinStreak5', 'HTLossStreak3', 'HTLossStreak5',
                                        'ATWinStreak3', 'ATWinStreak5', 'ATLossStreak3', 'ATLossStreak5',
                                        'DiffPts'])

# Split the dataset into features and target variable
X_all = dataset2.drop(['FTR'], axis=1)
y_all = dataset2['FTR']

# Standardizing numerical features
cols = [['HTGD', 'ATGD', 'HTP', 'ATP']]
for col in cols:
    X_all[col] = scale(X_all[col])

# Convert categorical features to dummy variables
X_all = pd.get_dummies(X_all, columns=['HM1', 'HM2', 'HM3', 'AM1', 'AM2', 'AM3'], drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3, random_state=2, stratify=y_all)

# Function to evaluate model performance
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.show()
    print(classification_report(y_test, y_pred))


# In[ ]:


from sklearn.preprocessing import LabelEncoder

# Encode target labels with value between 0 and n_classes-1
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Logistic Regression
lr_model = LogisticRegression(random_state=0)
lr_model.fit(X_train, y_train_encoded)
evaluate_model(lr_model, X_test, y_test_encoded)

# SVM
svm_model = SVC(kernel='rbf', random_state=0)
svm_model.fit(X_train, y_train_encoded)
evaluate_model(svm_model, X_test, y_test_encoded)

# Random Forest
rf_model = RandomForestClassifier(criterion='gini', n_estimators=700, min_samples_split=10, min_samples_leaf=1,
                                  max_features='sqrt', oob_score=True, random_state=1, n_jobs=-1)
rf_model.fit(X_train, y_train_encoded)
evaluate_model(rf_model, X_test, y_test_encoded)

# XGBoost
xgb_model = XGBClassifier(seed=82)
xgb_model.fit(X_train, y_train_encoded)
evaluate_model(xgb_model, X_test, y_test_encoded)

# Hyperparameter Tuning for XGBoost
parameters = {
    'learning_rate': [0.1],
    'n_estimators': [40],
    'max_depth': [3],
    'min_child_weight': [3],
    'gamma': [0.4],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
    'scale_pos_weight': [1],
    'reg_alpha': [1e-5]
}

f1_scorer = make_scorer(f1_score, pos_label=1)
grid_obj = GridSearchCV(XGBClassifier(seed=2), param_grid=parameters, scoring=f1_scorer, cv=5)
grid_obj = grid_obj.fit(X_train, y_train_encoded)
best_xgb = grid_obj.best_estimator_
print("Best XGBoost model:", best_xgb)

# Evaluate the best model
evaluate_model(best_xgb, X_test, y_test_encoded)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error

# Function to evaluate model performance with additional metrics
def evaluate_model_with_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.show()
    print(classification_report(y_test, y_pred))

    # Calculate additional metrics
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape}")
    print(f"R-squared (R²): {r2}")

# Encode target labels with value between 0 and n_classes-1
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Logistic Regression
lr_model = LogisticRegression(random_state=0)
lr_model.fit(X_train, y_train_encoded)
evaluate_model_with_metrics(lr_model, X_test, y_test_encoded)

# SVM
svm_model = SVC(kernel='rbf', random_state=0)
svm_model.fit(X_train, y_train_encoded)
evaluate_model_with_metrics(svm_model, X_test, y_test_encoded)

# Random Forest
rf_model = RandomForestClassifier(criterion='gini', n_estimators=700, min_samples_split=10, min_samples_leaf=1,
                                  max_features='sqrt', oob_score=True, random_state=1, n_jobs=-1)
rf_model.fit(X_train, y_train_encoded)
evaluate_model_with_metrics(rf_model, X_test, y_test_encoded)

# XGBoost
xgb_model = XGBClassifier(seed=82)
xgb_model.fit(X_train, y_train_encoded)
evaluate_model_with_metrics(xgb_model, X_test, y_test_encoded)

# Gradient Boosting Machines (GBM)
gbm_model = GradientBoostingClassifier(random_state=0)
gbm_model.fit(X_train, y_train_encoded)
evaluate_model_with_metrics(gbm_model, X_test, y_test_encoded)

# Hyperparameter Tuning for XGBoost
parameters = {
    'learning_rate': [0.1],
    'n_estimators': [40],
    'max_depth': [3],
    'min_child_weight': [3],
    'gamma': [0.4],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
    'scale_pos_weight': [1],
    'reg_alpha': [1e-5]
}

f1_scorer = make_scorer(f1_score, pos_label=1)
grid_obj = GridSearchCV(XGBClassifier(seed=2), param_grid=parameters, scoring=f1_scorer, cv=5)
grid_obj = grid_obj.fit(X_train, y_train_encoded)
best_xgb = grid_obj.best_estimator_
print("Best XGBoost model:", best_xgb)

# Evaluate the best XGBoost model
evaluate_model_with_metrics(best_xgb, X_test, y_test_encoded)


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# Define the parameter grid for each model with reduced search space
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [10, None],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

param_grid_gbm = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 4],
    'subsample': [0.8, 1.0],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

param_grid_xgb = {
    'learning_rate': [0.1, 0.05],
    'n_estimators': [40, 100],
    'max_depth': [3, 4],
    'min_child_weight': [1, 3],
    'gamma': [0.1, 0.3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_alpha': [1e-5, 1e-2]
}

# Hyperparameter tuning for Random Forest using RandomizedSearchCV
grid_rf = RandomizedSearchCV(RandomForestClassifier(random_state=1), param_distributions=param_grid_rf,
                             scoring=f1_scorer, cv=3, n_iter=10, n_jobs=-1, random_state=1)
grid_rf.fit(X_train, y_train_encoded)
best_rf = grid_rf.best_estimator_
print("Best Random Forest model:", best_rf)
evaluate_model_with_metrics(best_rf, X_test, y_test_encoded)

# Hyperparameter tuning for Gradient Boosting Machines (GBM) using RandomizedSearchCV
grid_gbm = RandomizedSearchCV(GradientBoostingClassifier(random_state=0), param_distributions=param_grid_gbm,
                              scoring=f1_scorer, cv=3, n_iter=10, n_jobs=-1, random_state=1)
grid_gbm.fit(X_train, y_train_encoded)
best_gbm = grid_gbm.best_estimator_
print("Best GBM model:", best_gbm)
evaluate_model_with_metrics(best_gbm, X_test, y_test_encoded)

# Hyperparameter tuning for XGBoost using RandomizedSearchCV
grid_xgb = RandomizedSearchCV(XGBClassifier(seed=2), param_distributions=param_grid_xgb,
                              scoring=f1_scorer, cv=3, n_iter=10, n_jobs=-1, random_state=1)
grid_xgb.fit(X_train, y_train_encoded)
best_xgb = grid_xgb.best_estimator_
print("Best XGBoost model:", best_xgb)
evaluate_model_with_metrics(best_xgb, X_test, y_test_encoded)

