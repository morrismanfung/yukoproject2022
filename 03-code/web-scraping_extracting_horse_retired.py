# A new version of script to extract information from html files.

import numpy as np
import pandas as pd
import os
from datetime import date
import re
import warnings

warnings.filterwarnings('ignore')

def HorseBasics( HorseBasics_html_file_name):
    target_file_name = HorseBasics_html_file_name
    with open( target_file_name, 'r', encoding = 'utf-8_sig') as html_file:
        html_text_HorseBasics = html_file.read()
    try:
        df_horse_basic = pd.read_html( html_text_HorseBasics, flavor = 'lxml', displayed_only = False)
        HorseName = df_horse_basic[1][0][0]
        HorseName = re.sub( '\(.*\)', '', HorseName).strip()
        Origin = df_horse_basic[2][2][0]
        Age = 7
        YearOfBirth = date.today().year - int( Age)
        Color = df_horse_basic[2][2][1].split( ' / ')[0]
        Sex = df_horse_basic[2][2][1].split( ' / ')[1]
        SeasonalStakes = 'Not Available'
        TotalStakes = df_horse_basic[2][2][3]
        No_1st = df_horse_basic[2][2][4].split('-')[0]
        No_2nd = df_horse_basic[2][2][4].split('-')[1]
        No_3rd = df_horse_basic[2][2][4].split('-')[2]
        No_TotalRaces = df_horse_basic[2][2][4].split('-')[3]
        Trainer = 'Not Available'
        LastRating = df_horse_basic[3][2][1]
        StartOfTheSeasonRating = 'NotAvailable'

        for i in range( len( df_horse_basic)):
            if df_horse_basic[i].iloc[ 0, 0] == '場次':
                tableRace = df_horse_basic[ i]

        # tableRace's columns
        # 0: 場次, 1: 名次, 2: 日期, 3: 馬場/跑道/賽道, 4: 途程, 5: 場地狀況, 6: 賽事班次, 7: 檔位, 8: 評分, 9: 練馬師, 10: 騎師,
        # 11: 頭馬距離, 12: 獨贏賠率, 13: 實際負磅, 14: 沿途走位, 15: 完成時間, 16: 排位體重, 17: 配備, 18: 賽事重播

        tableRace = tableRace[ [1, 2, 8, 16]]
        tableRace = tableRace.dropna()
        tableRace[ 2] = tableRace[ 2].astype( 'str')
        tableRace = tableRace[ ~tableRace[ 2].str.contains( '馬季')]
        tableRace.columns = [ 'RPlace', 'RDate', 'RRating', 'RWeight']
        tableRace.head()
        tableRace = tableRace.iloc[ 1:, ]

        tableRace[ 'RPlace'] = pd.to_numeric( tableRace[ 'RPlace'], errors = 'coerce')

        tmp = pd.DataFrame()
        tmp[ 'RDate'] = tableRace[ 'RDate']
        tmp[ 'RDate'] = pd.to_datetime( tmp[ 'RDate'], format = '%d/%m/%Y')
        tmp[ 'LastRace_Date'] = tmp[ 'RDate'].shift( -1)
        tmp[ 'LastRace_nDays'] = tmp[ 'RDate'] - tmp[ 'LastRace_Date']
        tmp[ 'LastRace_nDays'] = pd.to_numeric( tmp[ 'LastRace_nDays'] ) / 24 / 60 / 60 / (10 ** 9)
        tmp.loc[ tmp[ 'LastRace_nDays'] < 0, 'LastRace_nDays'] = 36

        tmp[ 'RPlace'] = tableRace[ 'RPlace']
        tmp[ 'PreviousPlace'] = tmp[ 'RPlace'].shift( -1)
        tmp[ 'PreviousPlace'] = tmp[ 'PreviousPlace'].fillna( 7)
        for i in tmp.index:
            df = tmp.loc[ i:, ].head( 3)
            mean_place = df[ 'PreviousPlace'].mean()
            tmp.loc[ i, 'AvgPlace3'] = mean_place

        for i in tmp.index:
            df = tmp.loc[ i:, ].head( 5)
            mean_place = df[ 'PreviousPlace'].mean()
            tmp.loc[ i, 'AvgPlace5'] = mean_place

        tmp[ 'RRating'] = tableRace[ 'RRating']

        tmp[ 'Weight'] = tableRace[ 'RWeight']
        tmp[ 'OldWeight'] = tmp[ 'Weight'].shift( -1)
        tmp[ 'WeightDiff'] = pd.Series( dtype = 'float64')
        tmp = tmp.replace( '--', 0)
        tmp = tmp.fillna( 0)

        tmp[ [ 'Weight', 'OldWeight']] = tmp[ [ 'Weight', 'OldWeight']].astype( 'float64')

        for i in tmp.index:
            if (tmp.loc[ i, 'Weight'] == 0) |  (tmp.loc[ i, 'OldWeight'] == 0):
                tmp.loc[ i, 'WeightDiff'] = 0
            else:
                tmp.loc[ i, 'WeightDiff'] = tmp.loc[ i, 'Weight'] - tmp.loc[ i, 'OldWeight']

        basic_info = [ HorseName, Origin, Age, YearOfBirth, Color, Sex, SeasonalStakes, TotalStakes, No_1st, No_2nd, No_3rd, No_TotalRaces, Trainer, LastRating, StartOfTheSeasonRating]

        tmp.to_csv( f'{HorseName}.csv', encoding = 'utf-8_sig')
        
        return( basic_info)
    except:
        return( 'HorseNotExist')
    

HorseBasicsInfo = []
all_horsebasics_html = os.listdir( 'HorsesBasics_data_html_retired')

for horse_basics_file_name in all_horsebasics_html[ :]:
    target_file_name = 'HorsesBasics_data_html_retired\\' + horse_basics_file_name
    horse_output = HorseBasics( target_file_name)
    if horse_output == 'HorseNotExist':
        continue
    else:
        HorseBasicsInfo.append( horse_output)
    print( f'{horse_basics_file_name} done')

HorseBasicsInfo = pd.DataFrame( HorseBasicsInfo)
HorseBasicsInfo.columns = [ 'HorseName', 'HorseOrigin', 'Age', 'YearOfBirth', 'Color', 'Sex', 'H_SeasonalStake', 'H_TotalStake', 'H_1st', 'H_2nd', 'H_3rd', 'H_Total', 'H_Trainer', 'CurrentRating', 'StartofTheSeasonRating']
HorseBasicsInfo.to_csv( 'HorseBasicsInformation_retired.csv', encoding = 'utf-8_sig')

# ---------- Testing ----------



# print( basic_info)

horse_basics_file_name = all_horsebasics_html[ 133]
target_file_name = 'HorsesBasics_data_html_retired\\' + horse_basics_file_name
with open( target_file_name, 'r', encoding = 'utf-8_sig') as html_file:
    html_text_HorseBasics = html_file.read()

df_horse_basic = pd.read_html( html_text_HorseBasics, flavor = 'lxml', displayed_only = False)
HorseName = df_horse_basic[1][0][0]
HorseName = re.sub( '\(.*\)', '', HorseName).strip()
Origin = df_horse_basic[2][2][0]
Age = 7
YearOfBirth = date.today().year - int( Age)
Color = df_horse_basic[2][2][1].split( ' / ')[0]
Sex = df_horse_basic[2][2][1].split( ' / ')[1]
SeasonalStakes = 'Not Available'
TotalStakes = df_horse_basic[2][2][3]
No_1st = df_horse_basic[2][2][4].split('-')[0]
No_2nd = df_horse_basic[2][2][4].split('-')[1]
No_3rd = df_horse_basic[2][2][4].split('-')[2]
No_TotalRaces = df_horse_basic[2][2][4].split('-')[3]
Trainer = 'Not Available'
LastRating = df_horse_basic[3][2][1]
StartOfTheSeasonRating = 'NotAvailable'

for i in range( len( df_horse_basic)):
    if df_horse_basic[i].iloc[ 0, 0] == '場次':
        tableRace = df_horse_basic[ i]

# tableRace's columns
# 0: 場次, 1: 名次, 2: 日期, 3: 馬場/跑道/賽道, 4: 途程, 5: 場地狀況, 6: 賽事班次, 7: 檔位, 8: 評分, 9: 練馬師, 10: 騎師,
# 11: 頭馬距離, 12: 獨贏賠率, 13: 實際負磅, 14: 沿途走位, 15: 完成時間, 16: 排位體重, 17: 配備, 18: 賽事重播

tableRace = tableRace[ [1, 2, 8, 16]]
tableRace = tableRace.dropna()
tableRace[ 2] = tableRace[ 2].astype( 'str')
tableRace = tableRace[ ~tableRace[ 2].str.contains( '馬季')]
tableRace.columns = [ 'RPlace', 'RDate', 'RRating', 'RWeight']
tableRace.head()
tableRace = tableRace.iloc[ 1:, ]

tableRace[ 'RPlace'] = pd.to_numeric( tableRace[ 'RPlace'], errors = 'coerce')

tmp = pd.DataFrame()
tmp[ 'RDate'] = tableRace[ 'RDate']
tmp[ 'RDate'] = pd.to_datetime( tmp[ 'RDate'], format = '%d/%m/%Y')
tmp[ 'LastRace_Date'] = tmp[ 'RDate'].shift( -1)
tmp[ 'LastRace_nDays'] = tmp[ 'RDate'] - tmp[ 'LastRace_Date']
tmp[ 'LastRace_nDays'] = pd.to_numeric( tmp[ 'LastRace_nDays'] ) / 24 / 60 / 60 / (10 ** 9)
tmp.loc[ tmp[ 'LastRace_nDays'] < 0, 'LastRace_nDays'] = 36

tmp[ 'RPlace'] = tableRace[ 'RPlace']
tmp[ 'PreviousPlace'] = tmp[ 'RPlace'].shift( -1)
tmp[ 'PreviousPlace'] = tmp[ 'PreviousPlace'].fillna( 7)
for i in tmp.index:
    df = tmp.loc[ i:, ].head( 3)
    mean_place = df[ 'PreviousPlace'].mean()
    tmp.loc[ i, 'AvgPlace3'] = mean_place

for i in tmp.index:
    df = tmp.loc[ i:, ].head( 5)
    mean_place = df[ 'PreviousPlace'].mean()
    tmp.loc[ i, 'AvgPlace5'] = mean_place

tmp[ 'RRating'] = tableRace[ 'RRating']

tmp[ 'Weight'] = tableRace[ 'RWeight']
tmp[ 'OldWeight'] = tmp[ 'Weight'].shift( -1)
tmp[ 'WeightDiff'] = pd.Series( dtype = 'float64')
tmp = tmp.replace( '--', 0)
tmp = tmp.fillna( 0)

tmp[ [ 'Weight', 'OldWeight']] = tmp[ [ 'Weight', 'OldWeight']].astype( 'float64')

for i in tmp.index:
    if (tmp.loc[ i, 'Weight'] == 0) |  (tmp.loc[ i, 'OldWeight'] == 0):
        tmp.loc[ i, 'WeightDiff'] = 0
    else:
        tmp.loc[ i, 'WeightDiff'] = tmp.loc[ i, 'Weight'] - tmp.loc[ i, 'OldWeight']

basic_info = [ HorseName, Origin, Age, YearOfBirth, Color, Sex, SeasonalStakes, TotalStakes, No_1st, No_2nd, No_3rd, No_TotalRaces, Trainer, LastRating, StartOfTheSeasonRating]

tmp.to_csv( f'{HorseName}.csv', encoding = 'utf-8_sig')