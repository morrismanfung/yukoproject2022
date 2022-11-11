# A new version of script to extract information from html files.

import numpy as np
import pandas as pd
import os

def RaceSingleMatch( race_html_file_name):
    with open( race_html_file_name, 'r', encoding = 'utf-8_sig') as html_file:
        html_text_RaceSingle = html_file.read()
    # print( html_text_RaceSingle)
    try:
        pd_html = pd.read_html( html_text_RaceSingle)
        Date = race_html_file_name.split( '_')[ 2].split( '\\')[ 1]
        RaceNo = race_html_file_name.split( '_')[ 3]
        Venue = pd_html[0][0][0][:-1] # Venue
        RaceName = pd_html[1].iloc[1, 0].split( ' - ')[0] # RaceName - Distance
        Distance = pd_html[1].iloc[1, 0].split( ' - ')[1] # RaceName - Distance
        VenueCondition = pd_html[1].iloc[1, 3] # Veune Condition
        RaceStakes = pd_html[1].iloc[3, 0] # Stakes

        RaceSingleInfo = pd_html[2]
        RaceSingleInfo = RaceSingleInfo.fillna( 0)
        RaceSingleInfo.loc[ :,'名次'] = RaceSingleInfo.loc[ :,'名次'].astype( 'int')
        RaceSingleInfo.loc[ :,'馬號'] = RaceSingleInfo.loc[ :,'馬號'].astype( 'int')
        RaceSingleInfo = RaceSingleInfo[ RaceSingleInfo[ '馬號'].isin( [ int( i) for i in range(1, 15)])]
        nHorse = RaceSingleInfo.shape[0]
        WinnerNo = RaceSingleInfo.iloc[ 0, 1]
        SecondNo = RaceSingleInfo.iloc[ 1, 1]
        ThirdNo = RaceSingleInfo.iloc[ 2, 1]
        ForthNo = RaceSingleInfo.iloc[ 3, 1]

        RaceSingleInfo = RaceSingleInfo.sort_values( by = '馬號', axis = 0)
        RaceSingleInfo[ ['Date', 'RaceNo', 'Venue', 'Class', 'Distance', 'VenueCondition', 'RaceStakes', 'WinnerNo', 'nHorse']] = \
        [ Date, RaceNo, Venue, RaceName, Distance, VenueCondition, RaceStakes, WinnerNo, nHorse]
        RaceSingleInfo[ 'Winner'] = RaceSingleInfo[ '名次'] == 1
        RaceSingleInfo[ 'PLACE'] = RaceSingleInfo[ '名次'] < 4
        RaceSingle_output = RaceSingleInfo
    except:
        return 'RaceNotExist'

    return RaceSingle_output

RaceInfo = pd.DataFrame()

all_races_html = os.listdir( '01-data\\01-historical\\Races_data_html')

for single_race in all_races_html[ :]:
    target_file_name = '01-data\\01-historical\\Races_data_html\\' + single_race
    single_race_output = RaceSingleMatch( target_file_name)
    if type( single_race_output) == str:
        continue
    else:
        RaceInfo = pd.concat( [RaceInfo, single_race_output], axis = 0)
    print( f'{single_race} done')

pd.DataFrame( RaceInfo).to_csv( 'data_Races_raw_RH_2022_0910.csv', encoding = 'utf-8_sig')

RaceInfo
# ------------------------- Testing -------------------

race_html_file_name = 'Races_data_html\\2020-10-04_10_html.txt'
with open( race_html_file_name, 'r', encoding = 'utf-8_sig') as html_file:
        html_text_RaceSingle = html_file.read()

pd_html = pd.read_html( html_text_RaceSingle)
Date = race_html_file_name.split( '_')[ 2].split( '\\')[ 1]
RaceNo = race_html_file_name.split( '_')[ 3]
Venue = pd_html[0][0][0][:-1] # Venue
RaceName = pd_html[1].iloc[1, 0].split( ' - ')[0] # RaceName - Distance
Distance = pd_html[1].iloc[1, 0].split( ' - ')[1] # RaceName - Distance
VenueCondition = pd_html[1].iloc[1, 3] # Veune Condition
RaceStakes = pd_html[1].iloc[3, 0] # Stakes

RaceSingleInfo = pd_html[2]
RaceSingleInfo = RaceSingleInfo.fillna( 0)
RaceSingleInfo.loc[ :,'馬號'] = RaceSingleInfo.loc[ :,'馬號'].astype( 'float32')
RaceSingleInfo = RaceSingleInfo[ RaceSingleInfo[ '馬號'].isin( [ int( i) for i in range(1, 15)])]
nHorse = RaceSingleInfo.shape[0]
WinnerNo = RaceSingleInfo.iloc[ 0, 1]
RaceChar = [ Date, RaceNo, Venue, RaceName, Distance, VenueCondition, RaceStakes, WinnerNo, nHorse]

RaceSingleInfo = RaceSingleInfo.sort_values( by = '馬號', axis = 0)
RaceSingleInfo[ ['Date', 'RaceNo', 'Venue', 'Class', 'Distance', 'VenueCondition', 'RaceStakes', 'WinnerNo', 'nHorse']] = \
    [ Date, RaceNo, Venue, RaceName, Distance, VenueCondition, RaceStakes, WinnerNo, nHorse]
RaceSingleInfo[ 'Winner'] = RaceSingleInfo[ '名次'] == '1'
RaceSingleInfo[ 'Winner']
RaceSingle_output = RaceSingleInfo
RaceSingleInfo.loc[ :,'馬號'] = RaceSingleInfo.loc[ :,'馬號'].astype( 'int')