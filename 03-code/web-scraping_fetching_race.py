# A new version of script to download all the html first before data extraction

from selenium import webdriver
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import random
import time
from datetime import datetime
import os

options = webdriver.ChromeOptions()
options.add_argument("start-maximized")
# to supress the error messages/logs
options.add_experimental_option('excludeSwitches', ['enable-logging'])


html_link = 'https://racing.hkjc.com/racing/information/Chinese/racing/LocalResults.aspx'

Browser = webdriver.Chrome( options = options, executable_path = 'chromedriver_win32\\chromedriver.exe')
Browser.get( html_link)
time.sleep( 5)

html_text = Browser.page_source
soup = BeautifulSoup( html_text, 'lxml')
values = soup.find_all( 'option')
values

dates = []
for i in values:
    dates.append( i.text)

def reformat( dates_i):
    d, m, y  = dates_i.split( '/')
    output_text = y + '/' + m + '/' + d
    return( output_text)

reformated_dates = list( map( reformat, dates))

current_time = datetime.now()
file_name = 'ReformatedDates' + '0801' +'.csv'
pd.DataFrame( reformated_dates).to_csv( file_name)

def RaceSingleMatch_html( target_date, target_raceno):
    html_link_RaceSingle = 'https://racing.hkjc.com/racing/information/chinese/Racing/LocalResults.aspx?RaceDate=' + target_date + link_venue + target_raceno
    Browser_RaceSingle = webdriver.Chrome( options = options, executable_path = 'chromedriver_win32\\chromedriver.exe')
    Browser_RaceSingle.get( html_link_RaceSingle)
    time.sleep( 3)
    html_text_RaceSingle = Browser_RaceSingle.page_source
    file_name = target_date + '_' + target_raceno + '_html.txt'
    file_name = file_name.replace( '/', '-')
    with open( file_name, 'w', encoding = 'utf-8_sig') as file:
        file.write( html_text_RaceSingle)    
    return
    
# for race_day in reformated_dates[ :6]:
for race_day in [ '2022/07/01', '2022/07/06', '2022/07/10', '2022/07/13', '2022/07/16']:
    date_RaceDay = race_day
    html_link_RaceDay = 'https://racing.hkjc.com/racing/information/chinese/Racing/LocalResults.aspx?RaceDate=' + date_RaceDay

    Browser_RaceDay = webdriver.Chrome( options = options, executable_path = 'chromedriver_win32\\chromedriver.exe')
    Browser_RaceDay.get( html_link_RaceDay)
    time.sleep( 5)

    html_text_RaceDay = Browser_RaceDay.page_source
        
    soup_RaceDay = BeautifulSoup( html_text_RaceDay, 'lxml')
    values_races = soup_RaceDay.find( 'table', class_ = 'f_fs12 js_racecard')

    if values_races is None:
        continue

    venue_info_table = pd.read_html( html_text_RaceDay)
    venue_info = venue_info_table[0].iloc[ 0, 0]
    
    if '跑馬地'  in venue_info:
        link_venue = '&Racecourse=HV&RaceNo='
    elif '沙田' in venue_info:
        link_venue = '&Racecourse=ST&RaceNo='
    else:
        continue

    nRace = len( values_races.find_all( 'a')) # A trick to find thenumber of races on the day

    for i in range( 1, nRace+1):
        noOfRace = str(i)
        RaceSingleMatch_html( date_RaceDay, noOfRace)