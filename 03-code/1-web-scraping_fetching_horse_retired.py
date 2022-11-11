from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import random
import time

options = webdriver.ChromeOptions()
options.add_argument("start-maximized")
# to supress the error messages/logs
options.add_experimental_option('excludeSwitches', ['enable-logging'])

missing = pd.read_csv( 'Missing horses_2.csv', encoding = 'utf-8_sig')
url_list = list( missing.iloc[ 409:, 1])

def RaceSingleMatch_html( horse_basics_url):
    html_link_HorseBasic = horse_basics_url
    Browser_RaceSingle = webdriver.Chrome( options = options, executable_path = 'C:\\Users\\User\\Documents\\ChromeDriver\\chromedriver.exe')
    Browser_RaceSingle.get( html_link_HorseBasic)
    time.sleep( 5)
    html_text_HorseBasics = Browser_RaceSingle.page_source
    df_horse_basic = pd.read_html( html_text_HorseBasics, flavor = 'lxml', displayed_only = False)
    HorseName = df_horse_basic[1][0][0]
    file_name = 'HorseBasics_' + HorseName + '_html.txt'
    with open( file_name, 'w', encoding = 'utf-8_sig') as file:
        file.write( html_text_HorseBasics)    
    return

for i_index, i_horse in enumerate( url_list):
    RaceSingleMatch_html( i_horse)
    print( i_horse)
    print( i_index, '/', len( url_list))