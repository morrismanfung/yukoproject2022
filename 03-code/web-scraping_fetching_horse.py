from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import random
import re
import time

html_link_2 = 'https://racing.hkjc.com/racing/information/chinese/Horse/SelectHorsebyChar.aspx?ordertype=2'
html_link_3 = 'https://racing.hkjc.com/racing/information/chinese/Horse/SelectHorsebyChar.aspx?ordertype=3'
html_link_4 = 'https://racing.hkjc.com/racing/information/chinese/Horse/SelectHorsebyChar.aspx?ordertype=4'

options = webdriver.ChromeOptions()
options.add_argument("start-maximized")
# to supress the error messages/logs
options.add_experimental_option('excludeSwitches', ['enable-logging'])

url_list = []

for i_content_page in [ html_link_2, html_link_3, html_link_4]:
    Browser = webdriver.Chrome( options = options, executable_path = 'C:\\Users\\User\\Documents\\ChromeDriver\\chromedriver.exe')
    Browser.get( i_content_page)
    time.sleep( 3)

    html_text_horse_list = Browser.page_source
    soup_horse_list = BeautifulSoup( html_text_horse_list, 'lxml')

    content = soup_horse_list.find_all( 'table')
    ahref_list = content[2].find_all( 'a') # content_test is the table with all the pages of each horse

    for i in range( len( ahref_list)):
        url_list.append( ahref_list[ i][  'href'])

    # for i in url_list:
    #    print( i, end = '\n')

url_list = list( map( lambda x: 'https://racing.hkjc.com' + x, url_list))

def RaceSingleMatch_html( horse_basics_url):
    html_link_HorseBasic = horse_basics_url
    Browser_RaceSingle = webdriver.Chrome( options = options, executable_path = 'C:\\Users\\User\\Documents\\ChromeDriver\\chromedriver.exe')
    Browser_RaceSingle.get( html_link_HorseBasic)
    time.sleep( 5)
    html_text_HorseBasics = Browser_RaceSingle.page_source
    df_horse_basic = pd.read_html( html_text_HorseBasics, flavor = 'lxml', displayed_only = False)
    HorseName = df_horse_basic[1][0][0]
    HorseName = re.sub( '\(.*\)', '', HorseName).strip()
    file_name = 'HorseBasics_' + HorseName + '_html.txt'
    with open( file_name, 'w', encoding = 'utf-8_sig') as file:
        file.write( html_text_HorseBasics)    
    return

horse_basics_url = url_list[ 0]

for i_index, i_horse in enumerate( url_list[ 1052:]):
    RaceSingleMatch_html( i_horse)
    print( i_horse)
    print( i_index, '/', len( url_list))

# ---------- Testing ----------
RaceSingleMatch_html( 'https://racing.hkjc.com/racing/information/chinese/Horse/OtherHorse.aspx?HorseId=HK_2017_B150')
html_link_HorseBasic = 'https://racing.hkjc.com/racing/information/Chinese/Horse/OtherHorse.aspx?HorseId=HK_2018_C233'
Browser_RaceSingle = webdriver.Chrome( options = options, executable_path = 'C:\\Users\\User\\Documents\\ChromeDriver\\chromedriver.exe')
Browser_RaceSingle.get( html_link_HorseBasic)
time.sleep( 5)
html_text_HorseBasics = Browser_RaceSingle.page_source
df_horse_basic = pd.read_html( html_text_HorseBasics, flavor = 'lxml', displayed_only = False)
HorseName = df_horse_basic[1][0][0]
file_name = 'HorseBasics_' + HorseName + '_html.txt'
with open( file_name, 'w', encoding = 'utf-8_sig') as file:
    file.write( html_text_HorseBasics) 