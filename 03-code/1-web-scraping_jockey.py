from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import random
import time

options = webdriver.ChromeOptions()
options.add_argument("start-maximized")
# to supress the error messages/logs
options.add_experimental_option('excludeSwitches', ['enable-logging'])

html_link = 'https://racing.hkjc.com/racing/information/Chinese/Jockey/JockeyRanking.aspx'

Browser = webdriver.Chrome( options = options, executable_path = 'C:\\Users\\User\\Documents\\VisualStudioCode\\WebScrappingTutorial\\chromedriver_win32_108\\chromedriver.exe')
Browser.get( html_link)
time.sleep( 5)

html_text_jockey_list = Browser.page_source
soup_jockey_list = BeautifulSoup( html_text_jockey_list, 'lxml')
df_jockey_list = pd.read_html( html_text_jockey_list, displayed_only  = False)

df_jockey_list_pd = pd.DataFrame( df_jockey_list[1])
df_jockey_list_pd = df_jockey_list_pd.set_axis ( [ 'JockeyName', 'No_1st', 'No_2nd', 'No_3rd', 'No_4th', 'No_5th', 'TotalRides', 'TotalStakes'], axis = 1)
df_jockey_list_pd = df_jockey_list_pd[ df_jockey_list_pd[ 'JockeyName'] != '其他'].dropna()

df_jockey_list_pd.to_csv( '01-data/basic-jockey.csv', encoding = 'utf-8_sig')