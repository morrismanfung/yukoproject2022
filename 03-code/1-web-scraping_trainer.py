from pickle import FALSE
from turtle import color
from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import random
import time

options = webdriver.ChromeOptions()
options.add_argument("start-maximized")
# to supress the error messages/logs
options.add_experimental_option('excludeSwitches', ['enable-logging'])

html_link = 'https://racing.hkjc.com/racing/information/Chinese/Trainers/TrainerRanking.aspx'

Browser = webdriver.Chrome( options = options, executable_path = 'C:\\Users\\User\\Documents\\ChromeDriver\\chromedriver.exe')
Browser.get( html_link)
time.sleep( 5)

html_text_trainer_list = Browser.page_source
soup_trainer_list = BeautifulSoup( html_text_trainer_list, 'lxml')
df_trainer_list = pd.read_html( html_text_trainer_list, displayed_only  = False)

df_trainer_list_pd = pd.DataFrame( df_trainer_list[1])
df_trainer_list_pd = df_trainer_list_pd.set_axis ( [ 'TrainerName', 'No_1st', 'No_2nd', 'No_3rd', 'No_4th', 'No_5th', 'TotalRides', 'TotalStakes'], axis = 1)
df_trainer_list_pd = df_trainer_list_pd[ df_trainer_list_pd[ 'TrainerName'] != '其他'].dropna()

df_trainer_list_pd.to_csv( 'Trainer.csv', encoding = 'utf-8_sig')