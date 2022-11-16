![header.png]
# Introduction on code file

## Preface

As the codes written are not updated for a long while during the later stage of the project, they may not be able to 1) fetch data online as the sites have changed their code, 2) read and create the correct files as the directory structure has changed serveral times.

## `1-web-scraping_fetching_race.py`

This script used `selenium` to fetch the `html` code of all the existing pages of single races possibly obtained by a user.

It saves 1 `.txt` file for each race. `html` code in the output files will be extracted in later stage.

## `1-web-scraping_fetching_horse.py` and `1-web-scraping_fetching_horse_retired.py`

This script used `selenium` to fetch the `html` code of all the horses.

It saves 1 `.txt` file for each horse. `html` code in the output files will be extracted in later stage.

Retired horses are handled separately as they cannot be found on the web page directly. Names and corresponding pages of retired horses are searched manually based on the missing data.

## `1-web-scraping_extracting_race.py`

The script reads `.txt` files and returns a `.csv` file storing the information of each race. Due to concerns of practicality, the resulting file does not store 1 race as 1 observation. Instead, each observation in the file will be the performance of a SINGLE horse in a SINGLE race. Thus, the observation contains information including horse's name, jockey's and trainer's name, odd, date, race number, whether the horse wins etc.

The output file will be further processed to match the information of horses, jockeys, and trainers before modeling.

## `1-web-scraping_extracting_horse.py` and `1-web-scraping_extracting_horse_retired.py`

The script reads `.txt` files storing the `html` code of the pages with horses' information. Other than a `.csv` storing the general information of the horses, i.e., win rate, age etc., the script also generates 1 `.csv` file for each horse. This unique `.csv` file stores the race history of the particular horse, which enables us to access to temporal features such as change in weight, average place before a race etc.

## `1-web-scraping_jockey.py` and `1-web-scraping_trainer.py`

As the number of jockeys and trainers is small and it is easy to read their winning rates from a single table, we did not create scripts for fetching and extracting directly but scraped the data directly with `pandas` methods.

## `2-data-preparation.py`

The script reads the data from the scripts above and return a `.csv` file named `data-merged_yyyymmdd.csv`. This file will preprocessed before exploratory data analysis.

## `3-normalization.py`

The script reads the file `data-merged_yyyymmdd.csv` and normalizes the features based on the race each horse participated in. The normalized scores can be used as relative indexes of how a horse performs relative to other contestants in the same race.

## `4-data-preparation-prediction`

The script reads the file `00-future-input` in the folder `05-prediction` and returns a `.csv` file named `01-preprocessed_yyyymmdd` for prediction. The preprocessing steps are almost identical to those in `2-data-preparation`.