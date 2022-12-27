# ゆうこ (Yuko) Project 2022

![header](06-img/header-README.jpg)

The ゆうこ Project aims to generate long term positive return with horse betting with the Hong Kong Jockey Club (HKJC). We use web-scraping and a variety of machine learning methods to predict horse racing results.

## Methodology

### Basic principle

It should be noted that there is always a trade-off between precision and recall in most classification problems. In this project, we place much more significance on precision than on recall. With high recall but low precision, we may be able to have more winning bets but each loss will cost much money. Based on our tests, payoffs by setting a high recall rate usually do not offset the losses due to the low precision. With high precision and low recall, we can have more confidence on each bet while having the disadvantage of missing many possible winners. Yet, it can be compensated by increasing the bet size.

For simplicity, in the current project, we only focus on betting on Win, in which we will get a payoff only when the selected horse finishes 1st.

There are also more complicated betting options such as Quinella Place and Triple Trio. However, due to the lack of precision of the model, we will not study these betting options but only stick to Win.

Two classification approaches can be used in predicting horse racing results. The first approach is to perform a multicategorical classification task on each race. This approach attempts to predict among all the horses in the race, which one will be the winner. The second approach  is to predict whether each horse will win their race with binary classification algorithms. This approach seems to be disadvantageous because it cannot consider the opponents that a horse will encounter in their race. It is also possible that the model will output predictions that no horse will win or that more than 1 horse will win, which both scenarios are impossible. However, we still adopt the latter approach.

With each horse in a race as a single sample to be analyzed (i.e., there are around 10 samples to be analyzed in each race), the number of independent variables is greatly reduced compared to when each race is a single sample to be analyzed (i.e., there are 1 sample to be analyzed in each race). Number of possible classes also reduces from 14 to 2. This approach requires a smaller number of observations for model training. Moreover, if one treats each race as a single sample, one will need to spend extra effort to deal with the uneven number of horses participating in each race. Moreover, our empirical results support the use of this simpler approach by showing an encouraging back test return.

### Web scraping

Web scraping is done with packages selenium, BeautifulSoup, and pandas. Scripts for fetching and for extracting were written separately. Data will be mainly from two types of pages, Information and Racing results. Information includes information of horses, jockeys, and trainers. To get an exhaustive list of horses, hyperlinks are obtained from [https://racing.hkjc.com/racing/information/chinese/Horse/SelectHorsebyChar.aspx?ordertype=2](https://racing.hkjc.com/racing/information/chinese/Horse/SelectHorsebyChar.aspx?ordertype=2). While we obtained historical data of races in the previous 3 seasons, some of the horses are already retired and are not shown in the above page. We manually search the horse name and copy the pages for those retired horses for further data extraction.

### Preprocessing and feature engineering

Independent variables used in the model can be categorized as specific or general. Specific variables such as odds and horse weight are race specific and they can provide crucial information about how a horse will perform on that racing day. General variables such as horse total stake earned and jockey’s winning rate are not easy to be updated. It is difficult to track how age and total stake earned change across time. (I admit that it is do-able and that I am lazy.) Still, this information gives us a general sense of how a horse, a jockey, and a trainer is and can be potentially useful in model development.

Feature engineering was done in order to calculate indicators with temporal significance. These indicators include number of days since the last race (default = 36), place in the previous race (default = 7), mean of place of the last 3 races (default = 7), mean of place of the last 5 races (default = 7), change in declared weight since last race (default = 0), and on-race rating.

Variables used include (1) actual weight, (2) declared horse weight, (3) draw, (4) Win odds, (5) jockey’s win rate, (6) jockey’s place rate, (7) jockey’s show rate, (8) trainer’s win rate, (9) trainer’s place rate, (10) trainer’s show rate, (11) total stake the horse has earned, (12) total race the horse has run, (13) horse’s win rate, (14) horse’s place rate, (15) horse’s show rate, (16) horse age, (17) number of days since last race, (18) place in the previous race, (19) average place in the previous 3 races, (20) average place in the previous 5 races, (21) change in horse weight since the last race, (22) rating.

While the main goal is to predict the winner among horses in the same races, we can levarage the absolute attributes of each horse by computing relative scores by comparing each horse to other entries in the same race. These relative scoeres present how good a horse is, compared to other competitors. In the analysis, all of these variables are with the suffix `_norm`.

Discussion on each individual features can be found in the EDA report [here](https://github.com/morrismanfung/yukoproject2022/blob/main/04-report/EDA.ipynb).

We used recursive feature elimination (RFE) to select only 75% of the orignal features. Final features used include (1) place in the previous race, (2) average place in the previous 3 places, (3)change in horse weight since the last race, (4) Win odds, (5) total race the horse race run, (6) number of days since last race, (7) jokcey's win rate, (8)jockey's place rate, (9) jockey's show rate, (10) trainer's win rate, (11) trainer's show rate, (12) horse's win rate, (13) horse's place rate, (14) horse's show rate, (15) _relative_ actual weight, (16) _relative_ Win odds, (17) _relative_ jockey's win rate, (18) _relative_ jockey's place rate, (19) _relative_ jockey's show rate, (20) _relative_ trainer's win rate, (21) _relative_ trainer's place rate, (22) _relative_ total stake the horse has won, (23) _relative_ number of races the horse has run, (24) _relative_ horse's win rate, (25) _relative_ horse's place rate, (26) _relative_ horse's show rate, (27) _relative_ horse age, (28) _relative_ number of days since last race, (29) _relative_ average place in the previous 3 places, (30) _relative_ average place in the previous 5 places, (31)
_relative_ change in horse weight since the last race, (32) _relative_ rating.

### Machine Learning Algorithms

We tested multiple machine learning algorithms to find the optimal strategy. We used a KNN classifier, a suppeort vector classifier (SVC), a random forest classifer (RFC), a Gaussian naive Bayes classifier, a logistic regression classifier, and a linear support vector classifer (LinearSVC).

### Training and Testing

In the current project, we used stratified splitting.

A limitation of this method is that it does not enable us to estimate how the model works in given timeframes. We could not assess the effectiveness of using past racing records in predicting future events, nor exame some risk-related performances such as maximum drawdown and consecutive losses directly.

### Imbalanced data

With each horse in a race as an observation, it is inevitable that the number of winning cases will be much less than the number of losing cases. This issue is not addressed yet in this stage.

Using the argument `class_weight='balanced'` in `scikit-learn` models were considered. However, given that using a balanced `class_weight` always reduces precision and improves recall, it defeats our purpose of maximizing precision.

### Hyperparameter optimization

Optimization was performed with `scikit-learn`'s `GridSearchCV` and `RandomizedSearchCV` with precision as the objective function to be maximized.

### Threshold tuning

To ensure the models are having sufficient precision, threshold tuning was performed by visually inspecting the precision-recall curve generated by `scikit-learn`'s `precision-recall-curve` function. The manually selected thresholds for each model were included in a [spreadsheet](https://github.com/morrismanfung/yukoproject2022/blob/v1.0.1-threshold/02-model/thresholds_used.csv) which would be read in model testing.

## Results

Detailed performance of each model is attached [here](https://github.com/morrismanfung/yukoproject2022/tree/main/02-model).

Cross-validation and testing results is showed in the table below.

|       | mean train precision | mean test precision | mean train recall | mean test recall | mean train f1 | mean test f1|
| --- | ---- | ---- | ---- | ---- | ---- | ---- |
| KNN | 0.75 | 0.69 | 0.09 | 0.08 | 0.16 | 0.15 |
| SVC | 0.20 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| RFC | 0.97 | 0.55 | 0.46 | 0.18 | 0.63 | 0.27 |
| Naive Bayes | 0.51 | 0.51 | 0.16 | 0.15 | 0.24 | 0.23 |
| Logistic Regression | 0.65 | 0.63 | 0.15 | 0.15 | 0.24 | 0.24 |
| Linear SVC | 0.72 | 0.72 | 0.11 | 0.11 | 0.20 | 0.20 |

**Table 1: Cross-validation results**

|     | precision | recall | f1 |
| --- | ---- | ---- | ---- |
| KNN | 0.65 | 0.06 | 0.12 |
| SVC | 0.57 | 0.12 | 0.20 |
| RFC | 0.63 | 0.11 | 0.19 |
| Naive Bayes | 0.43 | 0.08 | 0.13 |
| Logistic Regression | 0.64 | 0.16 | 0.25 |
| Linear SVC | 0.67 | 0.10 | 0.25 |
| Vote | 0.63 | 0.15 | 0.24 |
| Any | 0.56 | 0.19 | 0.28 |

**Table 2: Testing results**

It is determined that RFC, logistic regression and linear svc work the best. A voting mahcine was built manually using this 3 models, in which a positive signal will be returned when at least 2 models give a positive prediction. In the table above, the row `Vote` shows the performance of the voting machine. The row `Any` indicates testing scores when a union (OR) rule is used, in which a positive signal is returned as long as any of the models return a signal. The voting machine performs worse than a simple logistic regression model. Thus, we decided to use the logistic regression model per se at this stage.

## Discussion

### Limitations

It should be noted that the current project is highly limited by the data source as the data collection procedure is not robust and with flaws. Any application of the project will also be limited by the obstacles in data collection. To ensure best performance for prediction, one should constantly update their database in order to obtain the most updated data which is changing every week.

One of the most serious problems is the inclusion of features related to winning rates. As these features were collected at the end of the season, they contain much information about how a particular horse, jockey, or trainer performed in the past. For a more robust design, these features should be removed or calculated manually without using the data shown online.

### Room for Improvment

Many existing limitations not mentioned above can be improved in the future.

Currently the project only focuses on class prediction but ignore a more important factor which is profitability. It should be implemented in the future.