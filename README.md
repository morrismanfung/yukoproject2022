# ゆうこ (Yuko) Project 2022
	
The ゆうこ Project aims to generate long term positive return with horse betting with the Hong Kong Jockey Club (HKJC). We use web-scraping and a variety of machine learning methods to predict horse racing results. Using backtest data, the project can generate a 19,917% profit from November 21, 2021 to June 25, 2022.
    
# Methodology

## Basic principle
For the sake of concision (and due to laziness), the current report will not thoroughly discuss methods not being used in the final model. Previous attempted alternatives will only be discussed with minimum information provided.

It should be noted that there is always a trade-off between precision and recall in most classification problems. In this project, we place much more significance on precision than on recall. With high recall but low precision, we may be able to have more winning bets but each loss will cost much money. Based on our tests, payoffs by setting a high recall rate usually do not offset the losses due to the low precision. With high precision and low recall, we can have more confidence on each bet while having the disadvantage of missing many possible winners. Yet, it can be compensated by increasing the bet size.

In the current project, we only focus on betting on Win, in which we will get a payoff only when the selected horse finishes 1st. We decided not to bet on Show, in which we would get a payoff when the selected horse finishes 1st, 2nd, or 3rd in a race. The reason is that the model does not yield greater precision when predicting Show rate. Only the recall rate was improved when we tried to predict Show rate instead of Win rate. As we value precision more and that Show usually comes with lower odds, we focus on Win instead of Show.

There are also more complicated betting options such as Quinella Place and Triple Trio. However, due to the lack of precision of the model, we will not study these betting options but only stick to Win.

Two classification approaches can be used in predicting horse racing results. The first approach is to perform a multicategorical classification task on each race. This approach attempts to predict among all the horses in the race, which one will be the winner. The second approach  is to predict whether each horse will win their race with binary classification algorithms. This approach seems to be disadvantageous because it cannot consider the opponents that a horse will encounter in their race. It is also possible that the model will output predictions that no horse will win or that more than 1 horse will win, which both scenarios are impossible. However, we still adopt the latter approach.

With each horse in a race as a single sample to be analyzed (i.e., there are around 10 samples to be analyzed in each race), the number of independent variables is greatly reduced compared to when each race is a single sample to be analyzed (i.e., there are 1 sample to be analyzed in each race). Number of possible classes also reduces from 14 to 2. This approach requires a smaller number of observations for model training. Moreover, if one treats each race as a single sample, one will need to spend extra effort to deal with the uneven number of horses participating in each race. Moreover, our empirical results support the use of this simpler approach by showing an encouraging back test return.

## Web scraping

Web scraping is done with packages selenium, BeautifulSoup, and pandas. Scripts for fetching and for extracting were written separately. Data will be mainly from two types of pages, Information and Racing results. Information includes information of horses, jockeys, and trainers. To get an exhaustive list of horses, hyperlinks are obtained from https://racing.hkjc.com/racing/information/chinese/Horse/SelectHorsebyChar.aspx?ordertype=2. While we obtained historical data of races in the previous 3 seasons, some of the horses are already retired and are not shown in the above page. We manually search the horse name and copy the pages for those retired horses for further data extraction.

## Data Preprocessing & Data Engineering

Independent variables used in the model can be categorized as specific or general. Specific variables such as odds and horse weight are race specific and they can provide crucial information about how a horse will perform on that racing day. General variables such as horse total stake earned and jockey’s winning rate are not easy to be updated. It is difficult to track how age and total stake earned change across time. (I admit that it is do-able and that I am lazy.) Still, this information gives us a general sense of how a horse, a jockey, and a trainer is and can be potentially useful in model development.

Feature engineering was done in order to calculate indicators with ... ***Contents critical to the project were removed. Please contact the author.***

We attempted to use subsets of variables. The result shows that the more variables we used, the more accurate the model prediction is. Thus, we use all variables in the final model.

Variables used include... ***Contents critical to the project were removed. Please contact the author.***

## Machine Learning

### Algorithms

Multiple machine learning algorithms were used before we found the optimal strategy. We began with Support Vector Machine (SVM), K Nearest Neighbors (KNN), Random Forest Classifier (RFC), Gradient Boost Classifier (GBC), Histogram Gradient Boost Classifier (HGBC), and Artificial Neural Network (ANN). Ensemble methods including RFC, GBC, HGBC work the best.

With the aim to further improve precision, we attempted to ... ***Contents critical to the project were removed. Please contact the author.*** Thus, the final model will output a number ranging from 0 to 1, representing the probability that a horse will win a race. The use of ***removed*** yielded the best result and is currently used in the project.

### Training and Testing.

Instead of randomly splitting a proportion of the observed sample into training and testing sets, we decided to assign the first 10,000 observations (62%) to the training set and the remaining 6,059 (38%) to the testing set. The testing data consist of observations from November 10th, 2021 to July 1st, 2022. Although it makes the results highly susceptible to sampling bias, it allows us to assess the effectiveness of using past racing records to predict future events. It also enables us to examine some risk-related performances such as maximum drawdown and consecutive losses.

### Imbalanced data.

With each horse in a race as an observation, it is inevitable that the number of winning cases will be much less than the number of losing cases. While changing the class weight does not alternate the performance, we decided to keep the class weights as default without specifying them.

# Results

## Thresholds

Given the trade-off between precision and recall, we value precision much more than recall. Thus, we picked a threshold of 0.7 for classification, i.e., if the model output is greater than or equal 0.7, it should classify the horse as a winning horse, else the horse will be classified as losing.

![Threshold](https://github.com/morrismanfung/yukoproject2022/blob/main/image/precision-recall_20220808.png)

## Performance

Using the abovementioned threshold, the performance is promising. The precision is high although the recall is very low. As we value precision rather than recall, we deem the high precision is a success of our project, and is potentially leading to great return.

![Performance](https://github.com/morrismanfung/yukoproject2022/blob/main/image/classification_report_20220808.png)
## Backtest return

While we cannot be perfect in predicting horse racing results, we need to consider the frequency of losses as it will seriously affect our bankroll. Thus, we decided to bet 10% of the bankroll every time the model predicts that a horse will win a race. The percentage is tentative right now due to 2 reasons. First, it is an arbitrary number without any support with statistics. Secondly, as the bankroll increases, the betting percentage has to be reduced as the bet will be heavily affecting the odds when it is large enough.

## Betting in Hastings

We are currently testing the generalizability of the model to other racecourse. Below is our betting history in Hastings Racecourse, Vacouver.

| Date | RaceNo | HorseNo | HorseName | Bet | Win | Acc. |
| ---- |--------|--------|------------|-----|-----|------|
|8/22  |3       |2       |Be Quick    |1    |3.2  |3.2   |
|8/22  |5       |8       |Legacy Square|1   |0    |2.2   |

| Total Betting Amount | Total Earning |
| -------------------- | -------------- |
|2                     |3.2|
