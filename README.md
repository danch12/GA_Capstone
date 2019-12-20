# When Is it a good time to be a loan
Using Lending Club Data to predict the outcome of loans



## Table Of Contents
* [Technologies Used](https://github.com/danch12/GA_Capstone/blob/master/README.md#technologies-used)
* [Packages Used](https://github.com/danch12/GA_Capstone/blob/master/README.md#packages-used)
* [Methods Used](https://github.com/danch12/GA_Capstone/blob/master/README.md#methods-used)
* [Introduction](https://github.com/danch12/GA_Capstone/blob/master/README.md#introduction)
* [Gathering the Data](https://github.com/danch12/GA_Capstone/blob/master/README.md#gathering-the-data)




## Technologies Used
* Python 3.0
* Jupyter Notebook


## Packages Used
* Pandas
* Scikit-learn
* Matplotlib
* Numpy
* itertools
* scipy
* xgboost
* joblib

## Methods Used

* Classification
* Data Visualization
* NLP
* Model Stacking
* Web Scraping


## Introduction
Looking across the internet I saw that Lending Club (a peer to peer lending company) releases a massive amount of data every quarter on the status of every loan opened with them during that quarter. Therefore I thought it would be really interesting to predict the outcome of loans based on information you would know during the lifetime of the loan.The results of this project could have an exciting impact on how Lending Club looks at it's loanees as if we can predict a loan is going to default then we can take preventative measures to stop the loan from defaulting. Additionally as predicting the outcome of loans is quite difficult, it required me to seek out niche, powerful techniques that were previously unknown to me.


## Gathering The Data

[Link to relevant section of Notebook](https://github.com/danch12/GA_Capstone/blob/16b422104066f7b96929d8ae142c9320008343b4/Data%20Gathering%20and%20Cleaning%20Stage.ipynb#L53)

For this project most of the data was obtained from the Lending Club website, they provide all of their loan data in some handy csv files. To supplement this, I also included data scraped from [Wikepdia](https://github.com/danch12/GA_Capstone/blob/16b422104066f7b96929d8ae142c9320008343b4/Data%20Gathering%20and%20Cleaning%20Stage.ipynb#L258) and the [US cencus](https://github.com/danch12/GA_Capstone/blob/16b422104066f7b96929d8ae142c9320008343b4/Data%20Gathering%20and%20Cleaning%20Stage.ipynb#L590) to get a better idea of the loanee's income status compared to their state average on the intuition that money goes further in different parts of the US. For example in San Francisco 100K may not be so much compared to somewhere like Alabama.

Sources of the data -
* [Lending Club data](https://www.lendingclub.com/info/download-data.action)
* [Wikepidia average income data](https://en.wikipedia.org/wiki/Household_income_in_the_United_States)
* [US Census data](https://data.census.gov/cedsci/table?q=median%20income&g=&hidePreview=true&table=S1901&tid=ACSST1Y2018.S1901&t=Income%20%28Households,%20Families,%20Individuals%29&lastDisplayedRow=16&vintage=2018&mode=)


