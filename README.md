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


## Gathering the Data

* [Link to beginning of relevant section of Notebook](https://github.com/danch12/GA_Capstone/blob/16b422104066f7b96929d8ae142c9320008343b4/Data%20Gathering%20and%20Cleaning%20Stage.ipynb#L53)

* [Scraping data from Wikipedia](https://github.com/danch12/GA_Capstone/blob/master/Data%20Gathering%20and%20Cleaning%20Stage.ipynb?short_path=8abe515#L258)

* [Getting US census data](https://github.com/danch12/GA_Capstone/blob/16b422104066f7b96929d8ae142c9320008343b4/Data%20Gathering%20and%20Cleaning%20Stage.ipynb#L590)

For this project most of the data was obtained from the Lending Club website, they provide all of their loan data in some handy csv files. To supplement this, I also included data scraped from Wikipedia  and the US census to get a better idea of the loanee's income status compared to their state average on the intuition that money goes further in different parts of the US. For example in San Francisco 100K may not be so much compared to somewhere like Alabama.

Sources of the data -
* [Lending Club data](https://www.lendingclub.com/info/download-data.action)
* [Wikepidia average income data](https://en.wikipedia.org/wiki/Household_income_in_the_United_States)
* [US Census data](https://data.census.gov/cedsci/table?q=median%20income&g=&hidePreview=true&table=S1901&tid=ACSST1Y2018.S1901&t=Income%20%28Households,%20Families,%20Individuals%29&lastDisplayedRow=16&vintage=2018&mode=)


## Cleaning the Data and Feature Engineering

* [Link to beginning of relevant section of Notebook](https://github.com/danch12/GA_Capstone/blob/16b422104066f7b96929d8ae142c9320008343b4/Data%20Gathering%20and%20Cleaning%20Stage.ipynb#L816)


Even though the data came straight from Lending club it was fairly messy with quite a lot of NA values. I assumed that most of these were due to attributes that did not apply to the loanee, for example if a loanee took out a loan by themselves the joint income column would be a NA value. I grouped all of the cleaning steps into one function for ease of use. This function can be seen below -
```python
def cleaner(data,min_list=None,max_list=None,cat_list=None,date_list=None):
     
     
    all_data_na=data.isna().sum()/len(data)*100
    all_data_na=all_data_na.drop(all_data_na[all_data_na==0].index).sort_values(ascending=False)
    print('columns with missing data before clean-\n',all_data_na)
    print('-'*20)
     
    if cat_list != None:
         
        for column in cat_list:
            data[column].fillna(' ', inplace=True)
     
    if date_list != None:
        
        for column in date_list:
            data[column]=pd.to_datetime(data[column],infer_datetime_format=True)
             
             
    if min_list!=None:
         
        for column in min_list:
            data[column].fillna((data[column].min()),inplace=True)
     
    if max_list!=None:
         
        for column in max_list:
            data[column].fillna((data[column].max()),inplace=True)
     
    #then drop dregs
    data.dropna(inplace=True)
     
    all_data_na=data.isna().sum()/len(data)*100
    all_data_na=all_data_na.drop(all_data_na[all_data_na==0].index).sort_values(ascending=False)
    print('columns with missing data after clean-\n',all_data_na)
     
    return data
```

[put in image of dirty data]


Once the data was clean I created a couple of columns that made sense to me, such as the loanee's income vs the state average and how long it has been since the first credit line for the account was opened. 


At this stage I made the decision to limit my dataset to only include loans from 2014 which were given a grading of C or lower. This is because the dataset was massive and from doing some initial modelling I realised that the time it took to run any form of model would lead to me falling behind and missing deadlines.Because of this I decided to look at loans that were higher risk with higher returns for investors as those are the loans that have the greatest payoff. 


## EDA







