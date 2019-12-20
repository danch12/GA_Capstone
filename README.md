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

* [Getting US census data](https://github.com/danch12/GA_Capstone/blob/master/Data%20Gathering%20and%20Cleaning%20Stage.ipynb?short_path=8abe515#L590)

For this project most of the data was obtained from the Lending Club website, they provide all of their loan data in some handy csv files. To supplement this, I also included data scraped from Wikipedia  and the US census to get a better idea of the loanee's income status compared to their state average on the intuition that money goes further in different parts of the US. For example in San Francisco 100K may not be so much compared to somewhere like Alabama.

Sources of the data -
* [Lending Club data](https://www.lendingclub.com/info/download-data.action)
* [Wikepidia average income data](https://en.wikipedia.org/wiki/Household_income_in_the_United_States)
* [US Census data](https://data.census.gov/cedsci/table?q=median%20income&g=&hidePreview=true&table=S1901&tid=ACSST1Y2018.S1901&t=Income%20%28Households,%20Families,%20Individuals%29&lastDisplayedRow=16&vintage=2018&mode=)


## Cleaning the Data and Feature Engineering

* [Link to beginning of relevant section of Notebook](https://github.com/danch12/GA_Capstone/blob/master/Data%20Gathering%20and%20Cleaning%20Stage.ipynb?short_path=8abe515#L816)


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




Once the data was clean I created a couple of columns that made sense to me, such as the loanee's income vs the state average and how long it has been since the first credit line for the account was opened. 


At this stage I made the decision to limit my dataset to only include loans from 2014 which were given a grading of C or lower. This is because the dataset was massive and from doing some initial modelling I realised that the time it took to run any form of model would lead to me falling behind and missing deadlines.Because of this I decided to look at loans that were higher risk with higher returns for investors as those are the loans that have the greatest payoff. 


## EDA


[Link to EDA](https://github.com/danch12/GA_Capstone/blob/master/EDA.ipynb)


After cleaning the dataset I decided to explore the data visually. First I created some correlation heatmaps. You can see from below that there are a couple of areas that look extremely correlated, however a lot of these variables were later dropped before modeling because you would only know about them once the loan had completed. Having said that, the heatmap still indicates that using PCA would be a good option and this is an avenue I would like to look into in the future. After looking at the correlation of the variables generally, I looked the correlation between my target variable and my independent variables. Unfortunately almost all the most correlated variables had to be removed for the same reason as above. This left me with a couple of variables that had some correlation with loan outcome but nothing standout. The next step was to create bar graphs that visualized the distribution of good vs bad loans in different categorical variables in the data. Finally I used scatter graphs and histograms to further explore the relationship between various variables, focusing mainly on loanee income as on the face of it, income would seem like a key factor in loans defaulting.

Following on from this I looked at the distribution of good vs bad loans in different categorical variables in the data.

The main takeaways from the EDA were - 
1) We should focus on lower grade loans as they seem the most volatile
2) Income does not have as big of an impact on loans being paid as one may believe
3) There does not seem like any linear seperation between the two classes so linear models may perform badly
4) PCA seems like a good tool to use in this project as many independent variables are correlated


[Put in graphs here]



## Modelling 

[Link to beginning of modelling section](https://github.com/danch12/GA_Capstone/blob/master/Capstone%20Modelling%20and%20Evaluation%20phase.ipynb?short_path=90fa5f0#L262)

As alluded to in the EDA section, I was not hopeful when running a logistic regression model so I quickly diverted my attention towards other tree based models such as random forest and ada boost. I found much more success in these models so eventually I tried an XG boost model to comparitively great success. Overall I still was not happy with the results from the models which lead me to using NLP.

Results from the [XG Boost cv](https://github.com/danch12/GA_Capstone/blob/master/Capstone%20Modelling%20and%20Evaluation%20phase.ipynb?short_path=90fa5f0#L642) -



## NLP and Modelling

[Link to processing job titles](https://github.com/danch12/GA_Capstone/blob/master/Data%20Gathering%20and%20Cleaning%20Stage.ipynb?short_path=8abe515#L1073)

[Link to running models with job titles included](https://github.com/danch12/GA_Capstone/blob/master/Capstone%20Modelling%20and%20Evaluation%20phase.ipynb?short_path=90fa5f0#L925)

I will keep the NLP section brief as it had very little effect on the performance of any model. I used a count vectorizor on the employment title column with a high minimum appearance limit as I wanted the model to generalize well. Additionally as the models were already taking a long time to run I only included words that were discrinatory, for example 74% of engineers fully paid off their loan compared to a baseline of 66% so the word engineer was included. In the future I will improve this section of my project and expand the amount of words included. Model performance did not increase after including the job titles. 


## Stacking Models

As NLP did not provide the results I wanted, I took a different path towards stacking models. This part of my project I found incredibly interesting and would like to do more of in the future.

Before I start there are some really good articles on model stacking that helped me a lot with this part of my project-

1) First and foremost the ML Wave Ensemble guide seems to be the holy book on stacking models, it can be found [Here](https://mlwave.com/kaggle-ensembling-guide/)
2) For a slightly simpler overview on stacking models, there is a really good kaggle article using the titanic dataset [Here](https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python)
3) This is not an article but this script highlighted how you can reuse the same models to great effect [Here](https://github.com/emanuele/kaggle_pbr/blob/master/blend.py)
4) Finally most of the above articles were quite old so I thought I'd include one that has more up to date code [Here](https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/)

Explaination of Model Stacking-

[Picture of Spaceship]

The ML Wave article has an great example on why you would want to stack models on top of each other, this example looks at the first spaceships and how messages were communicated.

[Communications image]

You could imagine that making a communication error whilst trying to land on the moon could be very costly as lives could be lost. Therefore to solve this the spacemen used [Repitition coding](https://en.wikipedia.org/wiki/Repetition_code) meaning they would send the same message a number of times and then do a majority vote.

[Repition Code](http://www.inference.org.uk/mackay/itprnn/1997/l1/img13.gif)






