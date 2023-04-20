# Final Proposal 

## Project Title : Android application analysis and comparision guide
## Project Overview
There are mobile applications everywhere. They are simple to make and may be profitable. Because of these two considerations, an increasing number of applications are being produced. In this project, we will thoroughly study the Android app industry by comparing over ten thousand apps from Google Play store in various categories. We'll seek for insights in the data to develop growth and retention plans, as well as provide a competitive analysis.

## Objective
Predecitve analysis on market capture and user reach of an app based on app category and app description by a user, by providing a dashboard to the user with the details of predicted downloads count, expected age group of users using the app, suggesting optimal size of the app for high downloads, possible price of the app, and list the top 5 competeting apps.
  
## Important input features for analysis
  1. App Name
  2. App Description

## Expected outcome
  1. Predicted count of downloads/installs
  2. Predicted age group of the users
  3. Predicting app type (free or paid)
  4. Predicting price of the app if user wants to make the app a paid service
  5. list of top 5 competeting apps in the same category of the app
  6. Predicted app rating.

## Techniques Used
  1. Statistical approach for providing basic analytic for similar apps in the category selected by the user.
  2. Tree based algorithms or ensembles to decide on whether the app should be paid or free to have the better reach in the audience users.
  3. Using GridSearchCV or soft voting methods to select the best model for classification used in point 2.
  4. Using Cosine similarty metrics to get most accurate alternative apps for the user provided app description / app title and category of the app.
  5. Streamlit based user interface.

## Metrics for model evaluation
  1. Accuracy and confusion matrix
  2. Recall and Precision scores
  3. ROC and AUC curve

## Dataset description:

### 1. AppDescriptions.csv
   
   This Dataset is generated using *WebScraping* and contains two features as follows
   
   ### Dataset Columns
   
   1. App         : name of the app
   2. Description : description of the app

### 2. apps.csv
   
   This Dataset provides preliminary data regarding the apps on the google play store including the below mentioned details
   
   ### Dataset columns
   
   1.   App             : name of the app
   2.   Category        : category of the app
   3.   Rating          : Rating of the app
   4.   Reviews         : number of reviews for the app
   5.   Size            : size of the app
   6.   Installs        : number of installations done / count of downloads
   7.   Type            : Free or paid
   8.   Price           : price of the app
   9.   Content Rating  : age group of users who rated the app
   10.  Genres          : gener of the app / similar to category
   11.  Last Updated    : last updated date and time
   12.  Current Ver     : version of the app
   13.  Android Ver     : android supported version


   
### data source: Web scrapping webpages, Kaggle
### data References: 
  1. https://www.kaggle.com/code/mohammedmurtuzalabib/android-app-market-on-google-play-analysis/data


## Technologies Used
- programming language : Python
- libraries : Pandas, Plotly, sklearn, scipy, streamlit, tf-idf
- algorithms : Text based classification using SGD algorithms


