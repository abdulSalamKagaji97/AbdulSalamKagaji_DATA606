# Title: Android App Analyzer - Insights for App Development

## Introduction:
This project aims to analyze the Android app market to provide insights on market demand, estimated reach, and competition. The report presents a comprehensive analysis of 10,000+ Android apps across categories and aids in developing growth plans and decision-making for app development.

## Objective:
The primary objective of this project is to provide insights and recommendations for developers to improve their app's user experience, functionality, and regular updates. With these insights, developers can create more accessible, engaging, and higher quality apps that meet users' needs and ultimately achieve greater success in the Android app market.

## Literature Survey:
The Android app market has experienced significant growth in recent years, with over 2.87 million apps available in 2020 and a projected increase to 3.4 million by 2023. This increase in the number of apps available has also led to high competition in the market. In 2020, the uninstall rate of Android apps worldwide was approximately 30%, highlighting the importance of delivering high-quality apps that meet users' needs and expectations to succeed. Therefore, developers need to focus on creating engaging, user-friendly, and high-quality apps that provide value to their users to stand out in this competitive market.

## Data Collection:
The project uses two datasets, namely, apps.csv and AppDescriptions.csv, to provide preliminary data regarding the apps on the Google Play Store. The datasets contain information such as app name, category, rating, reviews, size, installs, type, price, content rating, genres, last updated, and android version. AppDescroptions dataset contains information like app name and add description

## Attribute description:

### apps.csv

App 	: name of the app
Category 	: category of the app
Rating 	: Rating of the app
Reviews 	: number of reviews for the app
Size 	: size of the app
Installs 	: number of installations done / count of downloads
Type 	: Free or paid
Price 	: price of the app
Content Rating : age group of users who rated the app
Genres 	: gener of the app / similar to category
Last Updated 	: last updated date and time
Current Ver 	: version of the app
Android Ver 	: android supported version

### AppDescriptions.csv

App : name of the app
Description : app description


## Merged Dataset based on App Name:
The merged dataset provides a comprehensive view of all the information gathered from the two datasets.
![mergedDataset][]

## App Category Segregation:
The app category segregation provides an insight into the number of apps available in each category and their respective ratings and reviews.
![image1][]

## Ratings of Paid and Free Apps per category:
This section presents a comparison of the ratings and reviews of paid and free apps in each category.
![image2][]

## Attribute Correlations:
The attribute correlations provide a correlation between the different attributes such as rating, reviews, and installs.
![image3][]

## Machine Learning Algorithm Used:
The project uses the SGD classifier, a commonly used algorithm for text-based classification tasks, to predict app ratings, count of downloads/installs, and user age group.

The SGD classifier is a commonly used algorithm for text-based classification tasks. It is known for its efficiency, scalability, and ability to handle large datasets. Many natural language processing applications, such as sentiment analysis and spam filtering, have achieved good results using the SGD classifier.

  ### Advantages:
  - The SGD classifier is efficient for text-based classification tasks.
  - It can handle large datasets and has a low memory footprint.
  - The algorithm is easily scalable for high-dimensional feature spaces.
  - The SGD classifier achieves good performance in natural language processing applications.
  - Some of these applications include sentiment analysis, spam filtering, and document classification.

## Machine Learning Pipeline:
The machine learning pipeline includes data preprocessing, model training, and model evaluation.
![image4][]
![image5][]

## User Interface:
The user interface allows users to input app details for analysis.

## Outcomes:
The project provides the predicted count of downloads/installs, app rating, age group of users, app type (free or paid), predicted price of the app if the user wants to make the app a paid service, and a list of top 5 competing apps in the same category of the app.

## Conclusion:
In conclusion, the project provides a comprehensive analysis of the Android app market and presents insights and recommendations for developers to improve their app's user experience, functionality, and regular updates. With these insights, developers can create more accessible, engaging, and higher quality apps that meet users' needs and ultimately achieve greater success in the Android app market.
