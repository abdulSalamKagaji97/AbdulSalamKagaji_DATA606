"""
Streamlit code for project frontend UI of Capstone project Data 606
Project Name : Android App Analyzer
Name : Abdul Salam Kagaji
Campus ID : XH61938
"""

# importing modules

import streamlit as st
import numpy as np
import plotly.figure_factory as ff

import plotly.express as px
from PIL import Image
import random
import pandas as pd
import math


import plotly.figure_factory as ff

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

import math
import pickle
import re
from collections import Counter
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np

# reading app datasets

appDf = pd.read_csv('https://raw.githubusercontent.com/abdulSalamKagaji97/AbdulSalamKagaji_DATA606/main/data/apps.csv')
appDescDf = pd.read_csv('https://raw.githubusercontent.com/abdulSalamKagaji97/AbdulSalamKagaji_DATA606/main/data/AppDescriptions.csv')

# image path for app icons
imagePath = "https://developer.android.com/static/guide/practices/ui_guidelines/images/article_icon_adaptive.gif"

# merging datasets
mergeDf = appDf.merge(appDescDf,on="App")

# data cleaning and feature selection
mergeDf.dropna(inplace=True)
mergeDf = mergeDf[[ 'App', 'Descriptions', 'Category', 'Rating', 'Reviews', 'Size',
       'Installs', 'Type', 'Price', 'Content Rating', 'Genres']]
mergeDf['Installs'] = [int(re.sub("[^\d\.]", "", value)) for value in mergeDf['Installs']]
mergeDf['Price'] = [float(re.sub("[^\d\.]", "", value)) for value in mergeDf['Price']]

# creating dataset for finding similar apps
for_similarity_x = list(mergeDf['Descriptions'])
for_similarity_x = for_similarity_x + list(mergeDf['App'])

for_similarity_y = list(mergeDf['App'] +"||"+ mergeDf['Size'].astype(str) \
                +"||"+ mergeDf['Rating'].astype(str) \
                + "||"+mergeDf['Installs'].astype(str)\
                + "||"+mergeDf['Type'].astype(str) \
                + "||"+mergeDf['Price'].astype(str) \
                + "||"+mergeDf['Genres'].astype(str) \
                + "||"+mergeDf['Content Rating'].astype(str) \
                + "||" + mergeDf['Descriptions'].astype(str))
for_similarity_y += for_similarity_y

df = pd.DataFrame({'utt':for_similarity_x,'app':for_similarity_y})
df.columns = ['utt','app']

WORD = re.compile(r"\w+")

# function to convert text to word vectors
def text_to_vector(text):
        words = WORD.findall(text)
        return Counter(words)

# function to calculate similarity between two word vectors
def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

# function to get list of similar apps
def get_similar_apps(intent):
    actual_intent = intent

    similarity_score_list = []
    similarity_checked_utterance = []
    similarity_checked_output = []
    for utterance in for_similarity_x:
        # print(utterance)
        text1 = actual_intent
        text2 = utterance

        vector1 = text_to_vector(text1)
        vector2 = text_to_vector(text2)

        cosine_value = get_cosine(vector1, vector2)

        similarity_score_list.append(cosine_value)
        similarity_checked_utterance.append([cosine_value, utterance])
    similarity_score_list_temp = similarity_score_list.copy()
    similarity_score_list.sort(reverse=True)
    similarity_checked_output_temp = []
    count = 0
    similarity_scores = []
    similar_app_details = []
    for _ in range(len(similarity_score_list)):
        if count <= 5:
            for utt in similarity_checked_utterance:
                if utt[0] == similarity_score_list_temp[similarity_score_list_temp.index(similarity_score_list[_])] and utt[1] not in similarity_checked_output_temp:
                    similarity_checked_output_temp.append(utt[1])
                    # similar_app_details.append(for_similarity_y[for_similarity_x.index(utt[1])])
                    similarity_scores.append(similarity_score_list[_])
                    count += 1
       

    for utt in similarity_checked_output_temp:
        if for_similarity_y[for_similarity_x.index(utt)] not in similarity_checked_output:
            # print(y[x.values.tolist().index(utt)])
            similarity_checked_output.append(
                for_similarity_y[for_similarity_x.index(utt)])
    print("app similars")
   
    return list(zip(similarity_checked_output[:3],similarity_scores[:3]))

# function to get classifications for user input
def get_classifications(intent):
    sgd = pickle.load(open("./sgdModelMultiOut.p",'rb'))
    y_pred_svm = sgd.predict([intent])
    y_pred_svm = y_pred_svm[0].split("||")
    vector1 = text_to_vector(y_pred_svm[0])
    vector2 = text_to_vector(intent)
    y_pred_svm.append(get_cosine(vector2,vector1))
    df_sample = df.sample(1000)
    y_true = [x.split("||")[0] for x in df_sample['app'].values.tolist()]
    y_pred = sgd.predict(df_sample['utt'])
    y_pred = [x.split("||")[0] for x in y_pred]
    y_pred_svm.append(accuracy_score(y_true,y_pred))

    return y_pred_svm
    
# function to plot accuracy graphs based on app name
def get_app_name_accuracy_plot(score):
    labels = ["Accuracy score","error score"]
    values = [score,100-score]
    fig4 = px.pie(labels, values = values, hole = 0.7,
                names = labels, color = labels,
                color_discrete_map = {"Accuracy score":'#8AFF8A',"error score":'transparent'},
                width=300,height=250,)

    fig4.update_traces(
                    title_font = dict(size=25,family='Verdana', 
                                        color='darkred'),
                    hoverinfo='label+percent',
                    textinfo='percent')
    # fig1.show()
    fig4.update_layout(showlegend=False)
    return fig4

# function to plot accuracy graphs based on app description
def get_description_accuracy_plot(score):
    labels = ["Accuracy score","error score"]
    values = [score,100-score]
    fig5 = px.pie(labels, values = values, hole = 0.7,
                names = labels, color = labels,
                color_discrete_map = {"Accuracy score":'#10c2ff',"error score":'transparent'},
                width=300,height=250,)

    fig5.update_traces(
                    title_font = dict(size=25,family='Verdana', 
                                        color='darkred'),
                    hoverinfo='label+percent',
                    textinfo='percent')
    # fig1.show()
    fig5.update_layout(showlegend=False)
    return fig5

def most_frequent(List):
    return max(set(List), key = List.count)



# creating data for app store summary graph

df_categories_app_count = mergeDf.groupby("Category").App.count()
df_categories_app_count = df_categories_app_count.reset_index()
df_categories_app_count.columns = ['Category','count']

df_categories_app_rating = mergeDf.groupby("Category").Rating.mean()
df_categories_app_rating = df_categories_app_rating.reset_index()
df_categories_app_rating.columns = ['Category','mean']

df_categories_app_installations = mergeDf.groupby("Category").Installs.mean()
df_categories_app_installations = df_categories_app_installations.reset_index()
df_categories_app_installations.columns = ['Category','mean']


# initializing a figure
fig = go.Figure()

values = list(range(40))

fig.add_trace(
    go.Bar(
        x=df_categories_app_count['Category'].values.tolist(),
        y=df_categories_app_count['count'].values.tolist(),
        name = "App count",
        marker={'color': '#63F5EF'}

    ))

fig.add_trace(
    go.Scatter(
        x=df_categories_app_count['Category'].values.tolist(),
        y=df_categories_app_rating['mean'].values.tolist(),
        name ="Average rating",
        marker = {'color':"#A6ACEC"})
    )

fig.add_trace(
    go.Scatter(
        x=df_categories_app_count['Category'].values.tolist(),
        y=df_categories_app_installations['mean'].values.tolist(),
        name = "Average Installations",
        marker={'color': '#8AFF8A'}
    ))
fig.update_yaxes( type="log")
fig.update_layout(
    title="Category wise app count, downloads count, rating",
    yaxis_title="count/rating",
    xaxis_title="Category",
    legend_title="",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="RebeccaPurple"
    )
)

st.set_page_config(layout="wide")

header = st.container()

title_alignment="""
<style>
.block-container{
    padding-top:30px;
}
#android-app-analyst {
  text-align:center;
}
.user-select-none{
    height:200px !important;
}
data-testid{
    width : 200px;
    overflow:scroll;
}
</style>
"""
st.markdown(title_alignment, unsafe_allow_html=True)

with header:
    st.title("App Analyzer (Capstone Project DATA 606)")


def return_description(text):
    return text
row1 = st.container()
row2 = st.container()

with row1:
    row1Container = st.container()
    with row1Container:
        col1, col2 = st.columns(2)

        with col1:
            row1col1container = st.container()
            with row1col1container:
                rowsub1,rowsub2 = st.container(),st.container()
                with rowsub1:
                    colsub1,colsub2 = st.columns(2)
                    with colsub1:
                        app_name = st.text_input(
                            "App Name",
                            "Photo editor",
                            key="placeholder",
                        )
                with rowsub2:
                    app_description = st.text_area("App description","photot editing app named canva")
                    st.plotly_chart(fig, use_container_width=True)
                if app_name or app_description:
                    print(return_description(app_name + app_description))
                    installs = random.randint(1,1000)
                    rating = 4.5
                    
                    
                    installs = []
                    ratings = []
                    type = []
                    users = []
                    category = []
                    price = []

                    if(app_name != ""):
                        app_name_prediction = get_classifications(app_name)
                        app_name_similar_apps = get_similar_apps(app_name)


                        
                    if(app_description != ""):
                        app_description_prediction = get_classifications(app_description)
                        app_description_similar_apps = get_similar_apps(app_description)
                    
                    installs.append(int(app_name_prediction[3]))
                    installs.append(int(app_description_prediction[3]))
                    ratings.append(float(app_name_prediction[2]))
                    ratings.append(float(app_description_prediction[2]))
                    type.append(app_name_prediction[4])
                    type.append(app_description_prediction[4])
                    price.append(float(app_name_prediction[5]))
                    price.append(float(app_description_prediction[5]))
                    category.append(app_name_prediction[6])
                    category.append(app_description_prediction[6])
                    users.append(app_name_prediction[7])
                    users.append(app_description_prediction[7])
                    
                    for _,x in enumerate(app_name_similar_apps):
                        installs.append(int(x[0].split("||")[3]))
                        installs.append(int(app_description_similar_apps[_][0].split("||")[3]))
                        ratings.append(float(x[0].split("||")[2]))
                        ratings.append(float(app_description_similar_apps[_][0].split("||")[2]))
                        type.append(x[0].split("||")[4])
                        type.append(app_description_similar_apps[_][0].split("||")[4])
                        price.append(float(x[0].split("||")[5]))
                        price.append(float(app_description_similar_apps[_][0].split("||")[5]))
                        category.append(x[0].split("||")[6])
                        category.append(app_description_similar_apps[_][0].split("||")[6])
                        users.append(x[0].split("||")[7])
                        users.append(app_description_similar_apps[_][0].split("||")[7])
                    
                    installs = sum(installs)/len(installs)
                    ratings = sum(ratings)/len(ratings)
                    type = most_frequent(type)
                    users = most_frequent(users)
                    category = most_frequent(category)
                    price = sum(price)/len(price)
                    if price < 0.5:
                        price = 0.0
                        type = 'Free'
                    else:
                        type = "Paid"
                    fig1 = get_app_name_accuracy_plot(app_name_prediction[-1]*100-2)
                    fig2 = get_description_accuracy_plot(app_description_prediction[-1]*100-2)

    with col2:
        col2rowsub1 = st.container()
        col2rowsub2 = st.container()
        with col2rowsub1:
            
            colsubsub1,colsubsub2,colsubsub3,colsubsub4 = st.columns(4)
            with colsubsub1:
                st.subheader("Predicted")
                st.text(" Installs     : ")
                st.text(" Rating       : ")
                st.text(" App type     : ")
                st.text(" App Price    : ")
                st.text(" Target Users : ")
                st.text(" App Category : ")
            with colsubsub2:
                st.subheader(":")
                st.text( str(math.ceil(installs)))
                st.text( str(rating))
                st.text(type)
                st.text(price)
                st.text(users)
                st.text("\n".join(category.split(" ")))
            with colsubsub3:
                st.plotly_chart(fig1, use_container_width=True)
                st.text("\tApp Name based \nclassification \n model scores")
                
            with colsubsub4:
                st.plotly_chart(fig2, use_container_width=True)
                st.text("\tApp description \nbased \nclassification \nmodel scores")
                

        with col2rowsub2:
            st.subheader("Best matched App")
            for x in range(1):
                app_card = st.container()
                with app_card:
                    app_card_col1,app_card_col2,app_card_col3,app_card_col4,app_card_col5 = st.columns(5)
                    with app_card_col1:
                        st.image(imagePath,width=80)
                    with app_card_col2:
                        st.text(app_name_prediction[0] + "\n" +"rating: "+app_name_prediction[2] + "\n" + "category: "+app_name_prediction[6])
                        
                    with app_card_col4:
                        st.image(imagePath,width=80)
                    with app_card_col5:
                        st.text(app_description_prediction[0] + "\n" +"rating: "+app_description_prediction[2] + "\n" + "category: "+app_description_prediction[6])
            st.subheader("Similar Apps")
            for x in range(3):
                app_card = st.container()
                with app_card:
                    # image = Image.open(imagePath)
                    app_card_col1,app_card_col2,app_card_col3,app_card_col4,app_card_col5 = st.columns(5)
                    with app_card_col1:
                        st.image(imagePath,width=80)
                    with app_card_col2:
                        print("apps details:",app_description_similar_apps[x])
                        st.text(app_name_similar_apps[x][0].split("||")[0] + "\n" +"rating: "+ app_name_similar_apps[x][0].split("||")[2]+ "\n" + "category: "+app_name_similar_apps[x][0].split("||")[6])
                        
                    with app_card_col4:
                        st.image(imagePath,width=80)
                    with app_card_col5:
                        
                        st.text(app_description_similar_apps[x][0].split("||")[0] + "\n" +"rating: "+ app_description_similar_apps[x][0].split("||")[2]+ "\n" + "category: "+app_description_similar_apps[x][0].split("||")[6])


