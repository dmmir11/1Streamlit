import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as maplib
import matplotlib.pyplot as plt
import plotly.express as px

st.title("Economic Indicators of countries*")
st.sidebar.title("Economic Indicators of countries")
st.markdown(" *This application is a Streamlit dashboard to show and analyse the economic indicators of Australia, Austria, Brazil, Germany, Russia, USA")
st.sidebar.markdown("This application is a Streamlit dashboard to show and analyse the economic indicators of certain countries")

countries_df = pd.read_csv('info_countries.csv')
st.sidebar.subheader("Show random country's information")
st.sidebar.subheader("Show last information of all countries")
#last_data = st.sidebar.button("Get the last information of all countries")
if st.sidebar.button("Get the last information of all countries"):
    st.subheader("The last information of all* countries")
    st.write(countries_df)
    if st.sidebar.button("Remove the information"):
        not st.write(countries_df)


#st.sidebar.markdown(data.query('airline_sentiment == @random_tweet')[["text"]].sample(n=1).iat[0,0])

#Australia
australia_df = pd.read_csv('Australia1.csv')
australia_df["Country"]="Australia"
#Austria
austria_df = pd.read_csv('Austria1.csv')
austria_df["Country"]="Austria"
#Brazil 
brazil_df = pd.read_csv('Brazil1.csv')
brazil_df["Country"]="Brazil"
#Germany
germany_df = pd.read_csv('Germany1.csv')
germany_df["Country"]="Germany"
#Russia
russia_df = pd.read_csv('Russia1.csv')
russia_df["Country"]="Russia"
#USA
usa_df = pd.read_csv('USA1.csv')
usa_df["Country"]="USA"

data_all = [australia_df,austria_df, brazil_df, germany_df, usa_df, russia_df]
df_all= pd.concat(data_all,ignore_index=True)

gdp_df = pd.read_csv('GDP_.csv')
inflation_df = pd.read_csv('Inflation-rate_.csv')
unemployment_df = pd.read_csv('Unemployment-rate.csv')



#REPLACE % 
#df2[df2.columns[1:]] = df2[df2.columns[1:]].replace('[%,]', '', regex=True)


# st.write(gdp_df)
# st.write(inflation_df)
# st.write(unemployment_df)
# st.write(australia_df)
# st.write(austria_df)
# st.write(brazil_df)
# st.write(germany_df)
# st.write(russia_df)
# st.write(USA_df)

#st.sidebar.markdown(data.query('airline_sentiment == @random_tweet')[["text"]].sample(n=1).iat[0,0])




st.sidebar.markdown("### Visualisation")
select = st.sidebar.selectbox('Visualisation type', ['Chose an option', 'Line Graph', 'Bar Chart', 'Map'], key='1')
# sentiment_count = data['airline_sentiment'].value_counts()
# sentiment_count = pd.DataFrame({'Sentiment':sentiment_count.index, 'Tweets':sentiment_count.values})

if select == "Line Graph":
    st.sidebar.subheader("Choose an indicator")
    lg_indicator = st.sidebar.radio('Indicator', ('GDP', 'Infaltion Rate', 'Unemployment Rate'))
    if lg_indicator == "GDP":
        st.sidebar.subheader("What timeperiod are you interested in?")
        year = st.sidebar.slider(
        'Select a range of years',
        1999, 2021, (2001, 2005))
        st.sidebar.write('Chosen:', year)
        st.sidebar.subheader("Select countries")
        choice = st.sidebar.multiselect('Pick countries', ('Australia', 'Austria', 'Brazil', 'Germany', 'Russia', 'USA'), key='0')
        if lg_indicator == "GDP" and len(choice) > 0: 
            result_df=df_all[(df_all.Country.isin(choice)) & (df_all["Year"]>=year[0]) & (df_all["Year"]<=year[1])][[lg_indicator,"Year","Country"]]
            fig = px.line(result_df, x='Year', y=lg_indicator[0], color="Country", height = 500)
            st.plotly_chart(fig)
        # st.sidebar.subheader("Select indicators")
        # indicator_choice = st.sidebar.multiselect('Pick indicators', ('GDP', 'Inflation rate ', 'Unemployment rate '), key='0') 


    # st.sidebar.subheader("What timeperiod are you interested in?")
    # year = st.sidebar.slider(
    #  'Select a range of years',
    #  1999, 2021, (2001, 2005))
    # st.sidebar.write('Chosen:', year)
    # st.sidebar.subheader("Select countries")
    # choice = st.sidebar.multiselect('Pick countries', ('Australia', 'Austria', 'Brazil', 'Germany', 'Russia', 'USA'), key='0')
    # st.sidebar.subheader("Select indicators")
    # indicator_choice = st.sidebar.multiselect('Pick indicators', ('GDP', 'Inflation rate ', 'Unemployment rate '), key='0')

#     print(indicator_choice)
#     print(choice)

#     if len(indicator_choice)>0 and len(choice) > 0: 
#         result_df=df_all[(df_all.Country.isin(choice)) & (df_all["Year"]>=year[0]) & (df_all["Year"]<=year[1])][[indicator_choice[0],"Year","Country"]]

#         fig = px.line(result_df, x='Year', y=indicator_choice[0], color="Country", height = 500)
#         st.plotly_chart(fig)

# if select == "Bar Chart":
#     st.sidebar.subheader("What timeperiod are you interested in?")
#     year = st.sidebar.slider(
#      'Select a range of years',
#      1999, 2021, (2001, 2005))
#     st.sidebar.write('Chosen:', year)
#     st.sidebar.subheader("Select countries")
#     choice = st.sidebar.multiselect('Pick countries', ('Australia', 'Austria', 'Brazil', 'Germany', 'Russia', 'USA'), key='0')
#     st.sidebar.subheader("Select indicators")
#     indicator_choice = st.sidebar.multiselect('Pick indicators', ('GDP', 'Inflation rate', 'Unemployment rate'), key='0')

    # fig = px.line(, x='', y='', color='', height = 500)
    #     st.plotly_chart(fig)

# st.sidebar.subheader("Select countries")
# choice = st.sidebar.multiselect('Pick countries', ('Australia', 'Austria', 'Brazil', 'Germany', 'Russia', 'USA'), key='0')
# st.sidebar.subheader("Select indicators")
# indicator_choice = st.sidebar.multiselect('Pick indicators', ('GDP', 'Inflation rate', 'Unemployment rate'), key='0')






# if not st.sidebar.checkbox("Close", True, key='1'):
#     st.markdown("### Tweets locations based on the time of day")
#     st.markdown("%i tweets between %i:00 and %i:00" % (len(modified_data), hour, (hour+1)%24))
#     st.map(modified_data)
#     if st.sidebar.checkbox("Show raw data", False):
#         st.write(modified_data)




# if not st.sidebar.checkbox("Hide", True):
#     st.markdown("### Visualisation of countries")
#     if select == "Histogram":
#         fig = px.bar(sentiment_count, x='Sentiment', y='Tweets', color='Tweets', height = 500)
#         st.plotly_chart(fig)
#     else:
#         fig = px.pie(sentiment_count, values='Tweets', names='Sentiment')
#         st.plotly_chart(fig)







#SHOW changes through the year 
# fig = px.choropleth(df_geo, locations="countryterritoryCode",
#                     color="inflation_rate",
#                     hover_name="geo",
#                     animation_frame="years",
#                     title = "Changes of __choosen indicator with the time_",
#                     color_continuous_scale="Sunsetdark",
#                     projection = 'equirectangular')

# fig.update_geos(fitbounds="locations")
# fig.update_layout(margin={'r':0,'t':50,'l':0,'b':0})
# fig.show() 

#st.sidebar.subheader("What timeperiod are you interested in?")

#year = st.sidebar.number_input("Years",min_value=1, max_value=24)
# modified_data = data[data['tweet_created'].dt.hour == hour]
# if not st.sidebar.checkbox("Close", True, key='1'):
#     st.markdown("### Tweets locations based on the time of day")
#     st.markdown("%i tweets between %i:00 and %i:00" % (len(modified_data), hour, (hour+1)%24))
#     st.map(modified_data)
#     if st.sidebar.checkbox("Show raw data", False):
#         st.write(modified_data)




# st.sidebar.markdown("### Number of tweets by sentiment")
# select = st.sidebar.selectbox('Visualisation type', ['Histogram', 'Pie chart'], key='1')
# sentiment_count = data['airline_sentiment'].value_counts()
# sentiment_count = pd.DataFrame({'Sentiment':sentiment_count.index, 'Tweets':sentiment_count.values})

# if not st.sidebar.checkbox("Hide", True):
#     st.markdown("### Number of Tweets by sentiment")
#     if select == "Histogram":
#         fig = px.bar(sentiment_count, x='Sentiment', y='Tweets', color='Tweets', height = 500)
#         st.plotly_chart(fig)
#     else:
#         fig = px.pie(sentiment_count, values='Tweets', names='Sentiment')
#         st.plotly_chart(fig)


# st.sidebar.subheader("When and where are users tweeting from?")
# hour = st.sidebar.slider("Hour of day", 0, 23)
# #hour = st.sidebar.number_input("Hour of day",min_value=1, max_value=24)
# modified_data = data[data['tweet_created'].dt.hour == hour]
# if not st.sidebar.checkbox("Close", True, key='1'):
#     st.markdown("### Tweets locations based on the time of day")
#     st.markdown("%i tweets between %i:00 and %i:00" % (len(modified_data), hour, (hour+1)%24))
#     st.map(modified_data)
#     if st.sidebar.checkbox("Show raw data", False):
#         st.write(modified_data)

# st.sidebar.subheader("Breakdown irline tweets by sentiment")
# choice = st.sidebar.multiselect('Pick airlines', ('Us Airways', 'United', 'American', 'Southwest', 'Delta', 'Virgin America'), key='0')

# if len(choice) > 0:
#     choice_data = data[data.airline.isin(choice)]
#     fig_choice = px.histogram(choice_data, x='airline', y='airline_sentiment', histfunc='count', color='airline_sentiment',
#     facet_col='airline_sentiment', labels={'airline_sentiment':'tweets'}, height=600, width=800)
#     st.plotly_chart(fig_choice)


# st.sidebar.header("Word Cloud")
# word_sentiment = st.sidebar.radio('Display word cloud for what sentiment?', ('positive', 'neutral', 'negative'))

# if not st.sidebar.checkbox("Close", True, key='3'):
#     st.header('Word cloud for %s sentiment' % (word_sentiment))
#     df = data[data['airline_sentiment']==word_sentiment]
#     words = ' '.join(df['text'])
#     processed_words = ' '.join([word for word in words.split() if 'http' not in word and not word.startswith('@') and word != 'RT'])
#     wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', height=640, width=800).generate(processed_words)
#     plt.imshow(wordcloud)
#     plt.xticks([])
#     plt.yticks([])
#     st.pyplot()
