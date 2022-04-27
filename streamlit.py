import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as maplib
import matplotlib.pyplot as plt
import plotly.express as px

st.title("Economic Indicators of countries*")
st.sidebar.title("Economic Indicators of countries")
st.markdown(
    " *This application is a Streamlit dashboard to show and analyse the economic indicators of Australia, Austria, Brazil, Germany, Russia, USA")
st.sidebar.markdown(
    " This application is a Streamlit dashboard to show and analyse the economic indicators of certain countries")

countries_df = pd.read_csv('info_countries.csv')
st.sidebar.subheader("Show random country's information")
# random_tweeet = st.sidebar.radio('Sentiment', ('positive', 'neutral', 'negative'))
# last_data = st.sidebar.button("Get the last information of all countries")
if st.sidebar.button("Get the last information of all countries"):
    st.subheader("The last information of all* countries")
    st.write(countries_df)
    if st.sidebar.button("Remove the information"):
        not st.write(countries_df)

# st.sidebar.markdown(data.query('airline_sentiment == @random_tweet')[["text"]].sample(n=1).iat[0,0])

# Australia
australia_df = pd.read_csv('Australia.csv')
australia_df["Country"] = "Australia"
australia_df["countryterritoryCode"] = "AU"
# Austria
austria_df = pd.read_csv('Austria.csv')
austria_df["Country"] = "Austria"
austria_df["countryterritoryCode"] = "AT"
# Brazil
brazil_df = pd.read_csv('Brazil.csv')
brazil_df["Country"] = "Brazil"
brazil_df["countryterritoryCode"] = "BR"
# Germany
germany_df = pd.read_csv('Germany.csv')
germany_df["Country"] = "Germany"
germany_df["countryterritoryCode"] = "DE"
# Russia
russia_df = pd.read_csv('Russia.csv')
russia_df["Country"] = "Russia"
russia_df["countryterritoryCode"] = "RU"
# USA
usa_df = pd.read_csv('USA.csv')
usa_df["Country"] = "USA"
usa_df["countryterritoryCode"] = "US"

data_all = [australia_df, austria_df, brazil_df, germany_df, usa_df, russia_df]
df_all = pd.concat(data_all, ignore_index=True)

gdp_df = pd.read_csv('GDP_.csv')
inflation_df = pd.read_csv('Inflation-rate_.csv')
unemployment_df = pd.read_csv('Unemployment-rate.csv')

# st.write(gdp_df)
# st.write(inflation_df)
# st.write(unemployment_df)
# st.write(australia_df)
# st.write(austria_df)
# st.write(brazil_df)
# st.write(germany_df)
# st.write(russia_df)
# st.write(USA_df)

st.sidebar.markdown("### Visualisation")
select = st.sidebar.selectbox('Visualisation type', ['Chose an option', 'Line Graph', 'Bar Chart', 'Map'], key='1')
# sentiment_count = data['airline_sentiment'].value_counts()
# sentiment_count = pd.DataFrame({'Sentiment':sentiment_count.index, 'Tweets':sentiment_count.values})



# if select == "Line Graph":
#     st.sidebar.subheader("Choose an indicator")
#     lg_indicator = st.sidebar.radio('Indicator', ('GDP', 'Infaltion Rate', 'Unemployment Rate'))
#     if lg_indicator == "GDP":
#         st.sidebar.subheader("What timeperiod are you interested in?")
#         year = st.sidebar.slider(
#         'Select a range of years',
#         1999, 2021, (2001, 2005))
#         st.sidebar.write('Chosen:', year)
#         st.sidebar.subheader("Select countries")
#         choice = st.sidebar.multiselect('Pick countries', ('Australia', 'Austria', 'Brazil', 'Germany', 'Russia', 'USA'), key='0')
#         if lg_indicator == "GDP" and len(choice) > 0: 
#             result_df=df_all[(df_all.Country.isin(choice)) & (df_all["Year"]>=year[0]) & (df_all["Year"]<=year[1])][[lg_indicator,"Year","Country"]]
#             fig = px.line(result_df, x='Year', y=lg_indicator[0], color="Country", height = 500)
#             st.plotly_chart(fig)
#         # st.sidebar.subheader("Select indicators")
#         # indicator_choice = st.sidebar.multiselect('Pick indicators', ('GDP', 'Inflation rate ', 'Unemployment rate '), key='0') 


if select == "Line Graph":
    st.sidebar.subheader("What time period are you interested in?")
    year = st.sidebar.slider(
        'Select a range of years',
        1999, 2021, (2001, 2005))
    st.sidebar.write('Chosen:', year)
    st.sidebar.subheader("Select countries")
    choice = st.sidebar.multiselect('Pick countries', ('Australia', 'Austria', 'Brazil', 'Germany', 'Russia', 'USA'),
                                    key='0')
    st.sidebar.subheader("Select indicators")
    indicator_choice = st.sidebar.multiselect('Pick indicators', ('GDP', 'Inflation rate ', 'Unemployment rate '),
                                              key='0')

    print(indicator_choice)
    print(choice)

    if len(indicator_choice) > 0 and len(choice) > 0:
        result_df = df_all[(df_all.Country.isin(choice)) & (df_all["Year"] >= year[0]) & (df_all["Year"] <= year[1])][
            [indicator_choice[0], "Year", "Country"]]

        print(indicator_choice[0])

        try:
            result_df[indicator_choice[0]] = result_df[indicator_choice[0]].map(lambda x: float(x.replace('%', '')))
        except AttributeError:
            print("not string, continue....")
        print(result_df)

        fig = px.line(result_df, x='Year', y=indicator_choice[0], color="Country", height=500)
        st.plotly_chart(fig)

if select == "Bar Chart":
    st.sidebar.subheader("What time period are you interested in?")
    year = st.sidebar.slider(
        'Select a range of years',
        1999, 2021, (2001, 2005))
    st.sidebar.write('Chosen:', year)
    st.sidebar.subheader("Select countries")
    choice = st.sidebar.multiselect('Pick countries', ('Australia', 'Austria', 'Brazil', 'Germany', 'Russia', 'USA'),
                                    key='0')
    st.sidebar.subheader("Select indicators")
    indicator_choice = st.sidebar.multiselect('Pick indicators', ('GDP', 'Inflation rate', 'Unemployment rate'),
                                              key='0')

    if len(indicator_choice) > 0 and len(choice) > 0:
        result_df = df_all[(df_all.Country.isin(choice)) & (df_all["Year"] >= year[0]) & (df_all["Year"] <= year[1])][
            [indicator_choice[0], "Year", "Country"]]

        fig = px.bar(result_df, x='Year', y=indicator_choice[0], color="Country", height=500)
        st.plotly_chart(fig)



# GDP карта без цвета, inflation and unemployment ERROR 
if select == "Map":
    st.sidebar.subheader("What time period are you interested in?")
    year = st.sidebar.slider(
        'Select a range of years',
        1999, 2021, (2001, 2005))
    st.sidebar.write('Chosen:', year)
    st.sidebar.subheader("Select countries")
    choice = st.sidebar.multiselect('Pick countries', ('Australia', 'Austria', 'Brazil', 'Germany', 'Russia', 'USA'),
                                    key='0')
    st.sidebar.subheader("Select indicators")
    indicator_choice = st.sidebar.multiselect('Pick indicators', ('GDP', 'Inflation rate', 'Unemployment rate'),
                                              key='0')

    if len(indicator_choice) > 0 and len(choice) > 0:
        result_df = df_all[(df_all.Country.isin(choice)) & (df_all["Year"] >= year[0]) & (df_all["Year"] <= year[1])][
            [indicator_choice[0], "Year", "Country"]]

        
        fig = px.choropleth(df_all, locations="countryterritoryCode",
                    color=indicator_choice[0],
                    hover_name="Country",
                    animation_frame="Year",
                    color_continuous_scale="Sunsetdark",
                    projection = 'equirectangular')
        st.plotly_chart(fig)


# fig.update_geos(fitbounds="locations")
# fig.update_layout(margin={'r':0,'t':50,'l':0,'b':0})
# fig.show()

# fig = px.line(df[filter1], x="Years", y=indicator_choice[0], color="Country")
# fig.show()
        
        
