import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as maplib
import matplotlib.pyplot as plt
import plotly.express as px

st.title("Macroeconomic information of the countries included in the study*")
st.sidebar.title("Macroeconomic information of certain countries*")

countries_df = pd.read_csv('info_countries.csv')
st.sidebar.subheader("Latest information about the countries")
# last_data = st.sidebar.button("Get the last information of all countries")
if st.sidebar.button("Show the last information of certain countries**"):
    st.subheader("The last information of certain countries**")
    st.write(countries_df)
    if st.sidebar.button("Hide the information"):
        not st.write(countries_df)

# Australia
australia_df = pd.read_csv('Australia.csv')
australia_df["Country"] = "Australia"
australia_df["countryterritoryCode"] = "AUS"
# Austria
austria_df = pd.read_csv('Austria.csv')
austria_df["Country"] = "Austria"
austria_df["countryterritoryCode"] = "AUT"
# Brazil
brazil_df = pd.read_csv('Brazil.csv')
brazil_df["Country"] = "Brazil"
brazil_df["countryterritoryCode"] = "BRA"
# Germany
germany_df = pd.read_csv('Germany.csv')
germany_df["Country"] = "Germany"
germany_df["countryterritoryCode"] = "DEU"
# Russia
russia_df = pd.read_csv('Russia.csv')
russia_df["Country"] = "Russia"
russia_df["countryterritoryCode"] = "RUS"
# USA
usa_df = pd.read_csv('USA.csv')
usa_df["Country"] = "USA"
usa_df["countryterritoryCode"] = "US"

data_all = [australia_df, austria_df, brazil_df, germany_df, usa_df, russia_df]
df_all = pd.concat(data_all, ignore_index=True)

gdp_df = pd.read_csv('GDP_.csv')
inflation_df = pd.read_csv('Inflation-rate_.csv')
unemployment_df = pd.read_csv('Unemployment-rate.csv')


## Visualization 
st.sidebar.markdown("### Visualisation")
select = st.sidebar.selectbox('Visualisation type', ['Chose an option', 'Line Graph', 'Bar Chart', 'Map'], key='1')

#Visualization Line Graph
if select == "Line Graph":
    st.sidebar.subheader("What time period are you interested in?")
    year = st.sidebar.slider(
        'Select a range of years',
        1999, 2021, (2001, 2005))
    st.sidebar.write('Chosen range:', year)
    st.sidebar.subheader("Select countries")
    choice = st.sidebar.multiselect('Pick countries', ('Australia', 'Austria', 'Brazil', 'Germany', 'Russia', 'USA'),
                                    key='0')
    st.sidebar.subheader("Select indicators")
    indicator_choice = st.sidebar.multiselect('Pick indicators', ('GDP', 'Inflation rate ', 'Unemployment Rate '),
                                            key='0')

    if len(indicator_choice) > 0 and len(choice) > 0:
        result_df = df_all[(df_all.Country.isin(choice)) & (df_all["Year"] >= year[0]) & (df_all["Year"] <= year[1])][[
           'GDP', 'Inflation rate ', 'Unemployment Rate ', "Year", "Country"]]

        for indicator in indicator_choice: 
            try: 
                result_df[indicator] = result_df[indicator].map(lambda x:  x if isinstance(x, float) else float(x.replace('%', '')))
            except AttributeError:
                print("not string, continue....")
    
            fig =px.line(result_df, x='Year', y=indicator, color="Country", height=500)
            st.plotly_chart(fig)

#Visualization Bar Chart
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
    indicator_choice = st.sidebar.multiselect('Pick indicators', ('GDP', 'Inflation rate ', 'Unemployment Rate '),
                                            key='0')

    if len(indicator_choice) > 0 and len(choice) > 0:
        result_df = df_all[(df_all.Country.isin(choice)) & (df_all["Year"] >= year[0]) & (df_all["Year"] <= year[1])][
            ['GDP', 'Inflation rate ', 'Unemployment Rate ', "Year", "Country"]]

        for indicator in indicator_choice: 
            try: 
                result_df[indicator] = result_df[indicator].map(lambda x:  x if isinstance(x, float) else float(x.replace('%', '')))
            except AttributeError:
                print("not string, continue....")  
            fig = px.bar(result_df, x='Year', y=indicator, color="Country", height=500, barmode='group')
            st.plotly_chart(fig)

# Visualization Map
if select == "Map":
    st.sidebar.subheader("Select indicators")
    indicator_choice = st.sidebar.multiselect('Pick indicators', ('GDP', 'Inflation rate ', 'Unemployment Rate '),
                                            key='0')

    if len(indicator_choice) > 0:
        
        for indicator in indicator_choice: 
            try: 
                df_all[indicator] = df_all[indicator].map(lambda x:  x if isinstance(x, float) else float(x.replace('%', '')))
            except AttributeError:
                print("not string, continue....")  
            fig = px.choropleth(df_all, locations="countryterritoryCode",
                        color=indicator,
                        hover_name="Country",
                        animation_frame="Year",
                        color_continuous_scale="Sunsetdark",
                        projection = 'equirectangular')
            st.plotly_chart(fig)
      
if st.sidebar.checkbox("Show Raw Data", False):
    raw_data = st.sidebar.radio('Indicator', ('GDP', 'Inflation rate ', 'Unemployment Rate '))
    if raw_data == "GDP":
        st.write(gdp_df)
    if raw_data == "Inflation rate ":
        st.write(inflation_df)
    if raw_data == "Unemployment Rate ":
        st.write(unemployment_df)



st.sidebar.caption("<monospace color = grey;>*This web application is a Streamlit dashboard for displaying and analyzing information from Australia, Austria, Brazil, Germany, Russia, USA on their macroeconomic indicators</monospace>", unsafe_allow_html=True)
st.sidebar.caption("<monospace color = grey;>** Australia, Austria, Brazil, Germany, Russia, USA </monospace>", unsafe_allow_html=True)
st.sidebar.caption("<monospace color = grey;>*** GDP, Inflation Rate, Unemployment Rate </monospace>", unsafe_allow_html=True)
st.markdown(
    " This web application is a Streamlit dashboard for displaying and analyzing information from certain countries on their macroeconomic indicators.")
