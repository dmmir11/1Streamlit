import pandas as pd

#Australia
australia_df = pd.read_csv('Australia.csv')
australia_df["Country"]="Australia"
#Austria
austria_df = pd.read_csv('Austria.csv')
austria_df["Country"]="Austria"
#Brazil 
brazil_df = pd.read_csv('Brazil.csv')
brazil_df["Country"]="Brazil"
#Germany
germany_df = pd.read_csv('Germany.csv')
germany_df["Country"]="Germany"
#Russia
russia_df = pd.read_csv('Russia.csv')
russia_df["Country"]="Russia"
#USA
usa_df = pd.read_csv('USA.csv')
usa_df["Country"]="USA"
# print(usa_df.columns)
# print(russia_df.columns)

data = [australia_df, austria_df, brazil_df, germany_df, usa_df, russia_df]
df2 = pd.concat(data,ignore_index=True)

#print(df2[(df2["Country"]=="USA") & (df2["Year"]>=1999) & (df2["Year"]<=2021)][["GDP","Year"]])


# gdp_df = pd.read_csv('GDP_.csv')
# inflation_df = pd.read_csv('Inflation-rate_.csv')
# unemployment_df = pd.read_csv('Unemployment-rate.csv')

# data = [gdp_df, inflation_df, unemployment_df]
# df3 = pd.concat(data,ignore_index=True)


df2[df2.columns[1:]] = df2[df2.columns[1:]].replace('[%,]', '', regex=True) 
df2["Inflation Rate"] = pd.to_numeric(df2["Inflation Rate"])
df2["Unemployment Rate"] = pd.to_numeric(df2["Unemployment Rate"])

# df2["Austria"] = pd.to_numeric(df2["Austria"])

# df2["Brazil"] = pd.to_numeric(df2["Brazil"])

# df2["Germany"] = pd.to_numeric(df2["Germany"])

# df2["Russia"] = pd.to_numeric(df2["Russia"])

# df2["USA"] = pd.to_numeric(df2["USA"])







print(df2)
