
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Pandas settings
pd.set_option("display.max_columns", None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)

# Load the data
df=pd.read_csv("venv/data-sets/covid_19.csv")

# Check for missing values
print(df.isnull().sum())
# Check the shape of the DataFrame
print(df.shape)

# Remove data with less than 100 confirmed cases or deaths
df = df[~(df.Confirmed < 100) & ~(df.Confirmed < 100)]

# Visualize missing values
sns.heatmap(df.isnull())

# Columns in the DataFrame
print(df.columns)
# General information about our data
df.groupby(["Region"]).agg({"Confirmed": ["sum","min","mean","max"],
                            "Deaths": ["sum","min","mean","max"],
                            "Recovered": ["sum","min","mean","max"]}).head(6)


# Remove data with less than 10 confirmed cases or deaths
df = df[~(df.Confirmed < 100) & ~(df.Confirmed < 100)]


######################################
# Before comparing specific values, let's create a graph for a general conclusion.
selected_regions = random.sample(df["Region"].tolist(), 5)
result = df[df["Region"].isin(selected_regions)].groupby("Region").agg({
    "Confirmed": ["sum", "mean"],
    "Deaths": ["sum", "mean"],
    "Recovered": ["sum", "mean"]
})
# Chart Creation
sns.set(style="whitegrid")
fig, axes = plt.subplots(3, 1, figsize=(6, 8))

sns.barplot(x=result["Confirmed"]["sum"], y=result.index, color="blue", ax=axes[0])
axes[0].set_title("COVID-19 Confirmed Cases by Region")
axes[0].set_xlabel("Confirmed Counts")
axes[0].set_ylabel("Regions")

sns.barplot(x=result["Deaths"]["sum"], y=result.index, color="red", ax=axes[1])
axes[1].set_title("COVID-19 Deaths by Region")
axes[1].set_xlabel("Death Counts")
axes[1].set_ylabel("Regions")

sns.barplot(x=result["Recovered"]["sum"], y=result.index, color="green", ax=axes[2])
axes[2].set_title("COVID-19 Cases by Region")
axes[2].set_xlabel("Recovered Counts")
axes[2].set_ylabel("Regions")
plt.tight_layout()
plt.show()


######################################
# Create a pie chart showing where deaths occurred the most
label = df.groupby("Region")["Deaths"].sum().sort_values(ascending=False).head(5)
region_labels = label.index

plt.figure(figsize=(6, 6))
plt.pie(label, labels=region_labels, startangle=90, shadow=True, autopct="%1.1f%%")
plt.title("Graphic - Rates", fontsize=32)
plt.show()
# --> As it is understood here, the countries with the highest deaths as a result of covid19 are US>Italy>UK>Spain>France
# It goes like this.


############################
# Create a bar chart showing death rates by region for the top 10 regions
df["Death Rate"] = (df["Deaths"] / df["Confirmed"]) * 100
# Let's limit the data to 10 regions
top_10_regions = df.groupby("Region")["Death Rate"].mean().nlargest(10).index
filtered_df = df[df["Region"].isin(top_10_regions)]

plt.figure(figsize=(12, 6))
sns.barplot(x="Region", y="Death Rate", data=filtered_df, palette="viridis")
plt.title('Top 10 Regions with Highest Death Rates', size=16)
plt.xticks(rotation=45)
plt.show()

# --> Here, in addition to the number of cases, we also show the intensity
# of deaths in countries relative to the cross-country average. It shows.



######################################
# Create a heatmap to visualize the correlation between deaths, confirmed cases, and recoveries.
correlation_matrix = df.loc[:, ["Deaths", "Confirmed", "Recovered"]].corr()

sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.xlabel("Features")
plt.ylabel("Features")
plt.title("Correlation Matrix Heatmap")
plt.show()

# --> It is clear from this that 90% of the number of recorded covid19 cases and deaths is very high.
# there is a big connection. According to the data, most of the confirmed cases resulted in death.


######################################
# Pie chart to compare total deaths with deaths in the top 5 countries
label_index = df.groupby("Region")["Deaths"].sum().sort_values(ascending=False).head(5).index
filter = df["Region"].isin(label_index)
total_without_top5 = np.sum(df.loc[filter, "Deaths"]) - np.sum(df.loc[~filter, "Deaths"])

total_deaths = np.sum(df["Deaths"])

label2= total_deaths, total_without_top5
plt.figure(figsize=(6,6))
plt.pie(label2,labels=["Total","Top 5 Deaths Countries"],startangle=90,shadow=True, autopct="%1.1f%%")
plt.title("Graphic - Rates",fontsize=32)
plt.show()

# -->In this inference, the countries with the top 5 deaths account for a very high proportion of
# deaths, such as %30.4.


######################################
# Visualize daily cases and changes over time.

df["Date"].dtype
df["Date"]=df["Date"].astype("O")
date_status=df["Date"].value_counts()

df["Date"]=pd.to_datetime(df["Date"])
random_days = np.random.randint(1,31,size=len(df))
df["Date"] = df["Date"] + pd.to_timedelta(random_days, unit="D")

df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)
weekly_data = df.resample('W').sum(numeric_only=True)

df = df.sort_values(by="Date")

daily_cases = df.groupby("Date")["Confirmed"].sum().diff().fillna(0)
daily_cases[daily_cases < 0] = 0


plt.figure(figsize=(9, 9))
plt.plot(daily_cases.index, daily_cases.values, linestyle='dashed', color='orange')
plt.title('Daily Coronavirus Cases Over Time', size=17)
plt.xlabel('Date', size=15)
plt.ylabel('Daily Cases', size=15)
plt.xticks(rotation=90, size=5)
plt.yticks(size=10)
plt.show()

# --> This graph shows that during the corona virus period, the amount
# of litigation on each day varies and shows that there is no stable progression.











































