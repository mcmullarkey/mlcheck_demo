import numpy as np
import pandas as pd
import altair as alt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress most warnings in TensorFlow
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
pd.set_option('display.float_format', lambda x: '%.3f' % x)
import streamlit as st
import sklearn
from sklearn.linear_model import LinearRegression
st.set_page_config(layout="wide")
st.balloons()
st.title("Anhong Yang's Math10 Final Project")
st.subheader('2021/12/05')
st.write("student ID:41845042")
st.markdown("[This](https://github.com/Anhong-Yang/Math10) is the link to my repository, and the final project name is FP.py")
st.markdown("The link for [Steam dataset](https://www.kaggle.com/connorwynkoop/steam-monthly-player-data/version/1)")
st.markdown("The link for [Covid DS](https://www.kaggle.com/ifashion/covid-statistics)")
st.subheader("Reference:")
st.markdown("[This](https://stackoverflow.com/questions/44908383/how-can-i-group-by-month-from-a-date-field-using-python-pandas) is the code I used for grouping data by 'Month' in SteamDS")
st.markdown("[This](https://discuss.streamlit.io/t/cannot-change-matplotlib-figure-size/10295/3) is the code I used for reshape the st.pyplot graph")

st.header("My project is to find some relationship between Covid-19 infection and the number of online gamers")

st.subheader("Covid-19 DS")
st.write("First, I will show the whole DS.")
Cov1=pd.read_csv("owid-covid-data.csv",na_values=" ")
Cov1=Cov1.replace(np.nan, 0)
Cov1["date"]=pd.to_datetime(Cov1["date"]) 
Cov1=Cov1[['location', 'date', 'total_cases','total_deaths',"new_cases","new_deaths"]]
Cov1
st.write("As we can see, there are different countries' daily report, we only need the monthly report for the world-wide.")
st.write("The LHS data is data after shaping, but we need to calculate 'new_cases' and 'new_death' by our owns since the data is not correctly match montly, and the calculated data in on the RHS")
Ccol1,Ccol2 = st.columns(2)
with Ccol1:
    Cov1=Cov1[Cov1["location"]=="World"]
    Cov1["end of month"]=Cov1["date"].dt.is_month_end
    Cov1=Cov1.loc[lambda Cov1: Cov1["end of month"]==True]
    Cov1
with Ccol2:
    for i in range(20):
        if i>=1:
            Cov1.iloc[i,4]=Cov1.iloc[i,2]-Cov1.iloc[i-1,2]
            Cov1.iloc[i,5]=Cov1.iloc[i,3]-Cov1.iloc[i-1,3]
        else:
            Cov1.iloc[i,4]=Cov1.iloc[i,2]
            Cov1.iloc[i,5]=Cov1.iloc[i,3]
    Cov1["new_cases"]=Cov1["new_cases"].astype(int)
    Cov1["new_deaths"]=Cov1["new_deaths"].astype(int)
    Cov1["total_cases"]=Cov1["total_cases"].astype(int)
    Cov1["total_deaths"]=Cov1["total_deaths"].astype(int)
    Cov1
Cov1=Cov1.drop(["end of month"],axis=1)
st.write("Let's plot theses data by line")
def CovPic(yvalue):
    CP=alt.Chart(Cov1).mark_line().encode(
        x=alt.X('date',scale=alt.Scale(zero=False)),    
        y=yvalue,
        tooltip=['date',"location",'new_cases',"new_deaths"],
        ).properties(
        title=f'The line graph about date and {yvalue}',
        width = 400,
        height= 400
    )
    return CP
C1=CovPic('new_cases')
C2=CovPic('new_deaths')
C3=CovPic('total_cases')
C4=CovPic('total_deaths')
(C1|C2)&(C3|C4)
st.write("So, there is some linear-like relationship for the 'total_cases'and 'total_cases' after 2020-10")
st.write('Let us try to predict the data for 2021-09')
Cov2=Cov1 # set a copy of Cov1
# Let Cov2 be data from 2020-10 to 2021-08
Cov2['date_last'] = (Cov2['date'] - Cov2.iloc[0,1]).dt.days
Cov2 =Cov2[(Cov2["date"]>="2020-10-1")&(Cov2["date"]<="2021-9-1")]
Cov2
regC1 = LinearRegression()
regC2 = LinearRegression()
regC1.fit(Cov2[["date_last"]],Cov2[["total_cases"]])
regC2.fit(Cov2[["date_last"]],Cov2[["total_deaths"]])
Case9=regC1.predict([[578+30]]).round()
Death9=regC2.predict([[578+30]]).round()
st.subheader("the 'date_last' is how far we are from the end of 2020-01")
st.write(f"The total case in 2021-09 is {Case9}, and total death is {Death9}")
st.markdown("According to the [data](https://ourworldindata.org/explorers/coronavirus-data-explorer?tab=table&zoomToSelection=true&time=2021-09-30&facet=none&uniformYAxis=0&pickerSort=asc&pickerMetric=location&Metric=Confirmed+cases&Interval=Cumulative&Relative+to+Population=false&Align+outbreaks=false&country=~OWID_WRL), the total case is 234.21M and the total death is 4.78M,which are closed to our predict.")
st.write("So I will predict my data in neural network first.")

st.subheader("Stream player DS")
if "df" not in st.session_state:
    df=pd.read_csv("AllSteamData.csv",na_values=" ")
    st.write("the first requirement is recent peak player is > 10000")
    DData=df[(df["Peak Players"]<10000)&(df["Month"]=="Last 30 Days")]
    Dname=list(DData.Name) #get those unqualify game's name
    df["Delete"]=[df.iloc[x,0] in Dname for x in range(len(df))] # set a delet col to determine if it need to be deleted
    df=df.loc[df["Delete"]==False] # only need those qualify game's data 
    #then delete the "Last 30 Days" data
    for i in range(len(df)):
        if df.iloc[i,1]=="Last 30 Days":
            df.iloc[i,6]=True
    df=df.loc[df["Delete"]==False] # only need those qualify game's data 
    df1=df # set a copy of df
    df1["Month"]="1-"+df1["Month"] # add a 1 to make it could be convert to datetime
    df1["Month"]=pd.to_datetime(df1["Month"])
    # lock them from "2020-1-1" to "2021-9-1"
    df1=df1[(df["Month"]>="2020-1-1")&(df1["Month"]<="2021-9-1")]
    st.session_state["df"]=df
    st.session_state["df1"]=df1
else:
    df = st.session_state["df"]
    df1= st.session_state["df1"]
st.write("Here, we import another DS about the game players on Steam,since it is too big for shown, I need to do something to seize the data I want")
df
st.write("since we have the covid data from 2020-01-31 to 2021-8-31, we also want the games with data in this time range,plus the 2021-09.")
st.write("If the netural network works well, then I will try to predict the 2021-09 to see how good the network is.")


# But it is not enough since if some games from 2020-3 to 2021-6 will also be inside
Dbegin=list(df1[(df1["Month"]=="2020-1-1")].Name)# the game name appears in 2020-01 
Dend=list(df1[(df1["Month"]=="2021-9-1")].Name)# the game name appears in 2020-09
df1["Delete"]=[((df1.iloc[x,0] in Dbegin)&(df1.iloc[x,0] in Dend)) for x in range(len(df1))]#
df1=df1[df1["Delete"]==True]
df1["Month"]=df1["Month"].dt.to_period('M').dt.to_timestamp('M')# change the month to the end of the month
df1["Gain"]=pd.to_numeric(df1["Gain"])
st.write("since we do not need the game name, but the total players, we could delete the 'Name' column and convert rows by 'Month'")
df1=df1.drop(["Name","% Gain","Delete"],axis=1) 
df1=df1.groupby([df["Month"]])['Avg. Players',"Gain","Peak Players"].sum() # Sum up all data in the same month
df1["Month"]=df1.index # restore the ["Month"] column
df1["Month"]=df1["Month"].dt.to_period('M').dt.to_timestamp('M')
df1=df1.rename(columns={"Avg. Players": "Avg_Players"})# rename for df1["Avg. Players"]
df1["Inc"]=[df1.iloc[x,1]>0 for x in range(len(df1))] # to predict if the data is increase 
df1["Inc50k"]=[df1.iloc[x,1]>50000 for x in range(len(df1))] # to predict if the data is increase over 50k
df2=df1[df1["Month"]=="2021-9-30"] #set a copy of df1 to get the games in Sep
df1=df1[(df1["Month"]>"2020-1-1")&(df1["Month"]<"2021-9-1")]
df1
df2

# Machine learning
x1c=st.selectbox("please select first input",("new_cases","new_deaths","total_cases","total_deaths"))
x2c=st.selectbox("please select second input",("new_cases","new_deaths","total_cases","total_deaths"))
yc=st.selectbox("please select real input for learning",("Inc","Inc50k"))
# keras learn
xdata=Cov1[[x1c,x2c]].values
y=df1[yc]
model = keras.Sequential(
    [
        keras.layers.InputLayer(input_shape = (2,)),
        keras.layers.Flatten(),
        keras.layers.Dense(100, activation="hard_sigmoid"),
        keras.layers.Dense(100, activation="exponential"),
        keras.layers.Dense(100, activation="hard_sigmoid"),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(2,activation="hard_sigmoid")
    ]
)
model.compile(
    loss="sparse_categorical_crossentropy", 
    optimizer=keras.optimizers.RMSprop(learning_rate=0.02),
    metrics=["accuracy"],
)
history=model.fit(xdata,y,epochs=10,validation_split=0.02)

width = st.sidebar.slider("plot width",min_value=5,max_value=8,step=1)
height = st.sidebar.slider("plot height",min_value=2,max_value=4,step=1)
fig, ax = plt.subplots(figsize=(width, height))
ax.plot(history.history['loss'])
ax.plot(history.history['val_loss'])
ax.set_ylabel('loss')
ax.set_xlabel('epoch')
ax.legend(['train', 'validation'], loc='upper right')
st.pyplot(fig)
st.write(f"this is the loss graph about {x1c} {x2c} in CovidDS with {yc} in SteamDS")
st.write("As we can see,every result is really bad for guess.")
st.subheader("Therefore, there is no relationship between the online player's number and the Covid-19 infection.")
st.subheader("")
st.subheader("An interesting finding during the project")
A=alt.Chart(df1).mark_point().encode(
        x=alt.X('Month',scale=alt.Scale(zero=False)),    
        y=alt.Y("Avg_Players", scale=alt.Scale(domain=[2000000, 3250000])),
        tooltip=['Month',"Avg_Players",'Peak Players',"Gain"],
        color=alt.value("pink"),
        shape = alt.value("diamond"),
        ).properties(
        title='The graph about date and Players in steam',
        width = 800,
        height= 400
    )
            
B=alt.Chart(df1).mark_point().encode(
        x=alt.X('Month',scale=alt.Scale(zero=False)),    
        y=alt.Y("Peak Players", scale=alt.Scale(domain=[4000000, 6500000])),
        tooltip=['Month',"Avg_Players",'Peak Players',"Gain"],
        color=alt.value("green"),
        shape = alt.value("square"),
        ).properties(
        width = 800,
        height= 400
    )
A+B
df1["rate"]=df1['Peak Players']/df1['Avg_Players']
st.write(f"There is a relationship between 'Avg_Players' and 'Peak Players'and the rate is closed to {round(df1['rate'].mean(),3)} for 'Peak Players'/'Avg_Players'")
