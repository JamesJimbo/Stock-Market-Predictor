#James McKenna - CSCADDC Python Implementation

#This machine learning algorithm predicts the value of stock of a certain company
#The prediction spans across the next 30 days when the data is published

import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
import xlrd
import datetime
#Modules used to plot graphs, store stock market data, mathematical functions, etc.

quandl.ApiConfig.api_key = "gixor5yfm776SbxoHtB3"
#This is my api key for my Quandl account, I need this code to run the AI
style.use('ggplot')
#Style used to plot the graph

def inputWikiCode():
    global inputCode
    #Makes the variable global so it can be defined by the program
    
    dataFrame = pd.read_excel("WikiCodes.xlsx","Wiki Codes")
    #Inputs the name of the document and the name of the sheet
    #There is an excel document that contains all the wiki codes
    for x in range(1,3200):
        #There are 3200 unique wiki codes to choose from
        listCodes = dataFrame['Codes'].values.tolist()
        #'Codes' is the name of the column
        #This appends each row in the excel file into a list

    codeChoice = input("Would you like to see the available wiki codes? ")
    if codeChoice.lower() == "yes" or codeChoice.lower() == "y":
        for x in range(len(listCodes)):
            print(listCodes[x])
    #Allows the user to view all the wiki codes before inputting their answer

    inputCode = str(input("\nPlease enter wiki code: ")).upper()
    #Converts the user input to upper case

    if inputCode in listCodes:
        #Checks if the user input is in the new list
        print(inputCode+" is a valid wiki code\n")
    else:
        print("That is an invalid wiki code, please try again\n")
        inputWikiCode()
        #If the input is invalid, it will iterate until a valid code is inputted

inputWikiCode()
#Runs the subroutine for the wiki code before running the AI

df = quandl.get('WIKI/'+inputCode)
#The data for this AI will be the stock market value depending
#on the wiki code input by the user

#The database for stock market values can be found on Quandl

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]
#The values for adjustment

forecastCol = ('Adj. Close')
df.fillna(-99999, inplace = True)

forecastOut = int(math.ceil(0.01 * len(df)))

df['label'] = df[forecastCol].shift(-forecastOut)
#shifts 30 days

x = np.array(df.drop(['label'], 1))
x = preprocessing.scale(x)
x = x[:-forecastOut]
xLately = x[-forecastOut:]
#last 30 days

df.dropna(inplace = True)
y = np.array(df['label'])
y = np.array(df['label'])

xTrain, xTest, yTrain, yTest = model_selection.train_test_split(x, y, test_size = 0.2)

clf = LinearRegression(n_jobs = -1)
#clf means Classifier
clf.fit(xTrain, yTrain)
#This trains the AI to make predictions based off previous data
#This is called a "bottom up" approach

with open('linearregression.pickle','wb') as f:
    pickle.dump(clf, f)

pickleIn = open('linearregression.pickle','rb')
clf = pickle.load(pickleIn)

accuracy = clf.score(xTest, yTest)
#Stores the value for predicted data by percentage
#For example %95 accuracy will be represented as 0.95

forecastSet = clf.predict(xLately)
print(forecastSet, accuracy, forecastOut)
df['Forecast'] = np.nan

lastDate = df.iloc[-1].name
lastUnix = lastDate.timestamp()
oneDay = 86400
#seconds in one day
nextUnix = lastUnix + oneDay
#creates variable for storing prediction

for i in forecastSet:
    nextDate = datetime.datetime.fromtimestamp(nextUnix)
    nextUnix += oneDay
    df.loc[nextDate] = [np.nan for _ in range(len(df.columns)-1)] + [i]
    #list of values of np.nan for the values of adjustment

print(df.tail())
#Displays the data for stock values in the shell
print(accuracy,"percent accuracy\n")
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc = 4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
#Plots the data into a legend graph
#The red graph show current data, the blue graph shows predicted data
