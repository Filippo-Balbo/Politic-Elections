#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 19:25:18 2024

@author: filippobalbo
"""
import pandas as pd
import numpy as np
import requests
import json 
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import preprocessing
from linearmodels.panel import PanelOLS
from linearmodels.panel import RandomEffects
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# IMPORTING DATASET FROM EXCEL 
Turnout_Data_Dirty = pd.read_excel("/Users/filippobalbo/Documents/Documents/Financial Projects/European Election/Turnout Europe.xls", header=1)
NewParties_Data = pd.read_excel("/Users/filippobalbo/Documents/Documents/Financial Projects/European Election/New Parties.xlsx")

# DATA CLEANING: 
    
NewParties_Data = NewParties_Data.drop(['Election_date', 'Abstensionism', 'Unnamed: 6', 'Unnamed: 7'], axis = 1) # Deleting useless columns
    
Turnout_Data_Dirty = Turnout_Data_Dirty.drop(['ISO2', 'ISO3'], axis = 1) # Deleting useless columns
Turnout_Data_Dirty = Turnout_Data_Dirty.sort_values(['Country', 'Date']) #Sorting columns based on Countries and Dates
Turnout_Data_Dirty['Country'] = Turnout_Data_Dirty['Country'].str.replace('United Kingdom', 'UK')
Turnout_Data_Dirty = Turnout_Data_Dirty.rename(columns={'Parliamentary>Voter Turnout': 'Participation'})

# Define years to be removed in order to make the two dataframes match:
years_to_remove = [2019, 2018, 2017, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960]

# Replace unusual hyphens with standard hyphens
Turnout_Data_Dirty['Date'] = Turnout_Data_Dirty['Date'].str.replace('‑', '-')

# Convert 'Date' into Datetime format:
Turnout_Data_Dirty['Date'] = pd.to_datetime(Turnout_Data_Dirty['Date'])  

# Extract year from 'Date' column and create a new 'Year' column
Turnout_Data_Dirty['Year'] = Turnout_Data_Dirty['Date'].dt.year

# Drop the 'Date' column if you only need the 'Year' column
Turnout_Data_Dirty = Turnout_Data_Dirty.drop(columns=['Date'])

# Filter the DataFrame to remove the specified years
Turnout_Data = Turnout_Data_Dirty[~((Turnout_Data_Dirty['Year'].isin(years_to_remove)) & (Turnout_Data_Dirty.groupby('Country').cumcount(ascending=False) < 2))]

# Merge the dataframes on 'Country' and 'Year'
Merged_Data = pd.merge(Turnout_Data, NewParties_Data, on=['Country', 'Year'], how='outer', indicator=True)

# Identify non-corresponding years
non_corresponding_years = Merged_Data[Merged_Data['_merge'] != 'both']

# Display non-corresponding years
non_corresponding_years[['Country', 'Year', '_merge']]

# Deleting the non-corresponding years/countries from the merged dataframe:
Election_Data = Merged_Data.loc[Merged_Data['_merge'] == 'both']

# PANEL DATA REGRESSION:

# Data Manipulation:
  
# Initialize the scaler
scaler = MinMaxScaler()

# Normalize selected columns
columns_to_normalize = ['Participation']  #Min max scaler
Election_Data[columns_to_normalize] = scaler.fit_transform(Election_Data[columns_to_normalize]) #Min max scaler 
    
Columns_to_Manipulate = ['PSInn', 'CPSInn']
Election_Data[Columns_to_Manipulate] = preprocessing.normalize(Election_Data[Columns_to_Manipulate]) # Z-normalization
Election_Data[Columns_to_Manipulate] = pd.DataFrame(Election_Data[Columns_to_Manipulate])

# Performing OLS Regression:
X = Election_Data['Participation']
y = Election_Data['PSInn']
X = sm.add_constant(X)  # Add a constant (intercept) to the model

# Fit the model
OLS_model = sm.OLS(y, X)
OLS_results = OLS_model.fit()

# Print results
print(OLS_results.summary())    

# Performing Fixed Effects Model (might be more appropriate):
# Define the model

# Convert 'Year' column to numeric

Election_Data['Year'] = pd.to_numeric(Election_Data['Year'])
Election_Data.set_index(['Country', 'Year'], inplace=True)

# Define dependent and independent variables
Dependent_Var = 'PSInn'
Independent_Var = ['Participation']

# Fit the model
FE_model = PanelOLS(Election_Data[Dependent_Var], Election_Data[Independent_Var], entity_effects=True, time_effects=True)
FE_result = FE_model.fit()

# Print the regression results
print(FE_result)

# Performing Random Effects Regression:  
# Create a RandomEffects model
RE_model = RandomEffects(Election_Data[Dependent_Var], Election_Data[Independent_Var])

# Fit the model
RE_results = RE_model.fit()

# Print the summary of the regression results
print(RE_results)

# FOCUS ON ITALY:
#OLS Regression
Italy = Election_Data.iloc[161:177,]

# Performing OLS Regression:
ITA_X = Italy['Participation']
ITA_y = Italy['PSInn']
ITA_X = sm.add_constant(ITA_X)  # Add a constant (intercept) to the model

# Fit the model
OLS_model_ITA = sm.OLS(ITA_y, ITA_X)
OLS_results_ITA = OLS_model_ITA.fit()

# Print results
print(OLS_results_ITA.summary()) 






