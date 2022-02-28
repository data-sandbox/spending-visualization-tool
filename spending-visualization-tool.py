#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Initial release: 2022 02

"""

from os.path import exists
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#%%
# get all csv files in directory
path = os.getcwd()
csv_files = glob.glob(os.path.join(path, "*.csv"))

# generate empty dataframe
spending_output_df = pd.DataFrame()

#%% iterate through list of csv files
for f in csv_files:
    
    # read the month
    month = f[71:75]
    print(month)
    
    # read the csv file
    spending_raw_df = pd.read_csv(f)
    
    # clean csv if header rows included
    date_check = ['Date', 'Post Date', 'Posted Date']
    # reset flag
    date_flag = 0 
    for d in date_check:
        if d in spending_raw_df.columns:
            date_flag = 1
    if date_flag == 0:
        # header rows need to be cleaned    
        spending_raw_df = pd.read_csv(f, skiprows=3)
        
    # data consistency check
    if 'Amount Debit' in spending_raw_df.columns:
        spending_raw_df['Debit'] = spending_raw_df['Amount Debit']
    if 'Amount Credit' in spending_raw_df.columns:
        spending_raw_df['Credit'] = spending_raw_df['Amount Credit']
    
    if 'Amount' in spending_raw_df.columns:
         spending_raw_df['Debit'] = spending_raw_df['Amount']
    if 'Credit' not in spending_raw_df.columns:
         spending_raw_df['Credit'] = 0

    # clean columns to be all positive
    spending_raw_df['Debit'] = spending_raw_df['Debit'].abs()
    spending_raw_df['Credit'] = spending_raw_df['Credit'].abs()

    # clean nan values with zeros for later math operations
    spending_raw_df = spending_raw_df.fillna(value={'Debit': 0, 'Credit': 0})
        
    # add net column
    spending_raw_df['Net'] = spending_raw_df['Debit'] - spending_raw_df['Credit']
    # absolute value net column
    spending_raw_df['Net'] = spending_raw_df['Net'].abs()
    
    # keep only category and amount columns
    spending_raw_df = spending_raw_df[['Category', 'Debit', 'Credit', 'Net']]
    
    # create summary dataframe with categories spending grouped
    dfMonth = spending_raw_df.groupby('Category', as_index=False).agg('sum')
    
    # add column identifying the year and month
    dfMonth.insert(1, 'Date', month)
    
    # add monthly data to running annual data
    spending_output_df = pd.concat([spending_output_df, dfMonth])
    
    # reset index
    spending_output_df = spending_output_df.reset_index(drop=True)

#%% data cleaning
# group by category for each month
spending_output_df = spending_output_df.groupby(['Date', 'Category'], as_index=False).sum()

#spending_output_df.sort_values('Date')

# convert date to string so Dec to Jan dates still plot side by side
#spending_output_df['Date'] = spending_output_df['Date'].astype(str)

#%% plot function

def plot_summary(data_all, categories1, categories2):
    """
    Plot subsets of the data.

    Parameters
    ----------
    data_all : dataframe
        Data for all categories.
    categories1 : list
        List of desired categories.
    categories2 : list
        List of desired categories.

    Returns
    -------
    Plots for spending visualization.

    """
    
    # inputs
    label_angle = 60
    
    # data consistency check
    for j in categories1:
        if j not in categories1:
            data_all[j] = 0
    
    for j in categories2:
        if j not in categories2:
            data_all[j] = 0
    
    # create dataframes
    data_subset1 = data_all.loc[data_all['Category'].isin(categories1)]
    data_subset2 = data_all.loc[data_all['Category'].isin(categories2)]
    
    # force sort by date?
    # data_subset1.sort_values('Date')
    # data_subset2.sort_values('Date')
    
    #data_tuple1 = tuple[categories1, data_subset1]
    
    # overview plot
    plt.close('all')
    
    #colors = sns.color_palette("Set2")
    plt.figure()
    g = sns.lineplot(data=data_all,
                  x='Date', y='Net', 
                  hue='Category',
                  marker='o')
    plt.xticks(rotation=label_angle)  
    plt.show()
    
    # category subset 1
    plt.figure()
    colors = sns.color_palette("Set2", len(categories1))
    g = sns.lineplot(data=data_subset1,
                 x='Date', y='Net', 
                 hue='Category',
                 marker='o', 
                 palette=colors)
    plt.xticks(rotation=label_angle)
    plt.show()
    
    # category subset 2
    plt.figure()
    colors = sns.color_palette("Set2", len(categories2))
    g = sns.lineplot(data=data_subset2,
                 x='Date', y='Net', 
                 hue='Category',
                 marker='o', 
                 palette=colors)
    plt.xticks(rotation=label_angle)
    plt.show()

#%% run function

if __name__ == "__main__":
    
    # set desired categories to visualize
    necessary = ['rent',
                 'auto',
                 'transit',
                 'utilities',
                 'groceries']
    
    discretionary = ['travel',
                     'hobby',
                     'restaurant',
                     'home',
                     'personal']
    
    plot_summary(spending_output_df, necessary, discretionary)


