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
spending_main_df = pd.DataFrame()

#%% functions
def clean_data(spending_df, account, month):
    """
    Clean the dataframe before further processing
    
    """
    # remove header rows if included
    date_check = ['Date', 'Post Date', 'Posted Date']
    # reset flag
    date_flag = 0 
    for d in date_check:
        if d in spending_df.columns:
            date_flag = 1
    if date_flag == 0:
        # remove header rows 
        spending_df = pd.read_csv(f, skiprows=3)
        
    # consistency check
    if 'Amount Debit' in spending_df.columns:
        spending_df['Debit'] = spending_df['Amount Debit']
    if 'Amount Credit' in spending_df.columns:
        spending_df['Credit'] = spending_df['Amount Credit']
    if 'Amount' in spending_df.columns:
         spending_df['Debit'] = spending_df['Amount']
    if 'Credit' not in spending_df.columns:
         spending_df['Credit'] = 0

    # make all positive
    spending_df['Debit'] = spending_df['Debit'].abs()
    spending_df['Credit'] = spending_df['Credit'].abs()

    # column filter
    spending_df = spending_df[['Category', 'Debit', 'Credit']]

    # clean nan values with zeros for later math operations
    spending_df = spending_df.fillna(value={'Debit': 0, 'Credit': 0})
    
    # add net column
    spending_df['Net'] = spending_df['Debit'] - spending_df['Credit']
    # absolute value net column
    spending_df['Net'] = spending_df['Net'].abs()
    
    # second df containing account name and month
    spending_all_df = spending_df.copy()
    spending_all_df['Account'] = account
    spending_all_df['Month'] = month
    
    
    return spending_df, spending_all_df

#%% NEW WORKFLOW
"""
- create main df with all transactions and new column with account name
- print latest month's df of transactions and require input before adding it?
- groupby just before plotting
- separate plotting by making new groupby df, add average column, sort by average
column, and split index with one half on one plot, other half on second plot
"""

#%% iterate through list of csv files
for f in csv_files:
    
    # slice filename to get the month
    month = f[71:75]
    # get account name
    account = f[75:]
    account = account[:-4]
    # print status
    print(month)
    
    # read the csv file and clean the data
    spending_raw_df, spending_all_df = clean_data(pd.read_csv(f), account, month)
    
    # create summary dataframe with categories spending grouped
    spending_grouped_df = spending_raw_df.groupby('Category', as_index=False).agg('sum')
    
    # add column identifying the year and month
    spending_grouped_df.insert(1, 'Date', month)
    
    # add monthly data to running annual data
    spending_output_df = pd.concat([spending_output_df, spending_grouped_df],
                                   ignore_index=True, sort=False)
    spending_main_df = pd.concat([spending_main_df, spending_all_df],
                                 ignore_index=True, sort=False)
# group by category for each month
spending_output_df = spending_output_df.groupby(['Date', 'Category'], as_index=False).sum()

# WIP. Get mean to split categories across multiple plots later on
mean_test = spending_output_df.groupby(['Category'], as_index=False).mean()

# convert date to string so Dec to Jan dates still plot side by side
#spending_output_df['Date'] = spending_output_df['Date'].astype(str)

#%% plot function

def plot_summary(data_all, categories1, categories2):
    """
    Plot subsets of the data.

    """
    
    # inputs
    label_angle = 60
    
    # data consistency check. if category does not exist, give 0 value
    for j in categories1:
        if j not in categories1:
            data_all[j] = 0
    
    for j in categories2:
        if j not in categories2:
            data_all[j] = 0
    
    # create dataframes
    data_subset1 = data_all.loc[data_all['Category'].isin(categories1)]
    data_subset2 = data_all.loc[data_all['Category'].isin(categories2)]
    
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

#%% plot function
def spending_threshold(data_df, threshold):
    
    data_df_low = data_df[data_df['Net'] < threshold]
    data_df_high = data_df[data_df['Net'] > threshold] 
    
    label_angle = 60
    
    plt.figure()
    g = sns.lineplot(data=data_df_low,
                 x='Date', y='Net', 
                 hue='Category',
                 marker='o')
    plt.xticks(rotation=label_angle)
    plt.show()
    
    plt.figure()
    g = sns.lineplot(data=data_df_high,
                 x='Date', y='Net', 
                 hue='Category',
                 marker='o')
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
    
    #plot_summary(spending_output_df, necessary, discretionary)
    spending_threshold(spending_output_df, 800)

