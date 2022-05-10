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

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

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

    # add columns
    spending_df['Account'] = account
    spending_df['Month'] = month

    # clean nan values with zeros for later math operations
    spending_df = spending_df.fillna(value={'Debit': 0, 'Credit': 0})
    
     # add net column. Negative value indicates credit
    spending_df['Net'] = spending_df['Debit'] - spending_df['Credit'] 
    
    # column filter and ordering
    spending_df = spending_df[['Month', 'Account', 'Description', 'Category', 
                               'Debit', 'Credit', 'Net']]

    # remove remaining nan values under category
    spending_df = spending_df.dropna(axis=0)
    
    return spending_df

#%% plot function

def plot_summary(data_all, categories1, categories2):
    """
    Plot subsets of the data.

    """
    # reset default parameters
    sns.set_theme()
    # font size
    sns.set_context("notebook")

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
                  x='Month', y='Net', 
                  hue='Category',
                  marker='o')
    plt.xticks(rotation=label_angle)  
    plt.show()
    
    # category subset 1
    plt.figure()
    #colors = sns.color_palette("Set2", len(categories1))
    g = sns.lineplot(data=data_subset1,
                 x='Month', y='Net', 
                 hue='Category',
                 marker='o')
    plt.xticks(rotation=label_angle)
    plt.xlabel('Year, Month')
    plt.ylabel('Expenses ($)')
    plt.title('Expenses versus Time')
    # Put the legend out of the figure
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.show()
    
    # category subset 2
    plt.figure()
    #colors = sns.color_palette("Set2", len(categories2))
    g = sns.lineplot(data=data_subset2,
                 x='Month', y='Net', 
                 hue='Category',
                 marker='o')
    plt.xticks(rotation=label_angle)
    plt.xlabel('Year, Month')
    plt.ylabel('Expenses ($)')
    plt.title('Expenses versus Time')
    # Put the legend out of the figure
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
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
    
    #% iterate through list of csv files
    for f in csv_files:
        
        # slice filename to get the month
        month = f[71:75]
        # get account name
        account = f[75:]
        account = account[:-4]
        # print status
        print(month)
        
        # read the csv file and clean the data
        spending_month_df = clean_data(pd.read_csv(f), account, month)
        
        # add monthly data to running annual data
        spending_main_df = pd.concat([spending_main_df, spending_month_df],
                                     ignore_index=True, sort=False)
    
    # group based on month and category totals
    spending_grouped_df = spending_main_df.groupby(['Month','Category'], as_index=False).agg('sum')
    
    # create df copy to add noise to
    spending_noise_df = spending_grouped_df.copy()
    # add noise
    spending_noise_df['Net'] = spending_noise_df['Net'] + np.random.normal(loc=50, scale=25)
    
    # overview plot
    # plt.close('all')
    # #colors = sns.color_palette("Set2")
    # plt.figure()
    # g = sns.lineplot(data=spending_grouped_df,
    #               x='Month', y='Net', 
    #               hue='Category',
    #               marker='o')
    # plt.xticks(rotation=60)  
    # plt.show()
    
    # group by category for each month
    #spending_output_df = spending_output_df.groupby(['Date', 'Category'], as_index=False).sum()

    
    # set desired categories to visualize
    necessary = ['rent',
                 'auto',
                 'transit',
                 'utilities',
                 'groceries']
    
    discretionary = ['restaurant',
                     'travel',
                     'groceries',
                     'home',
                     'utilities']
    
    plot_summary(spending_noise_df, necessary, discretionary)
    #spending_threshold(spending_output_df, 800)

