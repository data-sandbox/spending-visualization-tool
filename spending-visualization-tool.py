#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Initial release: 2022 02 22

"""

from os.path import exists
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

### INPUTS, BEGIN

# year, month
# date_start = 2101
# date_end = 2112
# date_range = np.arange(date_start, date_end + 1, 1)

### INPUTS, END

# get all csv files in directory
path = os.getcwd()
csv_files = glob.glob(os.path.join(path, "*.csv"))
  
# loop over the list of csv files
for f in csv_files:
    
    # read the month
    month = f[71:75]
    
    # read the csv file
    spending_raw_df = pd.read_csv(f)
    
    print(spending_raw_df)
    
    # generate dummy dataframe with all categories of interest
    categories = pd.read_csv('categories.csv')
    # add other columns
    categories['Debit'] = 0
    categories['Credit'] = 0
    categories['Net'] = 0
        
    # temporary dataframe to pull from
    # if account[i] == 'account1' and exists(file_name) == False:
    #     print('')
    #     print('No account1 statement')
    #     spending_raw_df = categories
    # elif account[i] == 'account2' and exists(file_name) == False:
    #     print('')
    #     print('No account2 statement')
    #     spending_raw_df = categories
    # elif account[i] == 'account3':
    #     # skip first few rows
    #     spending_raw_df = pd.read_csv(file_name, skiprows=3)
    # else:
    #     # skip no rows
    #     spending_raw_df = pd.read_csv(file_name)

    if 'Amount Debit' in spending_raw_df.columns:
        spending_raw_df['Debit'] = spending_raw_df['Amount Debit']
    if 'Amount Credit' in spending_raw_df.columns:
        spending_raw_df['Credit'] = spending_raw_df['Amount Credit']
    
    if 'Amount' in spending_raw_df.columns:
         spending_raw_df['Debit'] = spending_raw_df['Amount']
    if 'Credit' not in spending_raw_df.columns:
         spending_raw_df['Credit'] = 0

    # error checks
    # column_check = ['Debit', 'Credit']
    # for column in column_check:
    #     column in spending_raw_df.columns
    #     assert column > 0, "column not present"
    
    # column_check = 'Debit' in dfTemp1.columns
    # assert column_check > 0, "Debit column not present"
    # column_check = 'Credit' in dfTemp1.columns
    # assert column_check > 0, "Credit column not present"

    # force columns to be consistently positive
    spending_raw_df['Debit'] = spending_raw_df['Debit'].abs()
    spending_raw_df['Credit'] = spending_raw_df['Credit'].abs()

    # replace all nan values with zeros for later math operations
    spending_raw_df = spending_raw_df.fillna(value={'Debit': 0, 'Credit': 0})

    # add amount data to temporary dataframe summary
    spending_raw_df = pd.concat([spending_raw_df, spending_raw_df], axis=0)
        
    # add net column
    spending_raw_df['Net'] = spending_raw_df['Debit'] - spending_raw_df['Credit']
    # absolute value net column
    spending_raw_df['Net'] = spending_raw_df['Net'].abs()
    
    # add category list to include any categories with zero spending
    spending_raw_df = pd.concat([spending_raw_df, categories])
    
    # keep only category and amount columns
    spending_raw_df = spending_raw_df[['Category', 'Debit', 'Credit', 'Net']]
    
    # create summary dataframe with categories spending grouped
    dfMonth = spending_raw_df.groupby('Category', as_index=False).agg('sum')
    
    # add column identifying the year and month
    dfMonth.insert(1, 'Date', month)
    
    # determine name of running log data file
    # month = date_end % 100
    # if month < 2:
    #     # account for last entry being the prior year
    #     log_date = date_end - 89
    # else:
    #     log_date = date_end - 1
    
    # read running log data
    #spending_log_df = pd.read_csv('log' + str(log_date) + '.csv')
    
    # keep only category and amount columns
    #spending_log_df = spending_log_df[['Category', 'Date', 'Debit', 'Credit', 'Net']]
    
    # add monthly data to running annual data
    #spending_output_df = pd.concat([spending_log_df, dfMonth])
    
    # reset index
    #spending_output_df = spending_output_df.reset_index(drop=True)
    # export new log
    #spending_output_df.to_csv('log' + str(date_end) + '.csv')

# compute running totals
#dfTotals = spending_output_df.groupby('Category', as_index=False).agg('sum')

# compute expense totals only
spending_expenses_df = ['income', 'freelance', 'investment']
for i in spending_expenses_df:
    dfTotals = dfTotals[dfTotals['Category'].str.contains(spending_expenses_df[i])==False]
totalExp = dfTotals['Net'].sum()

# convert date to string so Dec to Jan dates still plot side by side
#spending_output_df['Date'] = spending_output_df['Date'].astype(str)


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
    
    # create dataframes
    data_subset1 = data_all.loc[data_all['Category'].isin(categories1)]
    data_subset2 = data_all.loc[data_all['Category'].isin(categories2)]
    
    data_tuple1 = tuple[categories1, data_subset1]
    
    # overview plot
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
    colors = sns.color_palette("Set2", len(data_tuple1[0]))
    g = sns.lineplot(data=data_tuple1[1],
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


