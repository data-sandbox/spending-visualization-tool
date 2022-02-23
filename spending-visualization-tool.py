#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Initial release: 2022 02 22

"""

from os.path import exists
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

### INPUTS, BEGIN

# year, month
startDate = 2101
endDate = 2112
rangeDate = np.arange(startDate, endDate+1, 1)

for i in range(len(rangeDate)):
    endDate = rangeDate[i]

    ### INPUTS, END
    
    # accounts
    accts = ['account1',
             'account2',
             'account3']
    
    # empty dataframe
    dfTemp2 = pd.DataFrame()
    
    # generate dummy dataframe with all categories of interest
    cat = pd.read_csv('categories.csv')
    # add other columns
    cat['Debit'] = 0
    cat['Credit'] = 0 
    cat['Net'] = 0
    
    for i in range(len(accts)):
    
        # file name of csv monthly statement
        fileName = str(endDate) + accts[i] + '.csv'
        print(fileName)
        
        # temporary dataframe to pull from
        if accts[i] == 'account1' and exists(fileName) == False:
            print('')
            print('No account1 statement')
            dfTemp1 = cat
        elif accts[i] == 'account2' and exists(fileName) == False:
            print('')
            print('No account2 statement')
            dfTemp1 = cat
        elif accts[i] == 'account3':
            # skip first few rows
            dfTemp1 = pd.read_csv(fileName, skiprows=3)
        else:
            # skip no rows
            dfTemp1 = pd.read_csv(fileName)
    
        if 'Amount Debit' in dfTemp1.columns:
            dfTemp1['Debit'] = dfTemp1['Amount Debit']
        if 'Amount Credit' in dfTemp1.columns:
            dfTemp1['Credit'] = dfTemp1['Amount Credit']
        
        if 'Amount' in dfTemp1.columns:
             dfTemp1['Debit'] = dfTemp1['Amount']
        if 'Credit' not in dfTemp1.columns:
             dfTemp1['Credit'] = 0      
    
        # error checks
        columnCheck = 'Debit' in dfTemp1.columns
        assert columnCheck > 0, "Debit column not present"
        columnCheck = 'Credit' in dfTemp1.columns
        assert columnCheck > 0, "Credit column not present"
    
        # force columns to be consistently positive
        dfTemp1['Debit'] = dfTemp1['Debit'].abs()
        dfTemp1['Credit'] = dfTemp1['Credit'].abs()
    
        # replace all nan values with zeros for later math operations
        dfTemp1 = dfTemp1.fillna(value={'Debit': 0, 'Credit': 0})
    
        # add amount data to temporary dataframe summary
        dfTemp2 = pd.concat([dfTemp2, dfTemp1], axis=0)
        
    # add net column
    dfTemp2['Net'] = dfTemp2['Debit'] - dfTemp2['Credit']
    # absolute value net column
    dfTemp2['Net'] = dfTemp2['Net'].abs()
    
    # add category list to include any categories with zero spending
    dfTemp2 = pd.concat([dfTemp2, cat])
    
    # keep only category and amount columns
    dfTemp2 = dfTemp2[['Category', 'Debit', 'Credit', 'Net']]
    
    # create summary dataframe with categories spending grouped
    dfMonth = dfTemp2.groupby('Category', as_index=False).agg('sum')
    
    # add column identifying the year and month
    dfMonth.insert(1, 'Date', str(endDate))
    
    # determine name of running log data file
    month = endDate % 100
    if month < 2:
        # account for last entry being the prior year
        logDate = endDate - 89
    else:
        logDate = endDate - 1
    
    # read running log data
    dfLog = pd.read_csv('log' + str(logDate) + '.csv')
    
    # keep only category and amount columns
    dfLog = dfLog[['Category', 'Date', 'Debit', 'Credit', 'Net']]
    
    # add monthly data to running annual data
    dfLogNew = pd.concat([dfLog, dfMonth])
    
    # reset index
    dfLogNew = dfLogNew.reset_index(drop=True)
    # export new log
    dfLogNew.to_csv('log' + str(endDate) + '.csv')

# compute running totals
dfTotals = dfLogNew.groupby('Category', as_index=False).agg('sum')

# compute expense totals only
dropCat = ['income', 'freelance', 'investment']
for i in range(len(dropCat)):
    dfTotals = dfTotals[dfTotals['Category'].str.contains(dropCat[i])==False]
totalExp = dfTotals['Net'].sum()

# convert date to string so Dec to Jan dates still plot side by side
dfLogNew['Date'] = dfLogNew['Date'].astype(str)


def plot_summary(data_all, categories1, categories2):
    
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

plot_summary(dfLogNew, necessary, discretionary)


