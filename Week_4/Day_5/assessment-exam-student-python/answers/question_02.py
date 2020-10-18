import pandas as pd
import sys
from supporting_files.data_loader import load_excel

df = load_excel('supporting_files/SaleData.xlsx')


"""
Write a Pandas program to find the total sale amount (Sale_amt) region and manager wise. 
Order the dataframe by Sale_amt, starting with the highest.
Output should be DataFrame with 3 columns (order is important):
    - Region
    - Manager
    - Sale_amt
and numeric index starting with 0 (after sorting).
"""

def compute_total_sale(data):
    temp = data.groupby(['Manager','Region']).sum()
    temp = data.groupby(['Manager','Region']).sum()
    temp.drop(['Units','Unit_price'], axis = 1, inplace = True)
    temp = temp.reset_index()
    temp = temp.sort_values(by='Sale_amt', ascending = False)
    temp= temp[['Region','Manager','Sale_amt']]
    temp = temp.reset_index()
    temp.drop('index',axis = 1, inplace= True)
    return(temp)