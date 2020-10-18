import pandas as pd
import sys
from supporting_files.data_loader import load_excel

df = load_excel('supporting_files/SaleData.xlsx')


"""
Write a function to count the manager wise sale (sale_cnt)
and mean value of sale amount (sale_mean). 
Order the output dataframe by sale_cnt, starting with the highest.
Output should be DataFrame with 3 columns (order is important):
    - Manager
    - sale_cnt
    - sale_mean
and numeric index starting with 0 (after sorting).
"""

def compute_agg_stats(data):
    sale_cnt = data.groupby('Manager').count()
    sale_mean = data.groupby('Manager').mean()
    sale_mean = pd.DataFrame(sale_mean['Sale_amt'])
    sale_mean = sale_mean.reset_index()
    sale_cnt = pd.DataFrame(sale_cnt['Sale_amt'])
    sale_cnt =sale_cnt.reset_index()
    final = pd.merge(sale_mean,sale_cnt,on='Manager')
    final = final[['Manager','Sale_amt_y','Sale_amt_x']]
    final.columns = ['Manager','sale_cnt','sale_mean']
    final = final.sort_values('sale_cnt', ascending = False)
    final = final.reset_index(drop = True)
    return(final)