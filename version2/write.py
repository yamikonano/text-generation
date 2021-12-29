import pandas as pd
import csv

a=pd.read_csv(r"nvd_data_2012_2017_with_time_and_bid.csv")

# with open('2012_2017data.csv','w') as f:
#     writer=csv.writer(f)
#     writer.writerows(zip(a["DESC"]))