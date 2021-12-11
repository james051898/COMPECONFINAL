import pandas as pd

chrt = pd.read_csv("Cable_Company_Case_Study/Datasets/CHTR_Trending Schedule.csv")
chrt = chrt.set_index(chrt.Quarter)
chrt = chrt.drop(columns='Quarter')
print(chrt)
