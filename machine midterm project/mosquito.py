import pandas as pd
import numpy as np

counts = range(2010,2023,1)

targetdatas =pd.DataFrame()
targ_df = pd.DataFrame()
for count in counts:
    path = "./mosquito/%d.xlsx" %count
    targetdatas = locals()['targetdatas' + str(count)] = pd.read_excel(path, index_col=0) #인덱스 제거
    targ_df = pd.concat([targ_df,targetdatas.at["평균", "계(total)"]])

print(targ_df)
targ_df.to_excel('./mosquito/export.xlsx', sheet_name='new_name')