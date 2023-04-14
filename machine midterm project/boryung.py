import pandas as pd

## 보령 데이터
for i in range(10,23):
    path = './mac/boryung/(%d).csv' %(i-9)
    locals()['df'+str(i)] = pd.read_csv(path, encoding="euc-kr")
#2010년 4/3 ~ 10/24 (94, 298) # for문 인덱스 -2 처리하기
df1 = pd.read_csv('./mac/boryung/(1).csv', encoding='euc-kr')
total_sum = pd.DataFrame()

for i in range(92, 297,7):
    avg_temp = df10.loc[i:i+6, "평균기온(°C)"].mean()
    avg_humidity = df10.loc[i+6, "평균 상대습도(%)"].mean()
    temp_df = pd.DataFrame({"평균기온(°C)": [avg_temp], "평균 상대습도(%)": [avg_humidity]})
    total_sum = pd.concat([total_sum, temp_df])

total_sum10 = total_sum.reset_index(drop=True)
print(total_sum10)

#2011년 (91, 299)
df2 = pd.read_csv('./mac/boryung/(2).csv', encoding='euc-kr')
total_sum = pd.DataFrame()

for i in range(89, 297,7):
    avg_temp = df11.loc[i:i+6, "평균기온(°C)"].mean()
    avg_humidity = df11.loc[i+6, "평균 상대습도(%)"].mean()
    temp_df = pd.DataFrame({"평균기온(°C)": [avg_temp], "평균 상대습도(%)": [avg_humidity]})
    total_sum = pd.concat([total_sum, temp_df])

total_sum11 = total_sum.reset_index(drop=True)
print(total_sum11)

#2012년 4/14 ~ 10/21 (106, 297)
df3 = pd.read_csv('./mac/boryung/(3).csv', encoding='euc-kr')
total_sum = pd.DataFrame()

for i in range(104, 295,7):
    avg_temp = df12.loc[i:i+6, "평균기온(°C)"].mean()
    avg_humidity = df12.loc[i+6, "평균 상대습도(%)"].mean()
    temp_df = pd.DataFrame({"평균기온(°C)": [avg_temp], "평균 상대습도(%)": [avg_humidity]})
    total_sum = pd.concat([total_sum, temp_df])

total_sum12 = total_sum.reset_index(drop=True)
print(total_sum12)

#2013년 3/31 ~ 10/28 (91,303)
df4 = pd.read_csv('./mac/boryung/(4).csv', encoding='euc-kr')
total_sum = pd.DataFrame()

for i in range(89, 301,7):
    avg_temp = df13.loc[i:i+6, "평균기온(°C)"].mean()
    avg_humidity = df13.loc[i+6, "평균 상대습도(%)"].mean()
    temp_df = pd.DataFrame({"평균기온(°C)": [avg_temp], "평균 상대습도(%)": [avg_humidity]})
    total_sum = pd.concat([total_sum, temp_df])

total_sum13 = total_sum.reset_index(drop=True)
print(total_sum13)

#2014년 4/6 ~ 10/27 (97, 302)
df5 = pd.read_csv('./mac/boryung/(5).csv', encoding='euc-kr')
total_sum = pd.DataFrame()

for i in range(95, 300,7):
    avg_temp = df14.loc[i:i+6, "평균기온(°C)"].mean()
    avg_humidity = df14.loc[i+6, "평균 상대습도(%)"].mean()
    temp_df = pd.DataFrame({"평균기온(°C)": [avg_temp], "평균 상대습도(%)": [avg_humidity]})
    total_sum = pd.concat([total_sum, temp_df])

total_sum14 = total_sum.reset_index(drop=True)
print(total_sum14)

#2015년 3/31 ~ 10/28 (91,303)
df6 = pd.read_csv('./mac/boryung/(6).csv', encoding='euc-kr')
total_sum = pd.DataFrame()

for i in range(89, 301,7):
    avg_temp = df15.loc[i:i+6, "평균기온(°C)"].mean()
    avg_humidity = df15.loc[i+6, "평균 상대습도(%)"].mean()
    temp_df = pd.DataFrame({"평균기온(°C)": [avg_temp], "평균 상대습도(%)": [avg_humidity]})
    total_sum = pd.concat([total_sum, temp_df])

total_sum15 = total_sum.reset_index(drop=True)
print(total_sum15)

#2016년 4/5 ~ 10/26 (97, 302)
df7 = pd.read_csv('./mac/boryung/(7).csv', encoding='euc-kr')
total_sum = pd.DataFrame()

for i in range(95, 300,7):
    avg_temp = df16.loc[i:i+6, "평균기온(°C)"].mean()
    avg_humidity = df16.loc[i+6, "평균 상대습도(%)"].mean()
    temp_df = pd.DataFrame({"평균기온(°C)": [avg_temp], "평균 상대습도(%)": [avg_humidity]})
    total_sum = pd.concat([total_sum, temp_df])

total_sum16 = total_sum.reset_index(drop=True)
print(total_sum16)

#2017년 3/27 ~ 10/31 (87, 306)
df8 = pd.read_csv('./mac/boryung/(8).csv', encoding='euc-kr')
total_sum = pd.DataFrame()

for i in range(85, 304,7):
    avg_temp = df17.loc[i:i+6, "평균기온(°C)"].mean()
    avg_humidity = df17.loc[i+6, "평균 상대습도(%)"].mean()
    temp_df = pd.DataFrame({"평균기온(°C)": [avg_temp], "평균 상대습도(%)": [avg_humidity]})
    total_sum = pd.concat([total_sum, temp_df])

total_sum17 = total_sum.reset_index(drop=True)
print(total_sum17)

#2018년 4/2 ~ 10/30 (93, 305)
df9 = pd.read_csv('./mac/boryung/(9).csv', encoding='euc-kr')
total_sum = pd.DataFrame()

for i in range(91, 303,7):
    avg_temp = df18.loc[i:i+6, "평균기온(°C)"].mean()
    avg_humidity = df18.loc[i+6, "평균 상대습도(%)"].mean()
    temp_df = pd.DataFrame({"평균기온(°C)": [avg_temp], "평균 상대습도(%)": [avg_humidity]})
    total_sum = pd.concat([total_sum, temp_df])

total_sum18 = total_sum.reset_index(drop=True)
print(total_sum18)

#2019년 4/1 ~ 10/30 (92, 305)
df10 = pd.read_csv('./mac/boryung/(10).csv', encoding='euc-kr')
total_sum = pd.DataFrame()

for i in range(90, 303,7):
    avg_temp = df19.loc[i:i+6, "평균기온(°C)"].mean()
    avg_humidity = df19.loc[i+6, "평균 상대습도(%)"].mean()
    temp_df = pd.DataFrame({"평균기온(°C)": [avg_temp], "평균 상대습도(%)": [avg_humidity]})
    total_sum = pd.concat([total_sum, temp_df])

total_sum19 = total_sum.reset_index(drop=True)
print(total_sum19)

#2020년 3/30 ~ 11/5 (90, 311)
df11 = pd.read_csv('./mac/boryung/(11).csv', encoding='euc-kr')
total_sum = pd.DataFrame()

for i in range(88, 309,7):
    avg_temp = df20.loc[i:i+6, "평균기온(°C)"].mean()
    avg_humidity = df20.loc[i+6, "평균 상대습도(%)"].mean()
    temp_df = pd.DataFrame({"평균기온(°C)": [avg_temp], "평균 상대습도(%)": [avg_humidity]})
    total_sum = pd.concat([total_sum, temp_df])

total_sum20 = total_sum.reset_index(drop=True)
print(total_sum20)

#2021년 3/29 ~ 10/26 (89, 301)
df12 = pd.read_csv('./mac/boryung/(12).csv', encoding='euc-kr')
total_sum = pd.DataFrame()

for i in range(87, 299,7):
    avg_temp = df21.loc[i:i+6, "평균기온(°C)"].mean()
    avg_humidity = df21.loc[i+6, "평균 상대습도(%)"].mean()
    temp_df = pd.DataFrame({"평균기온(°C)": [avg_temp], "평균 상대습도(%)": [avg_humidity]})
    total_sum = pd.concat([total_sum, temp_df])

total_sum21 = total_sum.reset_index(drop=True)
print(total_sum21)

#2022년 4/4 ~ 10/25 (95, 300)
df13 = pd.read_csv('./mac/boryung/(13).csv', encoding='euc-kr')
total_sum = pd.DataFrame()

for i in range(93, 298,7):
    avg_temp = df22.loc[i:i+6, "평균기온(°C)"].mean()
    avg_humidity = df22.loc[i+6, "평균 상대습도(%)"].mean()
    temp_df = pd.DataFrame({"평균기온(°C)": [avg_temp], "평균 상대습도(%)": [avg_humidity]})
    total_sum = pd.concat([total_sum, temp_df])

total_sum22 = total_sum.reset_index(drop=True)
print(total_sum22)

sum_df = pd.concat([total_sum10,total_sum11,total_sum12,total_sum13,total_sum14,total_sum15,total_sum16,total_sum17,total_sum18,total_sum19,total_sum20,total_sum21,total_sum22])
print(sum_df)
sum_df.to_excel('./mac/boryung/export.xlsx', sheet_name='new_name')
