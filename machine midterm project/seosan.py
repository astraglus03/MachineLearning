import pandas as pd
# 서산 데이터
for i in range(10,23):
    path = './mac/seosan/(%d).csv' %(i-9)
    locals()['df'+str(i)] = pd.read_csv(path, encoding="euc-kr")

# 2010년 : 4월3일 ~ 10월24일
total_sum = pd.DataFrame()
for i in range(92, 296,7):
    avg_temp = df10.loc[i:i+6, "평균기온(°C)"].mean()
    avg_humidity = df10.loc[i+6, "평균 상대습도(%)"].mean()
    temp_df = pd.DataFrame({"평균기온(°C)": [avg_temp], "평균 상대습도(%)": [avg_humidity]})
    total_sum = pd.concat([total_sum, temp_df])

total_sum10 = total_sum.reset_index(drop=True)
# print(total_sum10)

# 2011년 : 4월3일 ~ 10월24일
total_sum = pd.DataFrame()
for i in range(92, 297,7):
    avg_temp = df11.loc[i:i+6, "평균기온(°C)"].mean()
    avg_humidity = df11.loc[i+6, "평균 상대습도(%)"].mean()
    temp_df = pd.DataFrame({"평균기온(°C)": [avg_temp], "평균 상대습도(%)": [avg_humidity]})
    total_sum = pd.concat([total_sum, temp_df])

total_sum11 = total_sum.reset_index(drop=True)
# print(total_sum11)

# 2012년 : 4월14일 ~ 10월21일
total_sum = pd.DataFrame()
for i in range(104, 294,7):
    avg_temp = df12.loc[i:i+6, "평균기온(°C)"].mean()
    avg_humidity = df12.loc[i+6, "평균 상대습도(%)"].mean()
    temp_df = pd.DataFrame({"평균기온(°C)": [avg_temp], "평균 상대습도(%)": [avg_humidity]})
    total_sum = pd.concat([total_sum, temp_df])

total_sum12 = total_sum.reset_index(drop=True)
# print(total_sum12)

# 2013년 : 3월31일 ~ 10월28일
total_sum = pd.DataFrame()
for i in range(89, 300,7):
    avg_temp = df13.loc[i:i+6, "평균기온(°C)"].mean()
    avg_humidity = df13.loc[i+6, "평균 상대습도(%)"].mean()
    temp_df = pd.DataFrame({"평균기온(°C)": [avg_temp], "평균 상대습도(%)": [avg_humidity]})
    total_sum = pd.concat([total_sum, temp_df])

total_sum13 = total_sum.reset_index(drop=True)
# print(total_sum13)

# 2014년 : 4월6일 ~ 10월27일
total_sum = pd.DataFrame()
for i in range(95, 299,7):
    avg_temp = df14.loc[i:i+6, "평균기온(°C)"].mean()
    avg_humidity = df14.loc[i+6, "평균 상대습도(%)"].mean()
    temp_df = pd.DataFrame({"평균기온(°C)": [avg_temp], "평균 상대습도(%)": [avg_humidity]})
    total_sum = pd.concat([total_sum, temp_df])

total_sum14 = total_sum.reset_index(drop=True)
# print(total_sum14)

# 2015년 : 3월31일 ~ 10월28일
total_sum = pd.DataFrame()
for i in range(89, 300,7):
    avg_temp = df15.loc[i:i+6, "평균기온(°C)"].mean()
    avg_humidity = df15.loc[i+6, "평균 상대습도(%)"].mean()
    temp_df = pd.DataFrame({"평균기온(°C)": [avg_temp], "평균 상대습도(%)": [avg_humidity]})
    total_sum = pd.concat([total_sum, temp_df])

total_sum15 = total_sum.reset_index(drop=True)
# print(total_sum15)

# 2016년 : 4월5일 ~ 10월26일
total_sum = pd.DataFrame()
for i in range(95, 299,7):
    avg_temp = df16.loc[i:i+6, "평균기온(°C)"].mean()
    avg_humidity = df16.loc[i+6, "평균 상대습도(%)"].mean()
    temp_df = pd.DataFrame({"평균기온(°C)": [avg_temp], "평균 상대습도(%)": [avg_humidity]})
    total_sum = pd.concat([total_sum, temp_df])

total_sum16 = total_sum.reset_index(drop=True)
# print(total_sum16)

# 2017년 : 3월27일 ~ 10월31일
total_sum = pd.DataFrame()
for i in range(85, 303,7):
    avg_temp = df17.loc[i:i+6, "평균기온(°C)"].mean()
    avg_humidity = df17.loc[i+6, "평균 상대습도(%)"].mean()
    temp_df = pd.DataFrame({"평균기온(°C)": [avg_temp], "평균 상대습도(%)": [avg_humidity]})
    total_sum = pd.concat([total_sum, temp_df])

total_sum17 = total_sum.reset_index(drop=True)
# print(total_sum17)

# 2018년 : 4월2일 ~ 10월30일
total_sum = pd.DataFrame()
for i in range(91, 302,7):
    avg_temp = df18.loc[i:i+6, "평균기온(°C)"].mean()
    avg_humidity = df18.loc[i+6, "평균 상대습도(%)"].mean()
    temp_df = pd.DataFrame({"평균기온(°C)": [avg_temp], "평균 상대습도(%)": [avg_humidity]})
    total_sum = pd.concat([total_sum, temp_df])

total_sum18 = total_sum.reset_index(drop=True)
# print(total_sum18)

# 2019년 : 4월1일 ~ 10월30일
total_sum = pd.DataFrame()
for i in range(90, 302,7):
    avg_temp = df19.loc[i:i+6, "평균기온(°C)"].mean()
    avg_humidity = df19.loc[i+6, "평균 상대습도(%)"].mean()
    temp_df = pd.DataFrame({"평균기온(°C)": [avg_temp], "평균 상대습도(%)": [avg_humidity]})
    total_sum = pd.concat([total_sum, temp_df])

total_sum19 = total_sum.reset_index(drop=True)
# print(total_sum19)

# 2020년 : 3월30일 ~ 11월5일
total_sum = pd.DataFrame()
for i in range(89, 309,7):
    avg_temp = df20.loc[i:i+6, "평균기온(°C)"].mean()
    avg_humidity = df20.loc[i+6, "평균 상대습도(%)"].mean()
    temp_df = pd.DataFrame({"평균기온(°C)": [avg_temp], "평균 상대습도(%)": [avg_humidity]})
    total_sum = pd.concat([total_sum, temp_df])

total_sum20 = total_sum.reset_index(drop=True)
# print(total_sum20)

# 2021년 : 3월29일 ~ 10월26일
total_sum = pd.DataFrame()
for i in range(87, 298,7):
    avg_temp = df21.loc[i:i+6, "평균기온(°C)"].mean()
    avg_humidity = df21.loc[i+6, "평균 상대습도(%)"].mean()
    temp_df = pd.DataFrame({"평균기온(°C)": [avg_temp], "평균 상대습도(%)": [avg_humidity]})
    total_sum = pd.concat([total_sum, temp_df])

total_sum21 = total_sum.reset_index(drop=True)
# print(total_sum21)

# 2022년 : 4월4일 ~ 10월25일
total_sum = pd.DataFrame()
for i in range(93, 297,7):
    avg_temp = df22.loc[i:i+6, "평균기온(°C)"].mean()
    avg_humidity = df22.loc[i+6, "평균 상대습도(%)"].mean()
    temp_df = pd.DataFrame({"평균기온(°C)": [avg_temp], "평균 상대습도(%)": [avg_humidity]})
    total_sum = pd.concat([total_sum, temp_df])

total_sum22 = total_sum.reset_index(drop=True)
# print(total_sum22)

sum_df = pd.concat([total_sum10,total_sum11,total_sum12,total_sum13,total_sum14,total_sum15,total_sum16,total_sum17,total_sum18,total_sum19,total_sum20,total_sum21,total_sum22])
print(sum_df)
sum_df.to_excel('./mac/seosan/export.xlsx', sheet_name='new_name')