import pandas as pd
# 강우량 데이터, 주차별 평균 데이터로 가공

df = pd.read_csv('./mac/rain/guemsan.csv', encoding="euc-kr")

#NAN은 0으로 바꾸기, 무시하고 주차별 데이터 가공
df = df.fillna(0)
#print(df)

########## 2010년 (~10/24) index:206

#1주차 강수량 평균 : 0. (이틀치니까) 미리 넣기
datas10 = pd.DataFrame()
rn_df = pd.DataFrame({'강수량(mm)': [0]})
datas10 = pd.concat([datas10, rn_df])

#2주차부터 계산(2010년 끝까지) range에서는 범위 포함 안 되니까 -1 처리.
#x=2
for i in range(2, 205, 7):
    avg_rn = (df.loc[i:i+6,"강수량(mm)"].sum())/7
    #(avg_rn)
    #print(avg_rn)
    rn_df = pd.DataFrame({'강수량(mm)':[avg_rn]})
    datas10 = pd.concat([datas10, rn_df])
    avg_rn=0
    #x+=1

print(datas10)

########## 2011년 : 4월3일 ~ 10월24일
#1주차 평균 강수량 (이틀치니까) 미리 넣기
datas11 = pd.DataFrame()
rn_df = pd.DataFrame({'강수량(mm)': [(df.loc[367:368,"강수량(mm)"].sum())/2]}) #(367,368)
datas11 = pd.concat([datas11, rn_df])

#2주차부터 계산(2011년 끝까지(10/24)) range에서는 범위 포함 안 되니까 -1 처리.
#x=2
for i in range(369, 570, 7):
    avg_rn = (df.loc[i:i+6,"강수량(mm)"].sum())/7
    #print(avg_rn)
    rn_df = pd.DataFrame({'강수량(mm)':[avg_rn]})
    datas11 = pd.concat([datas11, rn_df])
    avg_rn=0
    #x+=1

print(datas11)

########## 2012년 : 4월14일 ~ 10월21일
#1주차 평균 강수량 (이틀치니까) 미리 넣기
datas12 = pd.DataFrame()
rn_df = pd.DataFrame({'강수량(mm)': [(df.loc[742:743,"강수량(mm)"].sum())/2]}) #(742,743)
datas12 = pd.concat([datas12, rn_df])

#2주차부터 계산(2012년 끝까지(10/21)) range에서는 범위 포함 안 되니까 -1 처리.
#x=2
for i in range(744, 933, 7):
    avg_rn = (df.loc[i:i+6,"강수량(mm)"].sum())/7
    #print(avg_rn)
    rn_df = pd.DataFrame({'강수량(mm)':[avg_rn]})
    datas12 = pd.concat([datas12, rn_df])
    avg_rn=0
    #x+=1

print(datas12)

########## 2013년 : 3월31일 ~ 10월28일
#1주차 평균 강수량 (이틀치니까) 미리 넣기
datas13 = pd.DataFrame()
rn_df = pd.DataFrame({'강수량(mm)': [(df.loc[1093:1094,"강수량(mm)"].sum())/2]}) #(1093,1094)
datas13 = pd.concat([datas13, rn_df])

#2주차부터 계산(2013년 끝까지(10/28)) range에서는 범위 포함 안 되니까 -1 처리.
#x=2
for i in range(1095, 1305, 7):
    avg_rn = (df.loc[i:i+6,"강수량(mm)"].sum())/7
    #print(avg_rn)
    rn_df = pd.DataFrame({'강수량(mm)':[avg_rn]})
    datas13 = pd.concat([datas13, rn_df])
    avg_rn=0
    #x+=1

print(datas13)

########## 2014년 : 4월6일 ~ 10월27일
#1주차 평균 강수량 (이틀치니까) 미리 넣기
datas14 = pd.DataFrame()
rn_df = pd.DataFrame({'강수량(mm)': [(df.loc[1464:1465,"강수량(mm)"].sum())/2]}) #(1464,1465)
datas14 = pd.concat([datas14, rn_df])

#2주차부터 계산(2014년 끝까지(10/27)) range에서는 범위 포함 안 되니까 -1 처리.
#x=2
for i in range(1466, 1669, 7):
    avg_rn = (df.loc[i:i+6,"강수량(mm)"].sum())/7
    #print(avg_rn)
    rn_df = pd.DataFrame({'강수량(mm)':[avg_rn]})
    datas14 = pd.concat([datas14, rn_df])
    avg_rn=0
    #x+=1

print(datas14)

########## 2015년 : 3월31일 ~ 10월28일
#1주차 평균 강수량 (이틀치니까) 미리 넣기
datas15 = pd.DataFrame()
rn_df = pd.DataFrame({'강수량(mm)': [(df.loc[1823:1824,"강수량(mm)"].sum())/2]}) #(1823,1824)
datas15 = pd.concat([datas15, rn_df])

#2주차부터 계산(2015년 끝까지(10/28)) range에서는 범위 포함 안 되니까 -1 처리.
#x=2
for i in range(1825, 2035, 7):
    avg_rn = (df.loc[i:i+6,"강수량(mm)"].sum())/7
    #print(avg_rn)
    rn_df = pd.DataFrame({'강수량(mm)':[avg_rn]})
    datas15 = pd.concat([datas15, rn_df])
    avg_rn=0
    #x+=1

print(datas15)

########## 2016년 : 4월5일 ~ 10월26일
#1주차 평균 강수량 (이틀치니까) 미리 넣기
datas16 = pd.DataFrame()
rn_df = pd.DataFrame({'강수량(mm)': [(df.loc[2194:2195,"강수량(mm)"].sum())/2]}) #(2194,2195)
datas16 = pd.concat([datas16, rn_df])

#2주차부터 계산(2016년 끝까지(10/26)) range에서는 범위 포함 안 되니까 -1 처리.
#x=2
for i in range(2196, 2399, 7):
    avg_rn = (df.loc[i:i+6,"강수량(mm)"].sum())/7
    #print(avg_rn)
    rn_df = pd.DataFrame({'강수량(mm)':[avg_rn]})
    datas16 = pd.concat([datas16, rn_df])
    avg_rn=0
    #x+=1

print(datas16)

########## 2017년 : 3월27일 ~ 10월31일
#1주차 평균 강수량 (이틀치니까) 미리 넣기
datas17 = pd.DataFrame()
rn_df = pd.DataFrame({'강수량(mm)': [(df.loc[2550:2551,"강수량(mm)"].sum())/2]}) #(2550,2551)
datas17 = pd.concat([datas17, rn_df])

#2주차부터 계산(2017년 끝까지(10/31)) range에서는 범위 포함 안 되니까 -1 처리.
#x=2
for i in range(2552, 2769, 7):
    avg_rn = (df.loc[i:i+6,"강수량(mm)"].sum())/7
    #print(avg_rn)
    rn_df = pd.DataFrame({'강수량(mm)':[avg_rn]})
    datas17 = pd.concat([datas17, rn_df])
    avg_rn=0
    #x+=1

print(datas17)

########## 2018년 : 4월2일 ~ 10월30일
#1주차 평균 강수량 (이틀치니까) 미리 넣기
datas18 = pd.DataFrame()
rn_df = pd.DataFrame({'강수량(mm)': [(df.loc[2921:2922,"강수량(mm)"].sum())/2]}) #(2921,2922)
datas18 = pd.concat([datas18, rn_df])

#2주차부터 계산(2018년 끝까지(10/30)) range에서는 범위 포함 안 되니까 -1 처리.
#x=2
for i in range(2923, 3133, 7):
    avg_rn = (df.loc[i:i+6,"강수량(mm)"].sum())/7
    #print(avg_rn)
    rn_df = pd.DataFrame({'강수량(mm)':[avg_rn]})
    datas18 = pd.concat([datas18, rn_df])
    avg_rn=0
    #x+=1

print(datas18)

########## 2019년 : 4월1일 ~ 10월30일
#1주차 평균 강수량 (이틀치니까) 미리 넣기
datas19 = pd.DataFrame()
rn_df = pd.DataFrame({'강수량(mm)': [(df.loc[3285:3286,"강수량(mm)"].sum())/2]}) #(3285,3286)
datas19 = pd.concat([datas19, rn_df])

#2주차부터 계산(2019년 끝까지(10/30)) range에서는 범위 포함 안 되니까 -1 처리.
#x=2
for i in range(3287, 3498, 7):
    avg_rn = (df.loc[i:i+6,"강수량(mm)"].sum())/7
    #print(avg_rn)
    rn_df = pd.DataFrame({'강수량(mm)':[avg_rn]})
    datas19 = pd.concat([datas19, rn_df])
    avg_rn=0
    #x+=1

print(datas19)

########## 2020년 : 3월30일 ~ 11월5일
#1주차 평균 강수량 (이틀치니까) 미리 넣기
datas20 = pd.DataFrame()
rn_df = pd.DataFrame({'강수량(mm)': [(df.loc[3649:3650,"강수량(mm)"].sum())/2]}) #(3649,3650)
datas20 = pd.concat([datas20, rn_df])

#2주차부터 계산(2020년 끝까지(11/5)) range에서는 범위 포함 안 되니까 -1 처리.
#x=2
for i in range(3651, 3872, 7):
    avg_rn = (df.loc[i:i+6,"강수량(mm)"].sum())/7
    #print(avg_rn)
    rn_df = pd.DataFrame({'강수량(mm)':[avg_rn]})
    datas20 = pd.concat([datas20, rn_df])
    avg_rn=0
    #x+=1

print(datas20)

########## 2021년 : 3월29일 ~ 10월26일
#1주차 평균 강수량 (이틀치니까) 미리 넣기
datas21 = pd.DataFrame()
rn_df = pd.DataFrame({'강수량(mm)': [(df.loc[4013:4014,"강수량(mm)"].sum())/2]}) #(4013,4014)
datas21 = pd.concat([datas21, rn_df])

#2주차부터 계산(2021년 끝까지(10/26)) range에서는 범위 포함 안 되니까 -1 처리.
#x=2
for i in range(4015, 4225, 7):
    avg_rn = (df.loc[i:i+6,"강수량(mm)"].sum())/7
    #print(avg_rn)
    rn_df = pd.DataFrame({'강수량(mm)':[avg_rn]})
    datas21 = pd.concat([datas21, rn_df])
    avg_rn=0
    #x+=1

print(datas21)

########## 2022년 : 4월4일 ~ 10월25일

#1주차 평균 강수량 (이틀치니까) 미리 넣기
datas22 = pd.DataFrame()
rn_df = pd.DataFrame({'강수량(mm)': [(df.loc[4384:4385,"강수량(mm)"].sum())/2]}) #(4384,4385)
datas22 = pd.concat([datas22, rn_df])

#2주차부터 계산(2022년 끝까지(10/25)) range에서는 범위 포함 안 되니까 -1 처리.
x=2
for i in range(4386, 4575, 7):
    avg_rn = (df.loc[i:i+6,"강수량(mm)"].sum())/7
    #print(avg_rn)
    rn_df = pd.DataFrame({'강수량(mm)':[avg_rn]})
    datas22 = pd.concat([datas22, rn_df])
    avg_rn=0
    x+=1

print(datas22)

sum_df= pd.DataFrame()
sum_df = pd.concat([datas10, datas11, datas12, datas13, datas14, datas15,
                    datas16, datas17, datas18, datas19, datas20,
                    datas21, datas22])
sum_df.to_excel('./r_geumsan.xlsx', sheet_name='new_name')