import pandas as pd
import numpy as np
import re

# 提取特征
def prepare(df):
    if len(df.columns) == 7:
        df = df.rename(columns={0:'uid',1:'mid',2:'datetime',3:'f',4:'c',5:'l', 6:'content'})
    else:
        df = df.rename(columns={0:'uid',1:'mid',2:'datetime',3:'content'})
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['weekday'] = df["datetime"].apply(lambda x: x.weekday())
    df['hour'] = df["datetime"].apply(lambda x: x.hour)
    return df

def check_ads(x):
    if x is np.nan:
        return 0
    if x.find('快的打车') != -1:
        return 1
    if x.find('红包') != -1:
        return 1
    if x.find('领取') != -1:
        return 1
    if x.find('你也来试试手气') != -1:
        return 1
    if x.find('超赞的文件') != -1:
        return 1
    if x.find('链接下载') != -1:
        return 1
    if x.find('开始报名') != -1:
        return 1
    return 0

def check_share(x):
    if x is np.nan:
        return 0
    if x.find('我分享了') != -1:
        return 1
    if x.find('分享自') != -1:
        return 1
    if x.find('我上传了') != -1:
        return 1
    if x.find('我更新了') != -1: 
        return 1
    if x.find('照片到专辑') != -1:
        return 1
    return 0

def check_IT(x):
    if x is np.nan:
        return 0
    if x.find('IT') != -1:
        return 1
    if x.find('CSDN') != -1:
        return 1
    return 0

# 手动提取关键字特征
def add_features(data):
    data['content'] = data['content'].fillna("")
    data['c_has_link'] = data['content'].str.contains('http', na=False).astype(int)
    data['c_has_at'] = data['content'].str.contains('@', na=False).astype(int)
    data['c_has_ex'] = data['content'].str.contains('\[', na=False).astype(int)
    # new
    data['c_has_video'] = data['content'].str.contains('视频', na=False).astype(int)
    data['c_has_ads'] = data['content'].apply(check_ads)
    data['c_has_share'] = data['content'].apply(check_share)
    data['c_has_it'] = data['content'].apply(check_IT)
    data['c_has_topic'] = data['content'].apply(lambda x: 0 if len(re.compile(r'[#【《](.*?)[#】》]',re.S).findall(x)) == 0 else 1)
    return data

# 本地评分
def do_score(real_data, predict_data):
    d_f = ((predict_data['f'] - real_data['f'])/(real_data['f'] + 5.0)).apply(lambda x: abs(x))
    d_c = ((predict_data['c'] - real_data['c'])/(real_data['c'] + 3.0)).apply(lambda x: abs(x))
    d_l = ((predict_data['l'] - real_data['l'])/(real_data['l'] + 3.0)).apply(lambda x: abs(x))
    count_i = real_data['f'] + real_data['l'] + real_data['c']
    precision = 1 - 0.5 * d_f - 0.25 * d_c - 0.25 * d_l
    sign = np.sign(precision - 0.8).apply(lambda x: 0 if x == -1 else 1)
    count_i[count_i > 100] = 100
    count_1 = sum((count_i + 1) * sign)
    count_2 = sum(count_i + 1)
    return count_1/count_2