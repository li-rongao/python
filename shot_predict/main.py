
import pandas as pd
#读取数据，并返回一个DataFrame对象
namea="data.csv"
nameb="Test.csv"
tablea=pd.read_csv(namea)
tableb=pd.read_csv(nameb)
result=pd.concat([tablea,tableb])
result.reset_index(drop=True,inplace=True)
result.to_csv("kobe.csv",index=False)


raw = pd.read_csv("kobe.csv")
#把剩余时间的分钟数和秒数合并到一列
raw['remaining_time'] = raw['minutes_remaining']*60 + raw['seconds_remaining']
#保留有是否命中值的行，0未命中1命中
kobe = raw[pd.notnull(raw['shot_made_flag'])]
print(kobe.shape)

from matplotlib import pyplot as plt
alpha = 0.02

#设置效果图的属性，figsize，效果图占据的位置大小
plt.figure(figsize=(10,10))
#选定子图，后面的绘图属性都作用于该子图：121含义：把图视为1行2列的划分，本图在这个划分的第一个位置
plt.subplot(121)
#绘制散点图
plt.scatter(kobe.loc_x, kobe.loc_y, color='R', alpha=alpha)
#设置标题
plt.title('loc_x and loc_y')
#选定另一个子图
plt.subplot(122)
plt.scatter(kobe.lon, kobe.lat, color='B', alpha=alpha)
plt.title('lon and lag')
plt.plot()


#Dataframe.属性：获取表格中的某一列，返回pandas.core.series.Series类对象
#调用Series.unique()，去重统计列中有哪些值
print(kobe.shot_type.unique())
#调用Series.value_counts()，去重统计每个不同值出现了多少次
print(kobe.shot_type.value_counts())
print(kobe['season'].unique())
#过滤掉原数据中的‘-’，只保留‘赛季’
kobe['season'] = kobe['season'].apply(lambda x: int(x.split('-')[1]))
print(kobe['season'].unique())

import matplotlib.cm as cm
import numpy as np

plt.figure(figsize=(20, 10))


def scatter_plot_by_category(feat):
    alpha = 0.1
    # 把数据集按照某列的取值不同，分为多个组
    gs = kobe.groupby(feat)
    cs = cm.rainbow(np.linspace(0, 1, len(gs)))
    for g, c in zip(gs, cs):
        plt.scatter(g[1].loc_x, g[1].loc_y, color=c, alpha=alpha)


plt.subplot(131)
scatter_plot_by_category('shot_zone_area')
plt.title('shot_zone_area')

plt.subplot(132)
scatter_plot_by_category('shot_zone_basic')
plt.title('shot_zone_basic')

plt.subplot(133)
scatter_plot_by_category('shot_zone_range')
plt.title('shot_zone_range')

drops = [ 'shot_zone_basic', 'shot_zone_range',
         'lon', 'lat', 'seconds_remaining', 'minutes_remaining', 'shot_distance', 'game_event_id', 'game_id',
         'game_date']
for drop in drops:
    raw = raw.drop(drop, 1)
category_vars = ['action_type', 'combined_shot_type', 'shot_type', 'opponent', 'season', 'shot_zone_area']
for var in category_vars:
    raw = pd.concat([raw, pd.get_dummies(raw[var], prefix=var)], 1)
    raw = raw.drop(var, 1)

#用原始数据中，命中数据列的有效值数据行作为训练数据
#用NAN，即无效值数据行作为测试数据
train_kobe = raw[pd.notnull(raw.shot_made_flag)]
train_label = train_kobe.shot_made_flag
train_kobe = train_kobe.drop('shot_made_flag', 1)
test_kobe = raw[pd.isnull(raw.shot_made_flag)]
test_kobe = test_kobe.drop('shot_made_flag', 1)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
import time

min_score = 100000
best_m = 0
best_n = 0
scores = []
ranges = np.logspace(0, 2, num=3).astype(int)
kf = KFold(n_splits=10, shuffle=True)
for m in ranges:
    for n in ranges:
        print('the max depth: ', m)
        t1 = time.time()
        rfc_score = 0
        rfc = RandomForestClassifier(n_estimators=n, max_depth=m)
        for train_k, test_k in kf.split(train_kobe):
            rfc.fit(train_kobe.iloc[train_k], train_label.iloc[train_k])
            pred = rfc.predict(train_kobe.iloc[test_k])
            rfc_score += log_loss(train_label.iloc[test_k], pred)/10
        scores.append(rfc_score)
        if rfc_score < min_score:
            min_score = rfc_score
            best_m = m
            best_n = n
        t2 = time.time()
        print('Done processing {0} depth {1} trees ({2:.3f}sec)'.format(m, n, t2-t1))
print(best_m, best_n, min_score)

rfc = RandomForestClassifier(n_estimators=best_n, max_depth=best_m)
rfc.fit(train_kobe, train_label)

np.savetxt('result.txt',rfc.predict_proba(test_kobe))
