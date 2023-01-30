import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("data1.csv")
print("数据集大小：",data.shape)
print(data.info())
data.head()
data.tail()
yes=0
no=0
for i in data[pd.notnull(data['shot_made_flag'])]['shot_made_flag']:
    if i==1.0:
        yes+=1
    else:
        no+=1

plt.bar([0,1], [yes,no])
plt.xticks([0,1])
plt.show()

print('命中次数：',yes)
print('未命中次数：',no)


#保留标签缺失值的数据
data_no = data[pd.isnull(data['shot_made_flag'])]
print(data_no.shape)

#保留标签不为缺失值的数据
data = data[pd.notnull(data['shot_made_flag'])]
print(data.shape)
"""
# 分配画布大小
plt.figure(figsize=(10, 10))

plt.subplot(1, 2, 1)
# alpha为不透明度，loc_x，loc_y为科比投篮的位置
plt.scatter(data.loc_x, data.loc_y, color='g', alpha=0.05)
plt.title('loc_x and loc_y')

plt.subplot(1, 2, 2)
# lat为纬度，lon为经度
plt.scatter(data.lon, data.lat, color='b', alpha=0.05)
plt.title('lat and lon')

plt.show()
"""
#原始特征中既有分钟又有秒，所以可以把这两组特征进行合并
data['remain_time'] = data['minutes_remaining']*60 + data['seconds_remaining']
data['remain_time'][:5]

#shot_zone_basic'，'shot_zone_basic'，'shot_zone_range' 这三个特征为投篮位置的不同表示，比较一下
#画图分析，其形状与坐标位置差不多，所以我选用更加准确的坐标来表示投篮位置
import matplotlib.cm as cm

plt.figure(figsize=(20, 10))


# data.groupyby(feature),是将数据根据feature里的类进行分类
def scatterbygroupby(feature):
    alpha = 0.1
    gb = data.groupby(feature)
    cl = cm.rainbow(np.linspace(0, 1, len(gb)))
    for g, c in zip(gb, cl):
        plt.scatter(g[1].loc_x, g[1].loc_y, color=c, alpha=alpha)


plt.subplot(1, 3, 1)
scatterbygroupby('shot_zone_basic')
plt.title('shot_zone_basic')

plt.subplot(1, 3, 2)
scatterbygroupby('shot_zone_range')
plt.title('shot_zone_range')

plt.subplot(1, 3, 3)
scatterbygroupby('shot_zone_area')
plt.title('shot_zone_area')
#plt.show()


drops = ['combined_shot_type',  'shot_zone_area', 'shot_zone_range', 'shot_zone_basic', \
          'lon', 'lat', 'seconds_remaining', 'minutes_remaining', \
         'shot_distance', 'game_event_id', 'game_id', 'game_date','season']
for drop in drops:
    data = data.drop(drop, 1)
data.head()
#将文字特征转换为数字特征,使用one-hot编码来实现
a = ['action_type', 'shot_type', 'opponent']
for i in a:
    #使用one-hot编码，将a中的特征里的属性值都当作新的特征附在数据的列上，特征名为前缀prefix加上该属性名
    data = pd.concat([data, pd.get_dummies(data[i], prefix=i)], 1)
    data = data.drop(i, 1) #0-行，1-列
data.head()

data.shape
data['shot_made_flag'].shape
data.to_csv("./data_processed.csv", encoding="utf-8-sig", mode="w", header=True, index=False)
data = pd.read_csv("data_processed.csv")
data_label = data['shot_made_flag']
#重新读入数据
data = pd.read_csv("data_processed.csv")
#显示大小
print("数据集大小:",data.shape)
# 数据集详细信息
print(data.info())

#分离特征和标签
data_feature = data.drop('shot_made_flag',1)
data_label = data['shot_made_flag']
data_label.shape

no=0
yes=0
for i in data['shot_made_flag']:
    if i==0.0:
        no+=1
    else:
        yes+=1
print('没有投进的次数：',no)
print('投进的次数：',yes)

data_label = np.array(data_label)
data_label.shape

#数据标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_feature = scaler.fit_transform(data_feature)
data_feature = pd.DataFrame(data_feature)
data_feature.head()
data_feature.to_csv("./data_feature_standard.csv", encoding="utf-8-sig", mode="w", header=True, index=False)
data_feature = pd.read_csv("data_feature_standard.csv")

knn_data = data_feature
knn_label = data_label
#构建训练（train）和测试数据(test)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(knn_data,knn_label, random_state=2020, test_size=0.25)
#构建和训练模型
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=3, p=2,
                     weights='uniform')

knn.predict(X_test)
print(knn.predict_proba(X_test))

# 参数调优
from sklearn.model_selection import cross_val_score
from time import time
import datetime
k_range = range(1,21,2)
cv_scores = []
time0 = time()
for n in k_range:
    knn = KNeighborsClassifier(n_neighbors=n)
    scores = cross_val_score(knn,X_train,y_train,cv=10,scoring='accuracy')
    cv_scores.append(scores.mean())
