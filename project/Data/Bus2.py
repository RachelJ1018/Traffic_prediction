import numpy as np
import pandas as pd
import tensorflow as tf
import os
from sklearn.preprocessing import OneHotEncoder  
from sklearn.preprocessing import LabelEncoder  


from xgboost import XGBClassifier
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score

 
#data_list=[]
#for info in os.listdir('D:/test_pd'):
#    domain=os.path.abspath(r'D:/test_pd')
#    info=os.path.join(domain,info)
#    data=pd.read_csv(info)
#    data_list.append(data)
#add_data=pd.concat(data_list)
#print(add_data.info())
#print(add_data['line_no'].value_counts())
#print(add_data['terminal_no'].value_counts())

#print(add_data['day_of_the_week'].value_counts())
#print(add_data['next_station_no'].value_counts())
#my_data=pd.read_csv('C:/Users/Administrator/Desktop/3/sort_second/')
data=pd.read_csv('D:/sortsecond/1_down_second.csv')
#data_nextstation=data.groupby(['terminal_no'],as_index=False)[
#print(data_nextstation)
#data_M=data['next_station_no'].groupby([data['terminal_no'],data['next_station_no']])
#print(data_M.count())
data_T=data.groupby(['terminal_no','next_station_no'],as_index=False)[['time_used_to_next_station']].mean()
print(data_T.head(10))
terminal=LabelEncoder().fit_transform(data['terminal_no'])
terminal_ohe_ = OneHotEncoder(sparse=False).fit_transform(terminal.reshape((-1,1))) 
#print(terminal_ohe_.shape)
station=LabelEncoder().fit_transform(data['next_station_no'])
station_ohe_ = OneHotEncoder(sparse=False).fit_transform(station.reshape((-1,1))) 


import numpy as np
import pandas as pd
import tensorflow as tf
import os
from sklearn.preprocessing import OneHotEncoder  
from sklearn.preprocessing import LabelEncoder  


from xgboost import XGBClassifier
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score

data=pd.read_csv('Data/1_down_second.csv')

data_T=data.groupby(['line_no','terminal_no','next_station_no'],as_index=False)[['time_used_to_next_station']].mean()
print(data_T.head(10))
line=LabelEncoder().fit_transform(data_T['line_no'])
line_ohe_= OneHotEncoder(sparse=False).fit_transform(line.reshape((-1,1))) 
terminal=LabelEncoder().fit_transform(data_T['terminal_no'])
terminal_ohe_ = OneHotEncoder(sparse=False).fit_transform(terminal.reshape((-1,1))) 
#print(terminal_ohe_.shape)
station=LabelEncoder().fit_transform(data_T['next_station_no'])
station_ohe_ = OneHotEncoder(sparse=False).fit_transform(station.reshape((-1,1))) 
# day_week=LabelEncoder().fit_transform(data['day_of_the_week'])
# day_ohe_ = OneHotEncoder(sparse=False).fit_transform(day_week.reshape((-1,1))) 
x=np.column_stack((line_ohe_,terminal_ohe_,station_ohe_ ))
y=data_T['time_used_to_next_station']
from sklearn import neighbors, datasets, metrics, preprocessing, tree
from sklearn.model_selection import StratifiedShuffleSplit,train_test_split
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
X_train, X_test,y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=42)
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor 
pre_model = DecisionTreeRegressor()
# Fit model
pre_model.fit(X_train, y_train)

# get predicted prices on validation data
val_predictions = pre_model.predict(X_test)
print(mean_absolute_error(y_test, val_predictions))
def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    new_model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    new_model.fit(predictors_train, targ_train)
    preds_val = new_model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, X_train, X_test,y_train, y_test)
    print("Max leaf nodes: %d \t\t  Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
    
from sklearn.ensemble import RandomForestRegressor

forest_model1 = RandomForestRegressor(min_impurity_decrease=0.2,n_estimators=50,min_samples_leaf=1,verbose=3, n_jobs=1)
forest_model1
forest_model1.fit(X_train, y_train)
pre_preds = forest_model1.predict(X_test)
print(mean_absolute_error(y_test, pre_preds))
from xgboost import XGBRegressor
my_model = XGBRegressor(n_estimators=100,learning_rate=0.5,max_depth=9,silent=True,colsample_bylevel=0.5)

# Adding silent=True to avoid printing out updates with each cycle
my_model.fit(X_train, y_train, verbose=False)
predictions = my_model.predict(X_test)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, y_test)))

predict=pd.read_csv('Data/toBePredicted_forUser.csv')
predict.head()
predict.set_index
predict.head()

station_series=[]
for i in range(len(predict)):
    m=predict['pred_start_stop_ID'].loc[i]
    n=predict['pred_end_stop_ID'].loc[i]
    i_series=[]
    for j in range(m,n):
#          predict.iloc[i,'M%d'%j]=j
          i_series.append(j)
    station_series.append(i_series)
    
    
predict['Middle_station']=station_series
predict.head()

col_selection=['O_LINENO','O_TERMINALNO','O_UP','Middle_station']
pre_data=predict.loc[:,col_selection]

y_array=[]
for i in range(len(pre_data)):
    for j in range(len(pre_data.Middle_station.values[i])):
            y_list=[]
            y_list.append(pre_data.O_LINENO.values[i])
            y_list.append(pre_data.O_TERMINALNO.values[i])
            y_list.append(pre_data.O_UP.values[i])
            y_list.append(pre_data.Middle_station.values[i][j])
            y_array.append(y_list)
        
        
pre_y=pd.DataFrame(y_array,columns=['line_no','terminal_no','O_UP','next_station_no'])



data=pd.read_csv('Data/pre_result_xgboost2.csv',names=['time'])
predict=pd.read_csv('Data/toBePredicted_forUser.csv')
station_series=[]
for i in range(len(predict)):
    m=predict['pred_start_stop_ID'].loc[i]
    n=predict['pred_end_stop_ID'].loc[i]+1
    i_series=[]
    for j in range(m,n):
#          predict.iloc[i,'M%d'%j]=j
          i_series.append(j)
    station_series.append(i_series)
    
predict['Middle_station']=station_series
predict.head()
predict.Middle_station.loc[1]
col_selection=['O_DATA','O_LINENO','O_TERMINALNO','predHour','pred_start_stop_ID','pred_end_stop_ID']
pre_data=predict.loc[:,col_selection]
flag_t=0
Timestamps_series=[]
for i in range(len(pre_data)):
    m=predict['pred_start_stop_ID'].loc[i]
    n=predict['pred_end_stop_ID'].loc[i]+1
    i_series=[]
    sum_time=0
    for j in range(m,n):
#          predict.iloc[i,'M%d'%j]=j
          sum_time=sum_time+data['time'].values[flag_t]
          i_series.append(sum_time)
          flag_t=flag_t+1
    Timestamps_series.append(i_series)
    
pre_data['TimeStamps']=Timestamps_series
pre_data.to_csv('Data/result_xgboost2.csv')
import csv
def createListCSV(fileName="", dataList=[]):

    with open(fileName, "w",newline='') as csvFile:
        csvWriter = csv.writer(csvFile,delimiter=';')
        for data in dataList:
            csvWriter.writerow(data)
    csvFile.close

createListCSV("Data/test.csv", pre_data['TimeStamps'])
data_time=pd.read_csv('Data/test.csv',names=['pred_timeStamps'])
data_time.set_index
col_selection=['O_DATA','O_LINENO','O_TERMINALNO','predHour','pred_start_stop_ID','pred_end_stop_ID']
pre_data2=predict.loc[:,col_selection]
forest_result=pre_data2.join(data_time)
xx=forest_result[forest_result['O_DATA']=='10-28']
forest_result.to_csv('Data/result_xgboost.csv',index=None)