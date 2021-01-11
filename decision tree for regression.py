import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('day.csv',
                    usecols=['season','holiday','weekday','workingday','weathersit','cnt']).sample(frac=1)
#
#x=dataset.iloc[:,:-1].values
#y=dataset.iloc[:,-1].values

#dataset=pd.read_csv('d_d.csv')


mean_data=np.mean(dataset[dataset.columns[-1]])

def var(data,split_attribute_name,target_name):
    variance_value=0
    feature_values=np.unique(data[split_attribute_name])
    for value in feature_values:
        subset=data.query('{0}=={1}'.format(split_attribute_name,value)).reset_index()
        w_var=len(subset)/len(data)*np.var(subset[target_name],ddof=1)
        if np.isnan(w_var):
           w_var=np.nan_to_num(0)
        variance_value+=w_var      
    return variance_value

def regression(data,original_data,features,target_name,min_instances,parent_node=None):
    if len(data)<=min_instances:
        return np.mean(data[target_name])
    elif len(data)==0:
        return np.mean(original_data[target_name])
    elif len(features)==0:
        return parent_node
    else:
        parent_node=np.mean(data[target_name])
        
        item_vals=[var(data,feature,target_name) for feature in features]
        best_feature_index=np.argmin(item_vals)
        best_feature=features[best_feature_index]
        
        tree={best_feature:{}}
        
        features=[feature for feature in features if feature!=best_feature]
        for val in np.unique(data[best_feature]):
            sub_data=data.where(data[best_feature]==val).dropna()
            sub_tree=regression(sub_data,original_data,features,target_name,min_instances,parent_node)
            tree[best_feature][val]=sub_tree
        return tree
    
        
        
def predict(tree,query,default=mean_data):
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                res=tree[key][query[key]]
            except:
                return default
            res=tree[key][query[key]]
            if isinstance(res,dict):
                return predict(res,query,default)
            else:
                return res
        
        
from sklearn.model_selection import *

train,test=train_test_split(dataset,test_size=0.2,random_state=0)


tree=regression(train,train,train.columns[:-1],train.columns[-1],1)

queries=test.iloc[:,:-1].to_dict(orient='records')

y_predict=[]
for q in queries:
    y_predict.append(predict(tree,q))
    

err=np.sqrt(np.sum((np.array(y_predict)-np.array(test[test.columns[-1]]))**2)/len(test))
        
        













