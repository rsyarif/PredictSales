
# coding: utf-8

# https://www.kaggle.com/c/competitive-data-science-final-project

# In[1]:

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

import utility.utility as ut
from preprocessing.createSets import *
from preprocessing.Sets import Sets

from sklearn.model_selection import KFold

########### switches ################################

saveName='Oct15-2018_1'
saveFolder = saveName
saveplots=False
createSubmit = False
if not os.path.exists(saveFolder) and (saveplots or createSubmit): 
    print 'creating folder:',os.getcwd()+'/'+saveFolder
    os.mkdir(saveFolder)
if(saveplots): print "SAVING PLOTS!"
if(createSubmit): print "WILL CREATE SUBMIT FILE!"


############# Options/Args #########################

verbose=False
lag_length = 3
diff = True
diffRel = True 
item_cat_count_feat=False

target = 'shop_item_cnt_month'
# target = 'shop_item_cnt_month_diff(0-1)'

#these columns will be dropped in createTrain/Val/Test
col_targets=[
             'shop_item_cnt_month',
#              'shop_cnt_month',
#              'item_cnt_month',
            ]
if(diff):
    col_targets+[
#                  'shop_item_cnt_month_diff(0-1)',
#                  'shop_cnt_month_diff(0-1)',
#                  'item_cnt_month_diff(0-1)',
                ]
if(diffRel):
    col_targets+[
#                  'shop_item_cnt_month_(0-1)/1',
#                  'shop_cnt_month_(0-1)/1',
#                  'item_cnt_month_(0-1)/1',
                ]

####### mean Encode (with Reg) #####

meanEncode=True #this is just necessary condition for mean encoding. but need to turn on individual switches below to include columsn of target encoding.
meanEncodeCol=[
             'shop',
             'item',
             'item_cat',
            ]
Regularize = True
enc_cnt_per_shop = False
enc_cnt_per_item = False
enc_cnt_per_item_cat = False
enc_priceRange = False

targEnc_to_Reg = {}
NaN_targEnc = {}

if(enc_cnt_per_shop):targEnc_to_Reg.update({'shop_id':'shop'})
if(enc_cnt_per_item):targEnc_to_Reg.update({'item_id':'item'})
if(enc_cnt_per_item_cat):targEnc_to_Reg.update({'item_category_id':'item_cat'})
if(enc_priceRange):targEnc_to_Reg.update({'price_range':'price_range'})

if(enc_cnt_per_shop):NaN_targEnc.update({'shop':-999})
if(enc_cnt_per_item):NaN_targEnc.update({'item':-999})
if(enc_cnt_per_item_cat):NaN_targEnc.update({'item_cat':-999})
if(enc_priceRange):NaN_targEnc.update({'price_range':-999})

####################################

#columns to keep
col_to_keep = [
                'shop_id',
                'item_id',
                #'item_price',
                'item_category_id',
                'item_cnt_day',
              ]

groupby_list = ['shop_id','item_id']

agg_dict = {
            #'item_price':'mean',
            'item_category_id':'mean',
            'item_cnt_day':'sum',
            }

agg_targ = {'item_cnt_day':'sum'} #target_encoding!
# agg_targ = {'item_cnt_day':'mean'} #target_encoding!

    
opt = {
        'verbose':verbose,
        'lag_length':lag_length,
        'diff':diff,
        'diffRel':diffRel,
        'item_cat_count_feat':item_cat_count_feat,
        'target':target,
        'col_to_keep':col_to_keep,
        'groupby_list':groupby_list,
        'agg_dict':agg_dict,
        'agg_targ':agg_targ,
        'col_targets':col_targets,
        'meanEncode':meanEncode,
        'meanEncodeCol':meanEncodeCol,
        } 


clipTarget = True
lowerClip = 0
upperClip = 20
if target=='shop_item_cnt_month_diff(0-1)':
    lowerClip = -20
    upperClip = 20    

########################################################


# # create train/val/test set

Sets = Sets(**opt)

dup_ids = Sets.checkDuplicates()
Sets.convertDatetime()
Sets.addItemCategoryId()
Sets.addYMcolumn()
bin_edges=[0,10,100,200,500,1000,2500,5000,20000,999999]
Sets.binPrice(bin_edges)
Sets.splitDataByYear()
data = Sets.getData()

x_train, y_train = Sets.createTrainSet()
x_val, y_val = Sets.createValSet()
x_test = Sets.createTestSet()


# # Adding price category to train,val, test

train = data['sales_train']
print 'train.shape:',train.shape

#Aggregate train by 'item_price' and take __minimum__ of price range category
train_agg = train.groupby(['item_id'], as_index=False).agg({'price_range':'min'})
train_agg[train_agg['price_range'].isna()]

#We're gonna do a hack. Change format to string, fill missing value, then change to 'category'
train_agg['price_range']=train_agg['price_range'].astype('string')
train_agg[train_agg['price_range']=='nan']
train[train['item_id']==2973]['price_range'].unique()

#It seems all of item 2973 are priced (1000,2500], let go ahead and fix the missing value
train_agg.at[2913,'price_range']='(1000,2500]'
train_agg['price_range']=train_agg['price_range'].astype('category')
train_agg[train_agg['price_range'].isna()]
train_agg[train_agg.index==2913]

#ok, we've fixed that missing value for train_agg. Now merge with x_train
x_train = pd.merge(x_train,train_agg[['item_id','price_range']],on='item_id',how='left')
x_train[x_train['price_range'].isna()]
x_val = pd.merge(x_val,train_agg[['item_id','price_range']],on='item_id',how='left')
x_val[x_val['price_range'].isna()]
x_test.shape
x_test = pd.merge(x_test,train_agg[['item_id','price_range']],on='item_id',how='left')
#print  'fraction rows with NaN price range in x_test:',1.0*x_test[x_test['price_range'].isna()].shape[0] / x_test.shape[0] 
# Two choices: 1. Ignore these small unkowns, let the BDT do its best with other known features. 2. Lets predict these new items with the average sales count from the month before, as the best case scenario.

#combine 2013+2014
# x_train  = pd.concat([x_train, x_val], ignore_index=True)
# y_train  = pd.concat([y_train, y_val], ignore_index=True)
# print x_train.shape


# # Clip y_train, y_val


nbins=20

y_train_clip = np.clip(y_train,lowerClip,upperClip)
y_val_clip = np.clip(y_val,lowerClip,upperClip)
print 'Sum y_train before clip [{}-{}]:'.format(lowerClip,upperClip),np.sum(y_train)
print 'Sum y_val before clip[{}-{}]:'.format(lowerClip,upperClip), np.sum(y_val)
if(clipTarget):
    y_train=y_train_clip
    y_val=y_val_clip
print 'Sum y_train after clip[{}-{}]:'.format(lowerClip,upperClip),np.sum(y_train)
print 'Sum y_val after clip[{}-{}]:'.format(lowerClip,upperClip), np.sum(y_val)


# # Target encode with KFold reg

#add target back to x_train
df = pd.merge(x_train,y_train.to_frame(),left_index=True,right_index=True,how='left')
#introduce price_range target encoding: price_range_cnt_month
df_temp=df.groupby('price_range',as_index=False).agg({'shop_item_cnt_month':'sum'}).rename(columns={'shop_item_cnt_month':'price_range_cnt_month'})
df = pd.merge(df,df_temp,on='price_range',how='left')

Reg=''
if(Regularize):
  if(targEnc_to_Reg.items()):print 'Regularizing target encoding!'
  Reg='_kFold' #this determines the columns to be mapped to val and test. 
  kf = KFold(5,shuffle=True,random_state=1234)
  for key,value in targEnc_to_Reg.items():
      #initialize
      df[value+'_cnt_month_kFold'] = df[value+'_cnt_month']

      #let's use median of the mean (per feat) for global stat for replacing NaN. can experiment later using min, mean, etc.
      replaceNaN = df.groupby(key)[value+'_cnt_month'].mean().median()
      NaN_targEnc.update({value:replaceNaN})
      for tr_ind,val_ind in kf.split(df):
          df_tr, df_val = df.iloc[tr_ind],df.iloc[val_ind]
          feat_target_sum = df_tr.groupby(key)['shop_item_cnt_month'].sum()
          df_val[value+'_cnt_month_kFold'] = df_val[key].map(feat_target_sum)  
          df_val[value+'_cnt_month_kFold'].fillna(replaceNaN, inplace=True)
          df.at[val_ind,value+'_cnt_month_kFold'] = df_val[value+'_cnt_month_kFold'] 
x_train = df

# map x_train targ_enc_kFol to x_val
df = x_val
for key,value in targEnc_to_Reg.items():
    print 'x_val: adding target encoding:',value+'_cnt_month'+Reg
    df_temp = x_train.groupby(key)[value+'_cnt_month'+Reg].mean()
    df_temp = df[key].map(df_temp)
    df[value+'_cnt_month'+Reg] = df_temp
    df[value+'_cnt_month'+Reg].fillna(NaN_targEnc[value], inplace=True)   
x_val = df

# map x_train targ_enc_kFol to x_test
df = x_test
for key,value in targEnc_to_Reg.items():
    print 'x_test: adding target encoding:',value
    df_temp = x_train.groupby(key)[value+'_cnt_month'+Reg].mean()
    df_temp = df[key].map(df_temp)
    df[value+'_cnt_month'+Reg] = df_temp
    df[value+'_cnt_month'+Reg].fillna(NaN_targEnc[value], inplace=True)    
x_test = df

x_train_ = x_train.drop(columns=['shop_item_id'])
x_val_ = x_val.drop(columns=['shop_item_id'])
x_test_ = x_test.drop(columns=['shop_item_id'])

x_train_ = x_train_.drop(columns=list(set(x_train_.columns.values)-set(x_test_.columns.values)))
x_val_ = x_val_.drop(columns=list(set(x_val_.columns.values)-set(x_test_.columns.values)))


print 'x_train.shape:',x_train_.shape
print 'x_val.shape:',x_val_.shape
print 'x_test.shape:',x_test_.shape 
#print 'train:',x_train_.columns.values
#print 'test:',x_test_.columns.values
assert (set(x_train_.columns.values)-set(x_test_.columns.values)==set([])), "train/val has more features than test!"
#print 'train-test:',set(x_train_.columns.values)-set(x_test_.columns.values)
#print 'val-test:',set(x_val_.columns.values)-set(x_test_.columns.values)


# # Model Training

### Boosted Decision Tree (lightgbm)

import lightgbm as lgb
from sklearn.metrics import r2_score

evals_result={}

lgb_train = lgb.Dataset(x_train_, label=y_train)
lgb_test = lgb.Dataset(x_val_, label=y_val)

lgb_params = {
               'feature_fraction': .75,
               'metric': 'rmse',
               'nthread':4, 
               'min_data_in_leaf': 2**7, 
               'bagging_fraction': 0.75,#0.75 
               'learning_rate': 0.03, 
               'objective': 'mse', 
               'bagging_seed': 2**7, 
               'num_leaves': 2**7,
               'bagging_freq':1,
               'verbose':1,
              }

num_boost_round = 1000
verbose_eval = num_boost_round/10
model = lgb.train(lgb_params, 
                  lgb_train,
                  valid_sets=[lgb_train, lgb_test],
                  valid_names=['train','eval'],
                  num_boost_round=num_boost_round,
                  evals_result=evals_result,
                  early_stopping_rounds=200,
                  verbose_eval=verbose_eval)

#print 'evals_result = ',evals_result

# print('Plot metrics recorded during training...')
# ax = lgb.plot_metric(evals_result, metric='rmse')
# if(saveplots):plt.savefig(saveFolder+"/"+"lgb_plot_metric_"+saveName+".pdf")
# #plt.show()

# print('Plot feature importances...')
# ax = lgb.plot_importance(model, max_num_features=x_test.shape[1])
# ax.figure.set_size_inches(6.4*2,4.8*3)
# if(saveplots):plt.savefig(saveFolder+"/"+"lgb_plot_importance_"+saveName+".pdf")
# plt.show()

pred_lgb = model.predict(x_train_)
print('Training R-squared for LightGBM is %f' % r2_score(y_train, pred_lgb))
pred_lgb = model.predict(x_val_)
print('Validation R-squared for LightGBM is %f' % r2_score(y_val, pred_lgb))


# # Predict with test data¶

pred = model.predict(x_test_)

pred_submit = pred

#translate back to shop_item_cnt_month in needed
if target =='shop_item_cnt_month_diff(0-1)':  
    last_month_shop_item_cnt_month = sales_2015[sales_2015['month']==(10)].groupby(groupby_list,as_index=False).agg(agg_dict).rename(columns={'item_cnt_day':'shop_item_cnt_month_lag_1'})
    last_month_shop_item_cnt_month = last_month_shop_item_cnt_month[['shop_id','item_id','shop_item_cnt_month_lag_1']].head()
    df_temp = pd.merge(x_test[['shop_id','item_id']],last_month_shop_item_cnt_month,on=('shop_id','item_id'),how='left')
    df_temp.fillna(0,inplace=True)
    df_temp.head()

    df_pred = pd.DataFrame(pred,columns=['y_pred_residual'])
    df_pred.head()

    df_pred['y_pred'] = df_temp['shop_item_cnt_month_lag_1'] + df_pred['y_pred_residual']
    df_pred.head()

    pred_submit = df_pred['y_pred'].values
    
# True target values are clipped into [0,20] range.
pred_submit = np.clip(pred_submit,0,20)
    
print 'pred_submit:',pred_submit


print 'total sales pred:',np.sum(pred_submit), ', mean:',np.mean(pred_submit)
hist_pred = plt.hist(pred_submit,nbins,log=True)


### adhoc scaling
#pred_submit = np.floor(pred_submit)
#pred_submit = (pred_submit)*0.25 
# print 'total sales pred:',np.sum(pred_submit),', mean:',np.mean(pred_submit)
# hist_pred = plt.hist(pred_submit,nbins,log=True)
# print pred_submit


# # Validation with yearly trend

y_test = pd.DataFrame(pred_submit,columns=['item_cnt_month'])
#saveName='constant_0p38'
#saveplots=True

total_item_cnt_2013 = data['sales_2013'].groupby(['date_block_num','Y_M'])['item_cnt_day'].sum().values
total_item_cnt_2014 = data['sales_2014'].groupby(['date_block_num','Y_M'])['item_cnt_day'].sum().values
total_item_cnt_2015 = data['sales_2015'].groupby(['date_block_num','Y_M'])['item_cnt_day'].sum().values
yearlySales = [total_item_cnt_2013,total_item_cnt_2014,total_item_cnt_2015]

kwargs = {'yearlySales':yearlySales,
        'y_test':y_test,
        'saveName':saveName,
        'saveplots':saveplots,
        'saveFolder':saveFolder}

# ut.plotYearly(**kwargs)
# ut.plotYearly_v2(**kwargs)
# ut.plotResidual(**kwargs)


# # Prepare submission file

test_sorted = data['test'].sort_values(by=groupby_list).reset_index(drop=True)
submit = pd.concat([test_sorted,y_test],axis=1)
submit = submit.sort_values(by="ID").reset_index(drop=True)

# sanity check
#print 'These numbers below should match:'
#print 'y_test.iloc[0] =',y_test.iloc[0].values[0] 
#print 'submit[submit["ID"]==22987] = ',submit[submit["ID"]==22987]["item_cnt_month"].values[0]

submit = submit[["item_cnt_month"]]
submit.index.name="ID"
submit.head()

submitName=''
if submitName=='': 
    submitName=saveName
if(createSubmit):submit.to_csv('submit_'+submitName+'.csv')

