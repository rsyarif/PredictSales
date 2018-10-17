
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

from utility.ML import ML

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
item_cat_count_feat=True

target = 'shop_item_cnt_month'
# target = 'shop_item_cnt_month_diff(0-1)'

#these columns will be dropped in createTrain/Val/Test
col_targets=[
             'shop_item_cnt_month',
             # 'shop_cnt_month',
             # 'item_cnt_month',
             # 'item_cat_cnt_month',
            ]
if(diff):
    col_targets+[
#                  'shop_item_cnt_month_diff(0-1)',
#                  'shop_cnt_month_diff(0-1)',
#                  'item_cnt_month_diff(0-1)',
#                  'item__cat_cnt_month_diff(0-1)',
                ]
if(diffRel):
    col_targets+[
#                  'shop_item_cnt_month_(0-1)/1',
#                  'shop_cnt_month_(0-1)/1',
#                  'item_cnt_month_(0-1)/1',
#                  'item_cat_cnt_month_(0-1)/1',
                ]

####### mean Encode (with Reg) #####

meanEncode=True #this is just necessary condition for mean encoding. but need to turn on individual switches below to include columsn of target encoding.
meanEncodeCol=[ #this is for lag features.
             'shop',
             'item',
             # 'item_cat',
            ]

agg_targ = {'item_cnt_day':'sum'} #target_encoding!
# agg_targ = {'item_cnt_day':'mean'} #target_encoding!

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
        'targEnc_to_Reg':targEnc_to_Reg,
        'NaN_targEnc':NaN_targEnc,
        } 


#clipTarget = True
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

x_train,x_val,x_test = Sets.addPriceRange(x_train,x_val,x_test)
y_train,y_val = Sets.clipSalesCount(y_train,y_val,lowerClip,upperClip)

#combine 2013+2014
# x_train  = pd.concat([x_train, x_val], ignore_index=True)
# y_train  = pd.concat([y_train, y_val], ignore_index=True)
# print x_train.shape

x_train,x_val,x_test = Sets.mapTargetEnc(x_train,y_train,x_val,x_test,Regularize=Regularize)

# dropping unnecessary columns
x_train = x_train.drop(columns=['shop_item_id'])
x_val = x_val.drop(columns=['shop_item_id'])
x_test = x_test.drop(columns=['shop_item_id'])
# make sure features are the same across sets
x_train = x_train.drop(columns=list(set(x_train.columns.values)-set(x_test.columns.values)))
x_val = x_val.drop(columns=list(set(x_val.columns.values)-set(x_test.columns.values)))

print 'x_train.shape:',x_train.shape
print 'x_val.shape:',x_val.shape
print 'x_test.shape:',x_test.shape 
print 'x_test columns:' 
for i,col in enumerate(x_test.columns.values): print ' '*3,i,col 
print
assert (set(x_train.columns.values)-set(x_test.columns.values)==set([])), "train/val has more features than test!"

# # Model Training

dataset = {
            'x_train':x_train,
            'x_val':x_val,
            'x_test':x_test,
            'y_train':y_train,
            'y_val':y_val,
            }

ml = ML(**dataset)

model,evals_result = ml.runBDT_lightgbm(
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
                                                        })

pred = ml.predict(model)

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

### adhoc scaling
#pred_submit = np.floor(pred_submit)
#pred_submit = (pred_submit)*0.25 
# print 'total sales pred:',np.sum(pred_submit),', mean:',np.mean(pred_submit)
# hist_pred = plt.hist(pred_submit,nbins,log=True)
# print pred_submit


# # Validation with yearly trend

y_test = pd.DataFrame(pred_submit,columns=['item_cnt_month'])
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

