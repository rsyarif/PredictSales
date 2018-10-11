import numpy as np
import pandas as pd
import os


### Initialize:
# col_to_keep = ['shop_id','item_id','item_category_id','item_cnt_day']

# groupby_list = ['shop_id','item_id']

# #agg_dict = {'item_price':'mean','item_category_id':'mean','item_cnt_day':'sum'}
# agg_dict = {'item_category_id':'mean','item_cnt_day':'sum'}
# agg_targ = {'item_cnt_day':'sum'}

# col_targets=['shop_item_cnt_month',
#              'shop_cnt_month',
#              'item_cnt_month',
#              'shop_item_cnt_month_diff(0-1)',
#              'shop_cnt_month_diff(0-1)',
#              'item_cnt_month_diff(0-1)']


def getData():

	path = '/Users/rizki/Dropbox/Coursera/AML_HowToKaggle/FinalProject/PredictSales/all/'

	items = pd.read_csv(path+'items.csv')
	item_categories = pd.read_csv(path+'item_categories.csv')
	shops = pd.read_csv(path+'shops.csv')
	sales_train = pd.read_csv(path+'sales_train.csv')
	test = pd.read_csv(path+'test.csv')

	# Format 'date' to datetime
	sales_train['date'] = pd.to_datetime(sales_train['date'],format='%d.%m.%Y')

	# Add item_category_id to sales_train
	sales_train = pd.merge(sales_train,items[['item_id','item_category_id']],on='item_id',how='left').sort_values(by='item_id')

	#add new columns: years, month, Y_M
	sales_train['year'] = sales_train['date'].dt.year
	sales_train['month'] = sales_train['date'].dt.month
	sales_train_year = sales_train['date'].dt.year.astype('string')
	sales_train_month = sales_train['date'].dt.month.astype('string')
	sales_train['Y_M'] = sales_train_year+'_'+sales_train_month

	#split by year
	sales_2013 = sales_train[sales_train['year']==2013]
	sales_2014 = sales_train[sales_train['year']==2014]
	sales_2015 = sales_train[sales_train['year']==2015]

	data = {'items':items,
			'item_categories':item_categories,
			'shops':shops,
			'sales_train':sales_train,
			'test':test,
			'sales_2013':sales_2013,
			'sales_2014':sales_2014,
			'sales_2015':sales_2015
			}

	return data

def createTrainSet(**kwargs):

	sales_2013 = kwargs['sales_2013']
	sales_2014 = kwargs['sales_2014']
	sales_2015 = kwargs['sales_2015']

	lag_length = kwargs['lag_length']
	diff = kwargs['diff']
	diffRel = kwargs['diffRel']
	item_cat_count_feat = kwargs['item_cat_count_feat']
	target = kwargs['target']

	print 'lag_length:',lag_length
	print 'diff:',diff
	print 'diffRel:',diffRel
	print 'item_cat_count_feat :',item_cat_count_feat
	print 'target:',target

	col_to_keep = kwargs['col_to_keep']
	groupby_list= kwargs['groupby_list']
	agg_dict= kwargs['agg_dict']
	agg_targ= kwargs['agg_targ']
	col_targets= kwargs['col_targets']

	x_train = sales_2013[sales_2013['month']==11]
	x_train = x_train[col_to_keep]

	#Target features:
	#agg shop_item 
	x_train_shop_item = x_train.groupby(groupby_list,as_index=False).agg(agg_dict).rename(columns={'item_cnt_day':'shop_item_cnt_month'})
	#agg shop 
	x_train_shop = x_train[['shop_id','item_cnt_day']].groupby(['shop_id'],as_index=False).agg(agg_targ).rename(columns={'item_cnt_day':'shop_cnt_month'})
	#agg item 
	x_train_item = x_train[['item_id','item_cnt_day']].groupby(['item_id'],as_index=False).agg(agg_targ).rename(columns={'item_cnt_day':'item_cnt_month'})
	#agg item_cat 
	if(item_cat_count_feat):x_train_itemcat = x_train[['item_category_id','item_cnt_day']].groupby(['item_category_id'],as_index=False).agg(agg_targ).rename(columns={'item_cnt_day':'item_cat_cnt_month'})

	#merge
	x_train_shop_item = pd.merge(x_train_shop_item,x_train_shop,on=['shop_id'],how='left')
	x_train_shop_item = pd.merge(x_train_shop_item,x_train_item,on=['item_id'],how='left')
	if(item_cat_count_feat):x_train_shop_item = pd.merge(x_train_shop_item,x_train_itemcat,on=['item_category_id'],how='left')


	#introduce lag features 10 months behind.
	for i in xrange(1,lag_length+1):
	    x_train_lag = sales_2013[sales_2013['month']==(11-i)]
	    x_train_lag = x_train_lag[col_to_keep]

	    #agg shop_item 
	    x_train_shop_item_lag = x_train_lag.groupby(groupby_list,as_index=False).agg(agg_dict).rename(columns={'item_cnt_day':'shop_item_cnt_month_lag_'+str(i)})
	    x_train_shop_item_lag.drop(columns=['item_category_id'],inplace=True)
	    #agg shop 
	    x_train_shop_lag = x_train_lag[['shop_id','item_cnt_day']].groupby(['shop_id'],as_index=False).agg(agg_targ).rename(columns={'item_cnt_day':'shop_cnt_month_lag_'+str(i)})
	    #agg item 
	    x_train_item_lag = x_train_lag[['item_id','item_cnt_day']].groupby(['item_id'],as_index=False).agg(agg_targ).rename(columns={'item_cnt_day':'item_cnt_month_lag_'+str(i)})
	    #agg item_cat 
	    if(item_cat_count_feat):x_train_itemcat_lag = x_train_lag[['item_category_id','item_cnt_day']].groupby(['item_category_id'],as_index=False).agg(agg_targ).rename(columns={'item_cnt_day':'item_cat_cnt_month_lag_'+str(i)})

	    #merge
	    x_train_shop_item = pd.merge(x_train_shop_item,x_train_shop_item_lag,on=['shop_id','item_id'],how='left')
	    x_train_shop_item = pd.merge(x_train_shop_item,x_train_shop_lag,on=['shop_id'],how='left')
	    x_train_shop_item = pd.merge(x_train_shop_item,x_train_item_lag,on=['item_id'],how='left')
	    if(item_cat_count_feat):x_train_shop_item = pd.merge(x_train_shop_item,x_train_itemcat_lag,on=['item_category_id'],how='left')
	    
	    #diffs
	    for col in ['shop_item','shop','item','item_cat']:
	        if(not item_cat_count_feat and col=='item_cat'): continue #skip item_cat if this feat is not turned on 

	        if i==1:
	            if(diff):x_train_shop_item[col+'_cnt_month_diff({}-{})'.format(str(i-1),str(i))] = x_train_shop_item[col+'_cnt_month'] - x_train_shop_item[col+'_cnt_month_lag_'+str(i)]   
	        if i>=2:
	            if(diff):x_train_shop_item[col+'_cnt_month_diff({}-{})'.format(str(i-1),str(i))] = x_train_shop_item[col+'_cnt_month_lag_'+str(i-1)] - x_train_shop_item[col+'_cnt_month_lag_'+str(i)]   
	        
	        if(diffRel):x_train_shop_item[col+'_cnt_month_({}-{})/{}'.format(str(i-1),str(i),str(i))] = x_train_shop_item[col+'_cnt_month_diff({}-{})'.format(str(i-1),str(i))] / (x_train_shop_item[col+'_cnt_month_lag_'+str(i)]+1e-7)   

	x_train_shop_item.fillna(0,inplace=True)

	#pick (meta)target            
	y_train = x_train_shop_item[target]

	#remove targets(s):
	x_train = x_train_shop_item.drop(columns=col_targets)

	print 'x_train.shape :',x_train.shape
	print 'y_train.shape :',y_train.shape


	return x_train, y_train

def createValSet(**kwargs):

	sales_2013 = kwargs['sales_2013']
	sales_2014 = kwargs['sales_2014']
	sales_2015 = kwargs['sales_2015']	

	lag_length = kwargs['lag_length']
	diff = kwargs['diff']
	diffRel = kwargs['diffRel']
	item_cat_count_feat = kwargs['item_cat_count_feat']
	target = kwargs['target']

	print 'lag_length:',lag_length
	print 'diff:',diff
	print 'diffRel:',diffRel
	print 'item_cat_count_feat :',item_cat_count_feat
	print 'target:',target

	col_to_keep = kwargs['col_to_keep']
	groupby_list= kwargs['groupby_list']
	agg_dict= kwargs['agg_dict']
	agg_targ= kwargs['agg_targ']
	col_targets= kwargs['col_targets']

	x_val = sales_2014[sales_2014['month']==11]
	x_val = x_val[col_to_keep]

	#agg shop_item 
	x_val_shop_item = x_val.groupby(groupby_list,as_index=False).agg(agg_dict).rename(columns={'item_cnt_day':'shop_item_cnt_month'})
	#agg shop 
	x_val_shop = x_val[['shop_id','item_cnt_day']].groupby(['shop_id'],as_index=False).agg(agg_targ).rename(columns={'item_cnt_day':'shop_cnt_month'})
	#agg item 
	x_val_item = x_val[['item_id','item_cnt_day']].groupby(['item_id'],as_index=False).agg(agg_targ).rename(columns={'item_cnt_day':'item_cnt_month'})
	#agg item_cat 
	if(item_cat_count_feat):x_val_itemcat = x_val[['item_category_id','item_cnt_day']].groupby(['item_category_id'],as_index=False).agg(agg_targ).rename(columns={'item_cnt_day':'item_cat_cnt_month'})

	#merge
	x_val_shop_item = pd.merge(x_val_shop_item,x_val_shop,on=['shop_id'],how='left')
	x_val_shop_item = pd.merge(x_val_shop_item,x_val_item,on=['item_id'],how='left')
	if(item_cat_count_feat):x_val_shop_item = pd.merge(x_val_shop_item,x_val_itemcat,on=['item_category_id'],how='left')

	#introduce lag features
	for i in xrange(1,lag_length+1):
	    x_val_lag = sales_2014[sales_2014['month']==(11-i)]
	    x_val_lag = x_val_lag[col_to_keep]

	    #agg shop_item 
	    x_val_shop_item_lag = x_val_lag.groupby(groupby_list,as_index=False).agg(agg_dict).rename(columns={'item_cnt_day':'shop_item_cnt_month_lag_'+str(i)})
	    x_val_shop_item_lag.drop(columns=['item_category_id'],inplace=True)
	    #agg shop 
	    x_val_shop_lag = x_val_lag[['shop_id','item_cnt_day']].groupby(['shop_id'],as_index=False).agg(agg_targ).rename(columns={'item_cnt_day':'shop_cnt_month_lag_'+str(i)})
	    #agg item 
	    x_val_item_lag = x_val_lag[['item_id','item_cnt_day']].groupby(['item_id'],as_index=False).agg(agg_targ).rename(columns={'item_cnt_day':'item_cnt_month_lag_'+str(i)})
	    #agg item_cat 
	    if(item_cat_count_feat):x_val_itemcat_lag = x_val_lag[['item_category_id','item_cnt_day']].groupby(['item_category_id'],as_index=False).agg(agg_targ).rename(columns={'item_cnt_day':'item_cat_cnt_month_lag_'+str(i)})

	    #merge
	    x_val_shop_item = pd.merge(x_val_shop_item,x_val_shop_item_lag,on=['shop_id','item_id'],how='left')
	    x_val_shop_item = pd.merge(x_val_shop_item,x_val_shop_lag,on=['shop_id'],how='left')
	    x_val_shop_item = pd.merge(x_val_shop_item,x_val_item_lag,on=['item_id'],how='left')
	    if(item_cat_count_feat):x_val_shop_item = pd.merge(x_val_shop_item,x_val_itemcat_lag,on=['item_category_id'],how='left')
	    
	    #diffs
	    for col in ['shop_item','shop','item','item_cat']:
	        if(not item_cat_count_feat and col=='item_cat'): continue #skip item_cat if this feat is not turned on 

	        if i==1: 
	            if(diff):x_val_shop_item[col+'_cnt_month_diff({}-{})'.format(str(i-1),str(i))] = x_val_shop_item[col+'_cnt_month'] - x_val_shop_item[col+'_cnt_month_lag_'+str(i)]   
	        if i>=2: 
	            if(diff):x_val_shop_item[col+'_cnt_month_diff({}-{})'.format(str(i-1),str(i))] = x_val_shop_item[col+'_cnt_month_lag_'+str(i-1)] - x_val_shop_item[col+'_cnt_month_lag_'+str(i)]   

	        if(diffRel):x_val_shop_item[col+'_cnt_month_({}-{})/{}'.format(str(i-1),str(i),str(i))] = x_val_shop_item[col+'_cnt_month_diff({}-{})'.format(str(i-1),str(i))] / (x_val_shop_item[col+'_cnt_month_lag_'+str(i)]+1e-7)   

	x_val_shop_item.fillna(0,inplace=True)

	#pick (meta)target            
	y_val = x_val_shop_item[target]

	#remove targets(s):
	x_val = x_val_shop_item.drop(columns=col_targets)

	print 'x_val.shape :',x_val.shape
	print 'y_val.shape :',y_val.shape

	return x_val,y_val

def createTestSet(**kwargs):

	sales_2013 = kwargs['sales_2013']
	sales_2014 = kwargs['sales_2014']
	sales_2015 = kwargs['sales_2015']
	test = kwargs['test']
	items = kwargs['items']

	lag_length = kwargs['lag_length']
	diff = kwargs['diff']
	diffRel = kwargs['diffRel']
	item_cat_count_feat = kwargs['item_cat_count_feat']

	print 'lag_length:',lag_length
	print 'diff:',diff
	print 'diffRel:',diffRel
	print 'item_cat_count_feat :',item_cat_count_feat

	col_to_keep = kwargs['col_to_keep']
	groupby_list= kwargs['groupby_list']
	agg_dict= kwargs['agg_dict']
	agg_targ= kwargs['agg_targ']
	col_targets= kwargs['col_targets']

	x_test = test.sort_values(by=groupby_list)

	#add item_category_id.
	x_test = pd.merge(x_test,items[['item_id','item_category_id']],on='item_id',how='left')

	#introduce lag features
	for i in xrange(1,lag_length+1):
	    x_test_lag = sales_2015[sales_2015['month']==(11-i)]
	    x_test_lag = x_test_lag[col_to_keep]

	    #agg shop_item 
	    x_test_shop_item_lag = x_test_lag.groupby(groupby_list,as_index=False).agg(agg_dict).rename(columns={'item_cnt_day':'shop_item_cnt_month_lag_'+str(i)})
	    x_test_shop_item_lag.drop(columns=['item_category_id'],inplace=True)
	    #agg shop 
	    x_test_shop_lag = x_test_lag[['shop_id','item_cnt_day']].groupby(['shop_id'],as_index=False).agg(agg_targ).rename(columns={'item_cnt_day':'shop_cnt_month_lag_'+str(i)})
	    #agg item 
	    x_test_item_lag = x_test_lag[['item_id','item_cnt_day']].groupby(['item_id'],as_index=False).agg(agg_targ).rename(columns={'item_cnt_day':'item_cnt_month_lag_'+str(i)})
	    #agg item_cat 
	    if(item_cat_count_feat):x_test_itemcat_lag = x_test_lag[['item_category_id','item_cnt_day']].groupby(['item_category_id'],as_index=False).agg(agg_targ).rename(columns={'item_cnt_day':'item_cat_cnt_month_lag_'+str(i)})

	    #merge
	    x_test = pd.merge(x_test,x_test_shop_item_lag,on=['shop_id','item_id'],how='left')
	    x_test = pd.merge(x_test,x_test_shop_lag,on=['shop_id'],how='left')
	    x_test = pd.merge(x_test,x_test_item_lag,on=['item_id'],how='left')
	    if(item_cat_count_feat):x_test = pd.merge(x_test,x_test_itemcat_lag,on=['item_category_id'],how='left')
	    
	    #diffs
	    for col in ['shop_item','shop','item','item_cat']:
	        if(not item_cat_count_feat and col=='item_cat'): continue #skip item_cat if this feat is not turned on 
	        if i>=2:  
	            if(diff):x_test[col+'_cnt_month_diff({}-{})'.format(str(i-1),str(i))] = x_test[col+'_cnt_month_lag_'+str(i-1)] - x_test[col+'_cnt_month_lag_'+str(i)]   
	            if(diffRel):x_test[col+'_cnt_month_({}-{})/{}'.format(str(i-1),str(i),str(i))] = x_test[col+'_cnt_month_diff({}-{})'.format(str(i-1),str(i))] / (x_test[col+'_cnt_month_lag_'+str(i)]+1e-7)   


	x_test = x_test.fillna(0)
	x_test.drop(columns=['ID'],inplace=True)

	print 'x_test.shape :',x_test.shape
	#x_test.head()

	return x_test
