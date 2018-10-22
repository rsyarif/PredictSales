import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold

class Sets:

	def __init__(self,**kwargs):

		self.verbose = kwargs['verbose']

		path = '/Users/rizki/Dropbox/Coursera/AML_HowToKaggle/FinalProject/PredictSales/all/'

		self.items = pd.read_csv(path+'items.csv')
		self.item_categories = pd.read_csv(path+'item_categories.csv')
		self.shops = pd.read_csv(path+'shops.csv')
		self.sales_train = pd.read_csv(path+'sales_train.csv')
		self.test = pd.read_csv(path+'test.csv')

		if(self.verbose):print '\nsales_train shape:',self.sales_train.shape

		self.data = {
				'items':self.items,
				'item_categories':self.item_categories,
				'shops':self.shops,
				'sales_train':self.sales_train,
				'test':self.test,
				}

		#options
		self.lag_length = kwargs['lag_length']
		self.diff = kwargs['diff']

		print
		print 'lag_length:',self.lag_length
		print 'diff:',self.diff
		print

	def convertDatetime(self):
		# Format 'date' to 
		if(self.verbose):print "\nFormat 'date' to 'datetime' in sales_train"
		self.sales_train['date'] = pd.to_datetime(self.sales_train['date'],format='%d.%m.%Y')


	def addItemCategoryId(self):
		# Add item_category_id to sales_train
		if(self.verbose):print "\nAdd new column: 'item_category_id' to sales_train"
		#self.sales_train = pd.merge(self.sales_train,self.items[['item_id','item_category_id']],on='item_id',how='left').sort_values(by='item_id')
		self.sales_train = pd.merge(self.sales_train,self.items[['item_id','item_category_id']],on='item_id',how='left')


	def translateItemCategoryId(self):
		#courtesy of https://www.kaggle.com/alexeyb/coursera-winning-kaggle-competitions

		l = list(self.item_categories.item_category_name)
		l_cat = l

		for ind in range(0,8):
		    l_cat[ind] = 'Access'

		for ind in range(10,18):
		    l_cat[ind] = 'Consoles'

		for ind in range(18,25):
		    l_cat[ind] = 'Consoles Games'

		for ind in range(26,28):
		    l_cat[ind] = 'phone games'

		for ind in range(28,32):
		    l_cat[ind] = 'CD games'

		for ind in range(32,37):
		    l_cat[ind] = 'Card'

		for ind in range(37,43):
		    l_cat[ind] = 'Movie'

		for ind in range(43,55):
		    l_cat[ind] = 'Books'

		for ind in range(55,61):
		    l_cat[ind] = 'Music'

		for ind in range(61,73):
		    l_cat[ind] = 'Gifts'

		for ind in range(73,79):
		    l_cat[ind] = 'Soft'


		self.item_categories['eng_cat_id'] = l_cat

		return self.item_categories


	def addTranslatedItemCategoryId(self):
		# Add item_category_id to sales_train
		if(self.verbose):print "\nAdd new column: 'english translated item_category_id' to sales_train"
		#self.sales_train = pd.merge(self.sales_train,self.item_categories[['item_category_id','eng_cat_id']],on='item_category_id',how='left').sort_values(by='item_id')
		self.sales_train = pd.merge(self.sales_train,self.item_categories[['item_category_id','eng_cat_id']],on='item_category_id',how='left')


	def addYMcolumn(self):
		#add new columns: years, month, Y_M
		if(self.verbose):print "\nAdd new column: years, month, Y_M to sales_train"
		self.sales_train['year'] = self.sales_train['date'].dt.year
		self.sales_train['month'] = self.sales_train['date'].dt.month
		self.sales_train_year = self.sales_train['date'].dt.year.astype('string')
		self.sales_train_month = self.sales_train['date'].dt.month.astype('string')
		self.sales_train['Y_M'] = self.sales_train_year+'_'+self.sales_train_month


	def getBins(self,bin_edges): #for binPrice method.
	    bins =[]
	    labels=[]
	    for (i,val) in enumerate(bin_edges):
	        if i < len(bin_edges)-1: 
	            bins.append((bin_edges[i],bin_edges[i+1]))
	            labels.append('{}to{}'.format(bin_edges[i],bin_edges[i+1]))
	    return bins,labels


	def binPrice(self,bin_edges):
		#count in each bins

		if(self.verbose):print '\nCounting based on the defined bins:\n'
		for i,ibin in enumerate(bin_edges):    
		    if i==len(bin_edges)-1: 
		        if(self.verbose):print '{}-:'.format(bin_edges[i],bin_edges[i]),
		        if(self.verbose):print self.sales_train[(self.sales_train.item_price>=bin_edges[i])].shape[0]
		        continue
		    else:        
		        if(self.verbose):print '{}-{} :'.format(bin_edges[i],bin_edges[i+1]),
		        if(self.verbose):print self.sales_train[(self.sales_train.item_price>=bin_edges[i])&(self.sales_train.item_price<bin_edges[i+1])].shape[0]

		bins,labels = self.getBins(bin_edges)

		if(self.verbose): print 'bins:',bins            
		if(self.verbose): print 'labels:',labels           

		df_bins = pd.IntervalIndex.from_tuples(bins)
		s_binned = pd.cut(self.sales_train['item_price'],bins=df_bins,labels=labels)

		if(self.verbose):
			print "\ncheck for out of bound / NaN binnings (index, item_id, price):"
			print 'Index:',s_binned[s_binned.isna()].index.values, 
			print 'item_id:',self.sales_train[s_binned.isna()]['item_id'].values, 
			print 'item_price:',self.sales_train[s_binned.isna()]['item_price'].values

		###MANUAL FIX for ID, Item_Id 2973, probably was a mistake filling the database:
		self.sales_train.loc[s_binned.isna(),'item_price'] = self.sales_train['item_price'].median() 
		if(self.verbose):print 'Fixing missing price range for index:',s_binned[s_binned.isna()].index.values

		### rebin again due to MANUAL FIX.
		s_binned = pd.cut(self.sales_train['item_price'],bins=df_bins,labels=labels)

		if(self.verbose):print '\nAdding new column: price_range to sales_train.'
		self.sales_train['price_range'] = s_binned


	def binPrice_v2(self,bin_edges,df,date_block_num):
		#count in each bins

		if(self.verbose):print '\nCounting based on the defined bins:\n'
		for i,ibin in enumerate(bin_edges):    
		    if i==len(bin_edges)-1: 
		        if(self.verbose):print '{}-:'.format(bin_edges[i],bin_edges[i]),
		        if(self.verbose):print df[(df[date_block_num]>=bin_edges[i])].shape[0]
		        continue
		    else:        
		        if(self.verbose):print '{}-{} :'.format(bin_edges[i],bin_edges[i+1]),
		        if(self.verbose):print df[(df[date_block_num]>=bin_edges[i])&(df[date_block_num]<bin_edges[i+1])].shape[0]

		bins,labels = self.getBins(bin_edges)

		if(self.verbose): print 'bins:',bins            
		if(self.verbose): print 'labels:',labels           

		df_bins = pd.IntervalIndex.from_tuples(bins)
		s_binned = pd.cut(df[date_block_num],bins=df_bins,labels=labels,include_lowest=True)

		if(self.verbose):
			print "\ncheck for out of bound / NaN binnings (index, item_id, price):"
			print 'Index:',s_binned[s_binned.isna()].index.values, 'total:',len(s_binned[s_binned.isna()].index.values) 
			print 'item_price:',df[s_binned.isna()][date_block_num].values

		return s_binned


	def splitDataByYear(self):
		#split by year

		if(self.verbose):print '\nSplitting sales_train to sales_train_2013, sales_train_2014, and sales_train_2015.'

		self.sales_2013 = self.sales_train[self.sales_train['year']==2013]
		self.sales_2014 = self.sales_train[self.sales_train['year']==2014]
		self.sales_2015 = self.sales_train[self.sales_train['year']==2015]

		data = {
				'sales_2013':self.sales_2013,
				'sales_2014':self.sales_2014,
				'sales_2015':self.sales_2015
				}

		self.data.update(data)


	def getData(self):
		#updates to latest data
		if(self.verbose):print '\nRetrieving latest (preprocessed) data'

		self.data.update({
				'items':self.items,
				'item_categories':self.item_categories,
				'shops':self.shops,
				'sales_train':self.sales_train,
				'test':self.test,
				})
		if(self.verbose):print '\nsales_train shape:',self.sales_train.shape

		return self.data


	def checkDuplicates(self):

		obj = ['sales_train','test']

		dup_ids = []

		for i,name in enumerate(obj):
			if(self.verbose):print '\nChecking for duplicates in',name
			dup = self.data[name][self.data[name].duplicated()].index.values
			dup_ids.append(dup) 
			if len(dup)>0:
				if(self.verbose):print '\nFound {} duplicates in {} : {}'.format(len(dup),name,dup)
				self.data[name].drop_duplicates(keep='first',inplace=True)
				if(self.verbose):print 'Kept first, removed duplicates'
			else:
				if(self.verbose):print 'Found no duplicates in {}'.format(name)

		return dup_ids


	def clipSalesCount(self,y_train,y_val,lowerClip,upperClip):

		print '\nClipping train and val targets [{}-{}]\n'.format(lowerClip,upperClip)
		y_train_clip = np.clip(y_train,lowerClip,upperClip)
		y_val_clip = np.clip(y_val,lowerClip,upperClip)
		#print 'Sum y_train before clip [{}-{}]:'.format(lowerClip,upperClip),np.sum(y_train)
		#print 'Sum y_val before clip[{}-{}]:'.format(lowerClip,upperClip), np.sum(y_val)
		y_train=y_train_clip
		y_val=y_val_clip
		#print 'Sum y_train after clip[{}-{}]:'.format(lowerClip,upperClip),np.sum(y_train)
		#print 'Sum y_val after clip[{}-{}]:'.format(lowerClip,upperClip), np.sum(y_val)

		return y_train,y_val


	def addPriceRange(self,x_train,x_val,x_test):
		# # Adding price category to train,val, test

		print 'train.shape:',self.sales_train.shape

		#Aggregate train by 'item_price' and take __minimum__ of price range category
		train_agg = self.sales_train.groupby(['item_id'], as_index=False).agg({'price_range':'min'})
		train_agg[train_agg['price_range'].isna()]

		#We're gonna do a hack. Change format to string, fill missing value, then change to 'category'
		train_agg['price_range']=train_agg['price_range'].astype('string')
		train_agg[train_agg['price_range']=='nan']
		#self.sales_train[self.sales_train['item_id']==2973]['price_range'].unique()

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
	
		return x_train,x_val,x_test

	
	def mapTargetEnc(self,x_train,y_train,x_val,x_test,Regularize):

		# # Target encode with KFold reg

		#add target back to x_train
		# df = pd.merge(x_train,y_train.to_frame(),left_index=True,right_index=True,how='left')
		# if(self.verbose):print 'x_train.shape',x_train.shape
		# if(self.verbose):print 'y_train.shape',y_train.shape
		# if(self.verbose):print 'df.shape',df.shape
		df = x_train
		#introduce price_range target encoding: price_range_cnt_month
		if('price_range' in df.columns.values):
			df_temp=df.groupby('price_range',as_index=False).agg({'shop_item_cnt_month':'sum'}).rename(columns={'shop_item_cnt_month':'price_range_cnt_month'})
			df = pd.merge(df,df_temp,on='price_range',how='left')

		Reg=''
		if(Regularize):
			if(self.targEnc_to_Reg.items()):print 'Regularizing target encoding!'
		  	Reg='_kFold' #this determines the columns to be mapped to val and test. 
		  	kf = KFold(5,shuffle=True,random_state=1234)
		  	for key,value in self.targEnc_to_Reg.items():
				#initialize
				df[value+'_cnt_month_kFold'] = df[value+'_cnt_month']

				#let's use median of the mean (per feat) for global stat for replacing NaN. can experiment later using min, mean, etc.
				replaceNaN = df.groupby(key)[value+'_cnt_month'].mean().median()
				self.NaN_targEnc.update({value:replaceNaN})
				for tr_ind,val_ind in kf.split(df):
					df_tr, df_val = df.iloc[tr_ind],df.iloc[val_ind]
					feat_target_sum = df_tr.groupby(key)['shop_item_cnt_month'].sum()
					df_val[value+'_cnt_month_kFold'] = df_val[key].map(feat_target_sum)  
					df_val[value+'_cnt_month_kFold'].fillna(replaceNaN, inplace=True)
					df.at[val_ind,value+'_cnt_month_kFold'] = df_val[value+'_cnt_month_kFold'] 
		x_train = df

		# map x_train targ_enc_kFol to x_val
		df = x_val
		for key,value in self.targEnc_to_Reg.items():
		    print 'x_val: adding target encoding:',value+'_cnt_month'+Reg
		    df_temp = x_train.groupby(key)[value+'_cnt_month'+Reg].mean()
		    df_temp = df[key].map(df_temp)
		    df[value+'_cnt_month'+Reg] = df_temp
		    df[value+'_cnt_month'+Reg].fillna(self.NaN_targEnc[value], inplace=True)   
		x_val = df

		# map x_train targ_enc_kFol to x_test
		df = x_test
		for key,value in self.targEnc_to_Reg.items():
		    print 'x_test: adding target encoding:',value
		    df_temp = x_train.groupby(key)[value+'_cnt_month'+Reg].mean()
		    df_temp = df[key].map(df_temp)
		    df[value+'_cnt_month'+Reg] = df_temp
		    df[value+'_cnt_month'+Reg].fillna(self.NaN_targEnc[value], inplace=True)    
		x_test = df

		return x_train,x_val,x_test


	# def addIsItemNew(self,df,dateblock): #depracated
		
	# 	if(self.verbose):print 'adding isItemNew feature ...'
	# 	df_now = df
	# 	df_prev = self.sales_train[self.sales_train['date_block_num']==dateblock-1].groupby(['shop_id','item_id'],as_index=False).agg({'item_cnt_day':'sum'})

	# 	df_now['isItemNew'] = df_now['item_id'].apply(lambda x: False if x in df_prev['item_id'].values else True)
	# 	#df_now['IsItemNew'] = ~df_now['item_id'].isin(df_prev['item_id'].values) # '~' assigns a NOT. This is just for notwes.
		
	# 	df = df_now

	# 	return df


	def addDiff(self,df,item_diff_df,date_block_num,start_of_lag_cols,lag_length):

		block_to_select = date_block_num

		block_end = block_to_select+1

		begin_col = start_of_lag_cols+block_end-lag_length-1
		end_col = start_of_lag_cols+block_end

		x = pd.concat([df,item_diff_df.loc[:,'{}-{}'.format(date_block_num-lag_length,date_block_num-lag_length-1):'{}-{}'.format(date_block_num-1,date_block_num-2)]],axis=1)
		#rename
		new_col_names_diff = ['item_cnt_diff_lag_{}'.format(i) for i in range(lag_length,0,-1)]
		col_names_diff = ['{}-{}'.format(i,i-1) for i in range(block_to_select-lag_length,block_to_select)]
		d_diff = dict(zip(col_names_diff,new_col_names_diff))
		x  = x.rename(d_diff, axis = 1)

		return x

		print x.shape

		if(self.verbose):print 'x.shape :',x.shape
		if(self.verbose):print 'y.shape :',y.shape


	def add_price_range(self,df,price_range,date_block_num,start_of_lag_cols,lag_length):

		block_to_select = date_block_num

		block_end = block_to_select+1

		begin_col = start_of_lag_cols+block_end-lag_length-1
		end_col = start_of_lag_cols+block_end

		x = pd.concat([df,price_range.loc[:,'price_range_{}'.format(date_block_num-lag_length):'price_range_{}'.format(date_block_num-1)]],axis=1)

		# #rename
		new_col_names = ['price_range_lag_{}'.format(i) for i in range(lag_length,0,-1)]
		col_names = ['price_range_{}'.format(i) for i in range(block_to_select-lag_length,block_to_select)]
		d = dict(zip(col_names,new_col_names))
		x  = x.rename(d, axis = 1)

		return x

		print x.shape

		if(self.verbose):print 'x.shape :',x.shape
		if(self.verbose):print 'y.shape :',y.shape


	def add_isItemNew(self,df,isItemNew,date_block_num,start_of_lag_cols,lag_length):

		block_to_select = date_block_num

		block_end = block_to_select+1

		begin_col = start_of_lag_cols+block_end-lag_length-1
		end_col = start_of_lag_cols+block_end

		x = pd.concat([df,isItemNew.loc[:,'isItemNew_{}'.format(date_block_num-lag_length):'isItemNew_{}'.format(date_block_num)]],axis=1)

		# #rename
		new_col_names = ['isItemNew_lag_{}'.format(i) for i in range(lag_length,-1,-1)]
		col_names = ['isItemNew_{}'.format(i) for i in range(block_to_select-lag_length,block_to_select+1)]
		d = dict(zip(col_names,new_col_names))
		x  = x.rename(d, axis = 1)

		return x

		print x.shape

		if(self.verbose):print 'x.shape :',x.shape
		if(self.verbose):print 'y.shape :',y.shape


	def createDateblockSet(self,df,date_block_num,start_of_lag_cols,lag_length):

		if(self.verbose): print 'processing dateblock:',date_block_num

		block_to_select = date_block_num

		block_end = block_to_select+1

		begin_col = start_of_lag_cols+block_end-lag_length-1
		end_col = start_of_lag_cols+block_end

		x = pd.concat([df.iloc[:,:start_of_lag_cols],df.iloc[:,begin_col:end_col]],axis=1)
		#rename
		new_col_names = ['item_cnt_lag_{}'.format(i) for i in range(lag_length,-1,-1)]
		d = dict(zip(x.columns[start_of_lag_cols:],new_col_names))
		x  = x.rename(d, axis = 1)

		y = df[block_to_select]

		print x.shape

		if(self.verbose):print 'x.shape :',x.shape
		if(self.verbose):print 'y.shape :',y.shape

		return x, y


	def addDiff_forTest(self,df,test,lag_length):

		x = pd.concat([df,test.loc[:,'{}-{}'.format(33-lag_length+1,32-lag_length+1):'{}-{}'.format(33,32)]],axis=1)

		new_col_names_diff = ['item_cnt_diff_lag_{}'.format(i) for i in range(lag_length,0,-1)]
		col_names_diff = ['{}-{}'.format(i,i-1) for i in range(34-lag_length,34)]
		d_diff = dict(zip(col_names_diff,new_col_names_diff))
		x  = x.rename(d_diff, axis = 1)

		return x

	def add_isItemNew_forTest(self,df,test,lag_length):

		x = df

		x['isItemNew_lag_0'] = np.where((df['price_range_lag_1']==0) & (df['price_range_lag_2']==0),'Yes','No')
		x['isItemNew_lag_0'] = x['isItemNew_lag_0'].astype('category')
		x['isItemNew_lag_0'] = x['isItemNew_lag_0'].cat.codes

		# x = pd.concat([df,test.loc[:,'isItemNew_{}'.format(33-lag_length+1):'isItemNew_{}'.format(33)]],axis=1)

		# new_col_names = ['isItemNew_lag_{}'.format(i) for i in range(lag_length,-1,-1)]
		# col_names = ['isItemNew_{}'.format(i) for i in range(34-lag_length-1,34)]
		# d = dict(zip(col_names,new_col_names))
		# x  = x.rename(d, axis = 1)

		return x

	def add_price_range_forTest(self,df,test,lag_length):

		x = pd.concat([df,test.loc[:,'price_range_{}'.format(33-lag_length+1):'price_range_{}'.format(33)]],axis=1)

		new_col_names = ['price_range_lag_{}'.format(i) for i in range(lag_length,0,-1)]
		col_names = ['price_range_{}'.format(i) for i in range(34-lag_length,34)]
		d = dict(zip(col_names,new_col_names))
		x  = x.rename(d, axis = 1)

		return x

	def createFeaturesForTest(self,test,start_of_lag_cols,lag_length):

		#merge test with our pivoted table
		# test = self.test.merge(df, how = "left", on = ["shop_id", "item_id"]).fillna(0.0)

		# Select relevant blocks
		block_to_select = 33 #DO NOT CHANGE

		block_end = block_to_select+1

		begin_col = start_of_lag_cols+block_end-lag_length-1
		end_col = start_of_lag_cols+block_end

		test.drop(columns='ID',inplace=True)
		x = pd.concat([test.iloc[:,:start_of_lag_cols],test.iloc[:,begin_col+1:end_col]],axis=1)

		#Rename cols
		new_col_names = ['item_cnt_lag_{}'.format(i) for i in xrange(lag_length,0,-1)]
		print new_col_names
		d = dict(zip(x.columns[4:],new_col_names))
		x  = x.rename(d, axis = 1)

		print x.shape

		return x


	# def checkOutliers(self):

	# def addDiffLagColums(self)

	# def getShapes


