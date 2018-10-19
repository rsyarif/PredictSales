import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold

class Sets:

	def __init__(self,**kwargs):

		# self.sales_2013 = kwargs['sales_2013']
		# self.sales_2014 = kwargs['sales_2014']
		# self.sales_2015 = kwargs['sales_2015']

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
		self.diffRel = kwargs['diffRel']
		self.target = kwargs['target']

		self.col_to_keep = kwargs['col_to_keep']
		self.groupby_list= kwargs['groupby_list']
		self.agg_dict= kwargs['agg_dict']
		self.agg_targ= kwargs['agg_targ']
		self.col_targets= kwargs['col_targets']

		self.meanEncode = kwargs['meanEncode']
		self.meanEncodeCol=kwargs['meanEncodeCol']

		self.targEnc_to_Reg=kwargs['targEnc_to_Reg']
		self.NaN_targEnc=kwargs['NaN_targEnc']

		print
		print 'lag_length:',self.lag_length
		print 'diff:',self.diff
		print 'diffRel:',self.diffRel
		print 'target:',self.target
		print '\ntarget encoding:',self.agg_targ
		print

	def convertDatetime(self):
		# Format 'date' to 
		if(self.verbose):print "\nFormat 'date' to 'datetime' in sales_train"
		self.sales_train['date'] = pd.to_datetime(self.sales_train['date'],format='%d.%m.%Y')


	def addItemCategoryId(self):
		# Add item_category_id to sales_train
		if(self.verbose):print "\nAdd new column: 'item_category_id' to sales_train"
		self.sales_train = pd.merge(self.sales_train,self.items[['item_id','item_category_id']],on='item_id',how='left').sort_values(by='item_id')


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

		df_bins = pd.IntervalIndex.from_tuples(bins)
		s_binned = pd.cut(self.sales_train['item_price'],bins=df_bins,labels=labels)

		if(self.verbose):print '\nAdding new column: price_range to sales_train.'
		self.sales_train['price_range'] = s_binned


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


	def aggAddNewColumns(self,dataset):
		#target features:

		#agg shop_item 
		df = dataset.groupby(self.groupby_list,as_index=False).agg(self.agg_dict).rename(columns={'item_cnt_day':'shop_item_cnt_month'})
		#agg shop (mean encoding)
		if(self.meanEncode):shop = dataset[['shop_id','item_cnt_day']].groupby(['shop_id'],as_index=False).agg(self.agg_targ).rename(columns={'item_cnt_day':'shop_cnt_month'})
		#agg item (mean encoding)
		if(self.meanEncode):item = dataset[['item_id','item_cnt_day']].groupby(['item_id'],as_index=False).agg(self.agg_targ).rename(columns={'item_cnt_day':'item_cnt_month'})
		#agg item_cat (mean encoding)
		if(self.meanEncode):itemcat = dataset[['item_category_id','item_cnt_day']].groupby(['item_category_id'],as_index=False).agg(self.agg_targ).rename(columns={'item_cnt_day':'item_cat_cnt_month'})

		#add new columns: shop_item_id
		df['shop_item_id']=df['shop_id'].astype('string')+'_'+df['item_id'].astype('string')

		#merge
		if(self.meanEncode):df = pd.merge(df,shop,on=['shop_id'],how='left')
		if(self.meanEncode):df = pd.merge(df,item,on=['item_id'],how='left')
		if(self.meanEncode):df = pd.merge(df,itemcat,on=['item_category_id'],how='left')

		return df


	def aggLagColumns(self,df,df_lag,i):

		    #agg shop_item 
		    shop_item_lag = df_lag.groupby(self.groupby_list,as_index=False).agg(self.agg_dict).rename(columns={'item_cnt_day':'shop_item_cnt_month_lag_'+str(i)})
		    shop_item_lag.drop(columns=['item_category_id'],inplace=True)

		    df = pd.merge(df,shop_item_lag,on=['shop_id','item_id'],how='left')

			#agg and merge
		    if(self.meanEncode):
			    for col in self.meanEncodeCol:
				    newcol = col
				    if(col == 'item_cat'): col = 'item_category'
				    lag = df_lag[[col+'_id','item_cnt_day']].groupby([col+'_id'],as_index=False).agg(self.agg_targ).rename(columns={'item_cnt_day':newcol+'_cnt_month_lag_'+str(i)})
				    df  = pd.merge(df,lag,on=[col+'_id'],how='left')


		    return df


	def addLagFeatures(self,df,year):

		#introduce lag features 10 months behind.
		for i in xrange(1,self.lag_length+1):
		    df_lag = self.sales_train[ (self.sales_train['year']==year) & ( self.sales_train['month']==(11-i) )]
		    df_lag = df_lag[self.col_to_keep]

		    df = self.aggLagColumns(df,df_lag,i)
		    
		    #self.diffs
		    for col in ['shop_item']+self.meanEncodeCol:#,'shop','item','item_cat']:

		        if i==1:
		        	if(year==2015):continue #2015 dont have lag_0 features
		        	if(self.diff):
		        		df[col+'_cnt_month_diff({}-{})'.format(str(i-1),str(i))] = df[col+'_cnt_month'] - df[col+'_cnt_month_lag_'+str(i)]
		        	if(self.diffRel):
		        		df[col+'_cnt_month_({}-{})/{}'.format(str(i-1),str(i),str(i))] = df[col+'_cnt_month_diff({}-{})'.format(str(i-1),str(i))] / (df[col+'_cnt_month_lag_'+str(i)]+1e-7)   
		        if i>=2:
		            if(self.diff):
		            	df[col+'_cnt_month_diff({}-{})'.format(str(i-1),str(i))] = df[col+'_cnt_month_lag_'+str(i-1)] - df[col+'_cnt_month_lag_'+str(i)]   
		        	if(self.diffRel):df[col+'_cnt_month_({}-{})/{}'.format(str(i-1),str(i),str(i))] = df[col+'_cnt_month_diff({}-{})'.format(str(i-1),str(i))] / (df[col+'_cnt_month_lag_'+str(i)]+1e-7)   

		return df


	def createTrainSet(self):

		x_train = self.sales_2013[self.sales_2013['month']==11]
		x_train = x_train[self.col_to_keep]

		x_train_shop_item = self.aggAddNewColumns(x_train)
		x_train_shop_item = self.addLagFeatures(x_train_shop_item,2013)

		x_train_shop_item.fillna(0,inplace=True)

		#pick (meta)self.target            
		y_train = x_train_shop_item[self.target]

		#remove self.targets(s):
		#x_train = x_train_shop_item.drop(columns=self.col_targets)

		x_train = x_train_shop_item

		if(self.verbose):print 'x_train.shape :',x_train.shape
		if(self.verbose):print 'y_train.shape :',y_train.shape

		return x_train, y_train


	def createValSet(self):

		x_val = self.sales_2014[self.sales_2014['month']==11]
		x_val = x_val[self.col_to_keep]

		x_val_shop_item = self.aggAddNewColumns(x_val)
		x_val_shop_item = self.addLagFeatures(x_val_shop_item,2014)
		x_val_shop_item.fillna(0,inplace=True)

		#pick (meta)self.target            
		y_val = x_val_shop_item[self.target]

		#remove self.targets(s):
		#x_val = x_val_shop_item.drop(columns=self.col_targets)

		x_val = x_val_shop_item

		if(self.verbose):print 'x_val.shape :',x_val.shape
		if(self.verbose):print 'y_val.shape :',y_val.shape

		return x_val,y_val


	def createTestSet(self):

		x_test = self.test.sort_values(by=self.groupby_list)

		#add item_category_id.
		x_test = pd.merge(x_test,self.items[['item_id','item_category_id']],on='item_id',how='left')
		#add new columns: shop_item_id
		x_test['shop_item_id']=x_test['shop_id'].astype('string')+'_'+x_test['item_id'].astype('string')

		x_test = self.addLagFeatures(x_test,2015)
		x_test = x_test.fillna(0)
		x_test.drop(columns=['ID'],inplace=True)

		if(self.verbose):print 'x_test.shape :',x_test.shape
		#x_test.head()

		return x_test


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


	def addLagFeatures_v2(self,df,dateblock):

		#introduce lag features.
		for i in xrange(1,self.lag_length+1):
		    df_lag = self.sales_train[ ( self.sales_train['date_block_num']==(dateblock-i) )]
		    df_lag = df_lag[self.col_to_keep]

		    df = self.aggLagColumns(df,df_lag,i)
		    
		    #self.diffs
		    for col in ['shop_item']+self.meanEncodeCol:#,'shop','item','item_cat']:

		        if i==1:
		        	if(dateblock==34):continue #2015 dont have lag_0 features
		        	if(self.diff):
		        		df[col+'_cnt_month_diff({}-{})'.format(str(i-1),str(i))] = df[col+'_cnt_month'] - df[col+'_cnt_month_lag_'+str(i)]
		        	if(self.diffRel):
		        		df[col+'_cnt_month_({}-{})/{}'.format(str(i-1),str(i),str(i))] = df[col+'_cnt_month_diff({}-{})'.format(str(i-1),str(i))] / (df[col+'_cnt_month_lag_'+str(i)]+1e-7)   
		        if i>=2:
		            if(self.diff):
		            	df[col+'_cnt_month_diff({}-{})'.format(str(i-1),str(i))] = df[col+'_cnt_month_lag_'+str(i-1)] - df[col+'_cnt_month_lag_'+str(i)]   
		        	if(self.diffRel):df[col+'_cnt_month_({}-{})/{}'.format(str(i-1),str(i),str(i))] = df[col+'_cnt_month_diff({}-{})'.format(str(i-1),str(i))] / (df[col+'_cnt_month_lag_'+str(i)]+1e-7)   

		return df


	def addIsItemNew(self,df,dateblock):
		
		if(self.verbose):print 'adding isItemNew feature ...'
		df_now = df
		df_prev = self.sales_train[self.sales_train['date_block_num']==dateblock-1].groupby(['shop_id','item_id'],as_index=False).agg({'item_cnt_day':'sum'})

		df_now['isItemNew'] = df_now['item_id'].apply(lambda x: False if x in df_prev['item_id'].values else True)
		#df_now['IsItemNew'] = ~df_now['item_id'].isin(df_prev['item_id'].values) # '~' assigns a NOT. This is just for notwes.
		
		df = df_now

		return df

	def createDateblockSet(self,dateblocks):

		x_ = pd.DataFrame([])
		y_ = pd.Series([])

		for dateblock in dateblocks:		
			if(self.verbose): print 'processing dateblock:',dateblock

			x = self.sales_train[ (self.sales_train['date_block_num']==dateblock) ]
			x = x[self.col_to_keep]

			x_shop_item = self.aggAddNewColumns(x)
			x_shop_item = self.addIsItemNew(x_shop_item,dateblock)
			x_shop_item = self.addLagFeatures_v2(x_shop_item,dateblock)

			x_shop_item.fillna(0,inplace=True)


			if(self.verbose): print ' '*3,'x_shop_item.shape', x_shop_item.shape

			x_ = x_.append(x_shop_item,sort=False)

		#pick (meta)self.target            
		y_ = x_[self.target]

		#remove self.targets(s):
		#x_train_ = x_train_.drop(columns=self.col_targets)

		if(self.verbose):print 'x.shape :',x_.shape
		if(self.verbose):print 'y.shape :',y_.shape

		return x_, y_


	# def checkOutliers(self):

	# def addDiffLagColums(self)

	# def getShapes


