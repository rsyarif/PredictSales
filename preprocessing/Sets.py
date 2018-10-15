import numpy as np
import pandas as pd
import os

class Sets:

	def __init__(self,**kwargs):

		# self.sales_2013 = kwargs['sales_2013']
		# self.sales_2014 = kwargs['sales_2014']
		# self.sales_2015 = kwargs['sales_2015']

		path = '/Users/rizki/Dropbox/Coursera/AML_HowToKaggle/FinalProject/PredictSales/all/'

		self.items = pd.read_csv(path+'items.csv')
		self.item_categories = pd.read_csv(path+'item_categories.csv')
		self.shops = pd.read_csv(path+'shops.csv')
		self.sales_train = pd.read_csv(path+'sales_train.csv')
		print '\nsales_train shape:',self.sales_train.shape
		self.test = pd.read_csv(path+'test.csv')

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
		self.item_cat_count_feat = kwargs['item_cat_count_feat']
		self.target = kwargs['target']

		print 'lag_length:',self.lag_length
		print 'diff:',self.diff
		print 'diffRel:',self.diffRel
		print 'item_cat_count_feat :',self.item_cat_count_feat
		print 'target:',self.target

		self.col_to_keep = kwargs['col_to_keep']
		self.groupby_list= kwargs['groupby_list']
		self.agg_dict= kwargs['agg_dict']
		self.agg_targ= kwargs['agg_targ']
		self.col_targets= kwargs['col_targets']

		self.meanEncode = kwargs['meanEncode']
		self.meanEncodeCol=kwargs['meanEncodeCol']


	def convertDatetime(self):
		# Format 'date' to 
		print "\nFormat 'date' to 'datetime' in sales_train"
		self.sales_train['date'] = pd.to_datetime(self.sales_train['date'],format='%d.%m.%Y')

	def addItemCategoryId(self):
		# Add item_category_id to sales_train
		print "\nAdd new column: 'item_category_id' to sales_train"
		self.sales_train = pd.merge(self.sales_train,self.items[['item_id','item_category_id']],on='item_id',how='left').sort_values(by='item_id')

	def addYMcolumn(self):
		#add new columns: years, month, Y_M
		print "\nAdd new column: years, month, Y_M to sales_train"
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

		print '\nCounting based on the defined bins:\n'
		for i,ibin in enumerate(bin_edges):    
		    if i==len(bin_edges)-1: 
		        print '{}-:'.format(bin_edges[i],bin_edges[i]),
		        print self.sales_train[(self.sales_train.item_price>=bin_edges[i])].shape[0]
		        continue
		    else:        
		        print '{}-{} :'.format(bin_edges[i],bin_edges[i+1]),
		        print self.sales_train[(self.sales_train.item_price>=bin_edges[i])&(self.sales_train.item_price<bin_edges[i+1])].shape[0]

		bins,labels = self.getBins(bin_edges)            

		df_bins = pd.IntervalIndex.from_tuples(bins)
		s_binned = pd.cut(self.sales_train['item_price'],bins=df_bins,labels=labels)

		print '\nAdding new column: price_range to sales_train.'
		self.sales_train['price_range'] = s_binned


	def splitDataByYear(self):
		#split by year

		print '\nSplitting sales_train to sales_train_2013, sales_train_2014, and sales_train_2015.'

		self.sales_2013 = self.sales_train[self.sales_train['year']==2013]
		self.sales_2014 = self.sales_train[self.sales_train['year']==2014]
		self.sales_2015 = self.sales_train[self.sales_train['year']==2015]

		data = {
				'sales_2013':self.sales_2013,
				'sales_2014':self.sales_2014,
				'sales_2015':self.sales_2015
				}

		self.data.update(data)

	# def getShapes

	def getData(self):
		#updates to latest data
		print '\nRetrieving latest (preprocessed) data'

		self.data.update({
				'items':self.items,
				'item_categories':self.item_categories,
				'shops':self.shops,
				'sales_train':self.sales_train,
				'test':self.test,
				})
		print '\nsales_train shape:',self.sales_train.shape

		return self.data


	def checkDuplicates(self):

		obj = ['sales_train','test']

		dup_ids = []

		for i,name in enumerate(obj):
			print '\nChecking for duplicates in',name
			dup = self.data[name][self.data[name].duplicated()].index.values
			dup_ids.append(dup) 
			if len(dup)>0:
				print '\nFound {} duplicates in {} : {}'.format(len(dup),name,dup)
				self.data[name].drop_duplicates(keep='first',inplace=True)
				print 'Kept first, removed duplicates'
			else:
				print 'Found no duplicates in {}'.format(name)

		return dup_ids

	# def checkOutliers(self):
	# def clipSalesCount(self):

	def aggAddNewColumns(self,dataset):
		#target features:

		#agg shop_item 
		df = dataset.groupby(self.groupby_list,as_index=False).agg(self.agg_dict).rename(columns={'item_cnt_day':'shop_item_cnt_month'})
		#agg shop (mean encoding)
		if(self.meanEncode):shop = dataset[['shop_id','item_cnt_day']].groupby(['shop_id'],as_index=False).agg(self.agg_targ).rename(columns={'item_cnt_day':'shop_cnt_month'})
		#agg item (mean encoding)
		if(self.meanEncode):item = dataset[['item_id','item_cnt_day']].groupby(['item_id'],as_index=False).agg(self.agg_targ).rename(columns={'item_cnt_day':'item_cnt_month'})
		#agg item_cat (mean encoding)
		if(self.item_cat_count_feat and self.meanEncode):itemcat = dataset[['item_category_id','item_cnt_day']].groupby(['item_category_id'],as_index=False).agg(self.agg_targ).rename(columns={'item_cnt_day':'item_cat_cnt_month'})

		#add new columns: shop_item_id
		df['shop_item_id']=df['shop_id'].astype('string')+'_'+df['item_id'].astype('string')

		#merge
		if(self.meanEncode):df = pd.merge(df,shop,on=['shop_id'],how='left')
		if(self.meanEncode):df = pd.merge(df,item,on=['item_id'],how='left')
		if(self.item_cat_count_feat and self.meanEncode):df = pd.merge(df,itemcat,on=['item_category_id'],how='left')

		return df


	def aggLagColumns(self,df,df_lag,i):

		    #agg shop_item 
		    shop_item_lag = df_lag.groupby(self.groupby_list,as_index=False).agg(self.agg_dict).rename(columns={'item_cnt_day':'shop_item_cnt_month_lag_'+str(i)})
		    shop_item_lag.drop(columns=['item_category_id'],inplace=True)
		    #agg shop (mean encoding)
		    if(self.meanEncode):shop_lag = df_lag[['shop_id','item_cnt_day']].groupby(['shop_id'],as_index=False).agg(self.agg_targ).rename(columns={'item_cnt_day':'shop_cnt_month_lag_'+str(i)})
		    #agg item (mean encoding)
		    if(self.meanEncode):item_lag = df_lag[['item_id','item_cnt_day']].groupby(['item_id'],as_index=False).agg(self.agg_targ).rename(columns={'item_cnt_day':'item_cnt_month_lag_'+str(i)})
		    #agg item_cat (mean encoding)
		    if(self.item_cat_count_feat and self.meanEncode):itemcat_lag = df_lag[['item_category_id','item_cnt_day']].groupby(['item_category_id'],as_index=False).agg(self.agg_targ).rename(columns={'item_cnt_day':'item_cat_cnt_month_lag_'+str(i)})

		    #merge
		    df = pd.merge(df,shop_item_lag,on=['shop_id','item_id'],how='left')
		    if(self.meanEncode):df = pd.merge(df,shop_lag,on=['shop_id'],how='left')
		    if(self.meanEncode):df = pd.merge(df,item_lag,on=['item_id'],how='left')
		    if(self.item_cat_count_feat and self.meanEncode):df = pd.merge(df,itemcat_lag,on=['item_category_id'],how='left')

		    return df

	# def addDiffLagColums(self)


	def addLagFeatures(self,df,year):

		#introduce lag features 10 months behind.
		for i in xrange(1,self.lag_length+1):
		    df_lag = self.sales_train[ (self.sales_train['year']==year) & ( self.sales_train['month']==(11-i) )]
		    df_lag = df_lag[self.col_to_keep]

		    df = self.aggLagColumns(df,df_lag,i)
		    
		    #self.diffs
		    for col in ['shop_item']+self.meanEncodeCol:#,'shop','item','item_cat']:
		        if(not self.item_cat_count_feat and col=='item_cat'): continue #skip item_cat if this feat is not turned on 

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
		x_train = x_train_shop_item.drop(columns=self.col_targets)

		print 'x_train.shape :',x_train.shape
		print 'y_train.shape :',y_train.shape

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
		x_val = x_val_shop_item.drop(columns=self.col_targets)

		print 'x_val.shape :',x_val.shape
		print 'y_val.shape :',y_val.shape

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

		print 'x_test.shape :',x_test.shape
		#x_test.head()

		return x_test

