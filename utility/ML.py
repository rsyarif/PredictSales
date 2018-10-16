import lightgbm as lgb
from sklearn.metrics import r2_score


class ML:

	def __init__(self,**kwargs):

		self.x_train = kwargs['x_train']
		self.x_val = kwargs['x_val']
		self.x_test = kwargs['x_test']

		self.y_train = kwargs['y_train']
		self.y_val = kwargs['y_val']

	def runBDT_lightgbm(self,lgb_params):

		evals_result={}

		lgb_train = lgb.Dataset(self.x_train, label=self.y_train)
		lgb_test = lgb.Dataset(self.x_val, label=self.y_val)

		#lgb_params = {
		              #  'feature_fraction': .75,
		              #  'metric': 'rmse',
		              #  'nthread':4, 
		              #  'min_data_in_leaf': 2**7, 
		              #  'bagging_fraction': 0.75,#0.75 
		              #  'learning_rate': 0.03, 
		              #  'objective': 'mse', 
		              #  'bagging_seed': 2**7, 
		              #  'num_leaves': 2**7,
		              #  'bagging_freq':1,
		              #  'verbose':1,
		              # }

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

		pred_lgb_tr = model.predict(self.x_train)
		print('Training R-squared for LightGBM is %f' % r2_score(self.y_train, pred_lgb_tr))
		pred_lgb_val = model.predict(self.x_val)
		print('Validation R-squared for LightGBM is %f' % r2_score(self.y_val, pred_lgb_val))

		return model,evals_result


	def predict(self):

		# # Predict with test data
		try:
			pred = model.predict(self.x_test)
		except Exception as e:
			print ('Exception:',e,)
			print ("Probably you haven't trained yet?")

		return pred

