import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plotYearly(yearlySales,y_test,saveName,saveplots=False,shift=0.0):

    y_test_total_cnt_month = y_test.sum()['item_cnt_month'] - shift
    print 'total sales count from '+saveName+' prediction:',y_test_total_cnt_month
    
    constant_total_cnt_month = 214200 * 0.5 - shift
    print 'total sales count from contant 0.5 prediction:',constant_total_cnt_month
    
    total_item_cnt_2013 = yearlySales[0]
    total_item_cnt_2014 = yearlySales[1]
    total_item_cnt_2015 = yearlySales[2]
    
    #Adding Oct8-2018_2 pred
    total_item_cnt_2015_1 = np.append(total_item_cnt_2015,[y_test_total_cnt_month])
    #Adding constant pred of 0.5 - benchmark
    total_item_cnt_2015_2 = np.append(total_item_cnt_2015,[constant_total_cnt_month])
    #Adding prev month - benchmark
    total_item_cnt_2015_3 = np.append(total_item_cnt_2015,total_item_cnt_2015[-1])
    
    #plot
    x13 = [i+1 for i in xrange(len(total_item_cnt_2013)) ]
    y13 = total_item_cnt_2013
    x14 = [i+1 for i in xrange(len(total_item_cnt_2014)) ]
    y14 = total_item_cnt_2014
    x15_1 = [i+1 for i in xrange(len(total_item_cnt_2015_1)) ]
    y15_1 = total_item_cnt_2015_1
    x15_2 = [i+1 for i in xrange(len(total_item_cnt_2015_2)) ]
    y15_2 = total_item_cnt_2015_2
    x15_3 = [i+1 for i in xrange(len(total_item_cnt_2015_3)) ]
    y15_3 = total_item_cnt_2015_3
    plt.plot(x13,y13,label='2013')
    plt.plot(x14,y14,label='2014')
    plt.plot(x15_1,y15_1,label=saveName+' prediction')
    plt.plot(x15_2,y15_2,label='Constant 0.5 prediction')
    plt.plot(x15_3,y15_3,label='Prev month prediction ')
    plt.ylabel('total item count')
    plt.xlabel('month')
    plt.title('Total sales')
    plt.legend()
    if(saveplots):plt.savefig('yearly_trend_compare_with_pred_'+saveName+'.pdf')
    plt.show() 

def plotYearly_v2(yearlySales,y_test,saveName,saveplots=False,shift=0.0,ShowPlot=True):
    y_test_total_cnt_month = y_test.sum()['item_cnt_month'] - shift
    #print 'total sales count from '+saveName+' prediction:',y_test_total_cnt_month
    
    constant_total_cnt_month = 214200 * 0.5 - shift
    #print 'total sales count from contant 0.5 prediction:',constant_total_cnt_month
    
    total_item_cnt_2013 = yearlySales[0]
    total_item_cnt_2014 = yearlySales[1]
    total_item_cnt_2015 = yearlySales[2]
    
    #Adding Oct8-2018_2 pred
    total_item_cnt_2015_1 = np.append(total_item_cnt_2015,[y_test_total_cnt_month])
    #Adding constant pred of 0.5 - benchmark
    total_item_cnt_2015_2 = np.append(total_item_cnt_2015,[constant_total_cnt_month])
    #Adding prev month - benchmark
    total_item_cnt_2015_3 = np.append(total_item_cnt_2015,total_item_cnt_2015[-1])
    
    #plot
    x13 = [str(i+1)+'_2013' for i in xrange(len(total_item_cnt_2013)) ]
    y13 = total_item_cnt_2013
    x14 = [str(i+1)+'_2014' for i in xrange(len(total_item_cnt_2014)) ]
    y14 = total_item_cnt_2014
    x15 = [str(i+1)+'_2015' for i in xrange(len(total_item_cnt_2015_1)) ]
    y15_1 = total_item_cnt_2015_1
    y15_2 = total_item_cnt_2015_2
    y15_3 = total_item_cnt_2015_3
   
    x = x13+x14+x15
    y_1 = np.concatenate((y13,y14,y15_1))
    y_2 = np.concatenate((y13,y14,y15_2))
    y_3 = np.concatenate((y13,y14,y15_3))

    if(ShowPlot):
        plt.figure(figsize=(6.4*3,4.8))
        plt.plot(x,y_1,label=saveName+' prediction')
        plt.plot(x,y_2,label='Constant 0.5 prediction')
        plt.plot(x,y_3,label='Prev month prediction ')
        plt.ylabel('total item count')
        plt.xlabel('month_year')
        plt.xticks(rotation=45)
        plt.title('Total sales')    
        plt.legend()
        if(saveplots):plt.savefig('yearly_trend_v2_compare_with_pred_'+saveName+'.pdf')
        plt.show() 

    return (x,y_1,y_2,y_3)

def plotResidual(yearlySales,y_test,saveName,saveplots=False,shift=0.0):

    x,y_1,y_2,y_3 = plotYearly_v2(yearlySales,y_test,saveName,saveplots,shift,ShowPlot=False)

    x.remove(x[0])
    y1 = [y_1[i]-y_1[i-1] for i in xrange(1,len(y_1))]
    y2 = [y_2[i]-y_2[i-1] for i in xrange(1,len(y_2))]
    y3 = [y_3[i]-y_3[i-1] for i in xrange(1,len(y_3))]

    plt.figure(figsize=(6.4*3,4.8))
    plt.plot(x,y1,label=saveName+' prediction')
    plt.plot(x,y2,label='Constant 0.5 prediction')
    plt.plot(x,y3,label='Prev month prediction ')
    plt.ylabel('total item count')
    plt.xlabel('month_year')
    plt.xticks(rotation=45)
    plt.title('Total sales residuals')
    plt.legend()
    if(saveplots):plt.savefig('yearly_sales_residual_compare_with_pred_'+saveName+'.pdf')
    plt.show() 



