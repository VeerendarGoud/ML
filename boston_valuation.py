import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error

# Gather Data

boston_dataset = load_boston()
data = pd.DataFrame(data=boston_dataset.data,
                    columns=boston_dataset.feature_names)
features = data.drop(['INDUS','AGE'],axis=1)
log_price = np.log(boston_dataset.target)
target = pd.DataFrame(log_price,columns=['PRICE'])

CRIME_IDX = 0
ZN_IDX =1
CHAS_IDX = 2
RM_IDX=4
PTRATIO_IDX=8

property_stats = np.ndarray(shape=(1,11))
property_stats = features.mean().values.reshape(1,11)

#Regression

reg = LinearRegression().fit(features,target)
fitted_vals = reg.predict(features)

MSE = mean_squared_error(target,fitted_vals)
RMSE = np.sqrt(MSE)

def get_log_estimate(nr_rooms,
                    students_per_classroom,
                    next_to_river=False,
                    high_confidence=True):
    
    # Configure property
    
    property_stats[0][RM_IDX] = nr_rooms
    property_stats[0][PTRATIO_IDX]= students_per_classroom
    
    if next_to_river:
        property_stats[0][CHAS_IDX] = 1   
    else:
        property_stats[0][CHAS_IDX] = 0
    
    # Make prediction
    
    log_estimate = reg.predict(property_stats)[0][0]
    
    # Calc Range
    
    if high_confidence:
        upper_bound = log_estimate +2 *RMSE
        lower_bound = log_estimate -2 *RMSE
        interval = 95
    else:
        upper_bound = log_estimate + RMSE
        lower_bound = log_estimate - RMSE
        interval=68
    
    return log_estimate,upper_bound,lower_bound,interval

ZILLO_MEDIAN_PRICE = 583.3

SCALE_FACTOR = ZILLO_MEDIAN_PRICE/np.median(boston_dataset.target)

def get_dollar_estimate(rm,ptratio,chas=False,large_range=True):
    
    """
    Estimate the price of the property.
    
    Parameters
    -----------
    rm:        number of rooms in the property
    ptration:  number of students per teacher in the classroom for 
               the school in the area
    chas:      near to river(0,1)
    large_range: cofidence interval True for 95% or False for 68%
    
    """
    
    if rm <0 or ptratio < 1:
        print('That is unrealistic.Try again.')
        return
    

    log_est,upper,lower,conf = get_log_estimate(rm,
                                                ptratio,
                                                next_to_river=chas,
                                                high_confidence=large_range)
    dollar_est = np.around(np.e**log_est * 1000 * SCALE_FACTOR,-3) 
    dollar_hi = np.around(np.e**upper * 1000 * SCALE_FACTOR,-3)
    dollar_low = np.around(np.e**lower * 1000 * SCALE_FACTOR,-3)
    
    print(f'The estimated property value is {dollar_est}.')
    print(f'At {conf}% confidence the valuation range.')
    print(f'USD {dollar_low} at the lower and to USD {dollar_hi} at the high end.')
    











