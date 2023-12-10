from scipy.stats import median_abs_deviation
from collections import Counter
import numpy as np


def IQR_method (df,n,features):
    """
    Takes a dataframe and returns an index list corresponding to the observations 
    containing more than n outliers according to the Tukey IQR method.
    """
    outlier_list = []
    
    for column in features:
                
        # 1st quartile (25%)
        Q1 = np.percentile(df[column], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[column],75)
        
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determining a list of indices of outliers
        outlier_list_column = df[(df[column] < Q1 - outlier_step) | (df[column] > Q3 + outlier_step )].index
        
        # appending the list of outliers 
        outlier_list.extend(outlier_list_column)
        
    # selecting observations containing more than x outliers
    outlier_list = Counter(outlier_list)        
    multiple_outliers = list( k for k, v in outlier_list.items() if v > n )
    
    # Calculate the number of records below and above lower and above bound value respectively
    df1 = df[df[column] < Q1 - outlier_step]
    df2 = df[df[column] > Q3 + outlier_step]
    
    print('Total number of outliers is:', df1.shape[0]+df2.shape[0])
    return  multiple_outliers

    
 

def StDev_method (df,n,features,d_from_sd=3):
    """
    Takes a dataframe df of features and returns an index list corresponding to the observations 
    containing more than n outliers according to the standard deviation method.
    """
    outlier_indices = []
    
    for column in features:
        # calculate the mean and standard deviation of the data frame
        data_mean = df[column].mean()
        data_std = df[column].std()
        
        # calculate the cutoff value
        cut_off = data_std * d_from_sd
        
        # Determining a list of indices of outliers for feature column        
        outlier_list_column = df[(df[column] < data_mean - cut_off) | (df[column] > data_mean + cut_off)].index
        
        # appending the found outlier indices for column to the list of outlier indices 
        outlier_indices.extend(outlier_list_column)
        
    # selecting observations containing more than x outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    # Calculate the number of records below and above lower and above bound value respectively
    df1 = df[df[column] > data_mean + cut_off]
    df2 = df[df[column] < data_mean - cut_off]
    print('Total number of outliers is:', df1.shape[0]+ df2.shape[0])
    
    return multiple_outliers   

# detecting outliers
# Outliers_StDev = StDev_method(df,1,feature_list)

# dropping outliers
# df_out2 = df.drop(Outliers_StDev, axis = 0).reset_index(drop=True)
    
    
    
    
def z_score_method (df,n,features,threshold = 3):
    """
    Takes a dataframe df of features and returns an index list corresponding to the observations 
    containing more than n outliers according to the z-score method.
    """
    outlier_list = []

    for column in features:
        # calculate the mean and standard deviation of the data frame
        data_mean = df[column].mean()
        data_std = df[column].std()

        z_score = abs( (df[column] - data_mean)/data_std )

        # Determining a list of indices of outliers for feature column        
        outlier_list_column =  df[z_score > threshold].index

        # appending the found outlier indices for column to the list of outlier indices 
        outlier_list.extend(outlier_list_column)

    # selecting observations containing more than x outliers
    outlier_list = Counter(outlier_list)        
    multiple_outliers = list( k for k, v in outlier_list.items() if v > n )

    # Calculate the number of outlier records
    df1 = df[z_score > threshold]
    print('Total number of outliers is:', df1.shape[0])

    return multiple_outliers


'''Z-scores can be affected by unusually large or small data values. If there is one extreme value, the z-score corresponding to that point will also be extreme which is why a more robust way to detect outliers is to use a modified z-score.

It has the potential to significantly move the mean away from its actual value. Modified z-score is calculated as:

Modified z-score = 0.6745(xi – x̃) / MAD

where:

xi: A single data value
x̃: The median of the dataset
MAD: The median absolute deviation of the dataset
The median absolute deviation (MAD) is a robust statistic of variability that measures the spread of a dataset. It’s less affected by outliers than other measures of dispersion like standard deviation and variance. If your data is normal, the standard deviation is usually the best choice for assessing spread. However, if your data isn’t normal, the MAD is one statistic you can use instead.

MAD = median(|xi – xm|)

where:

xi: The ith value in the dataset
xm: The median value in the dataset'''


def z_scoremod_method (df,n,features,threshold = 3):
    """
    Takes a dataframe df of features and returns an index list corresponding to the observations 
    containing more than n outliers according to the z-score modified method.
    """
    outlier_list = []
    
    for column in features:
        # calculate the mean and standard deviation of the data frame
        data_mean = df[column].mean()
        data_std = df[column].std()
        MAD = median_abs_deviation
        
        mod_z_score = abs(0.6745*(df[column] - data_mean)/MAD(df[column]) )
                
        # Determining a list of indices of outliers for feature column        
        outlier_list_column =  df[mod_z_score >threshold].index
        
        # appending the found outlier indices for column to the list of outlier indices 
        outlier_list.extend(outlier_list_column)
        
    # selecting observations containing more than x outliers
    outlier_list = Counter(outlier_list)        
    multiple_outliers = list( k for k, v in outlier_list.items() if v > n )
    
    # Calculate the number of outlier records
    df1 = df[mod_z_score >threshold]
    print('Total number of outliers is:', df1.shape[0])
    
    return multiple_outliers



'''Isolation Forest and DBSCAN - Density-Based Spatial Clustering of Applications with Noise are more advanced methods of identifying outliers. They are more generally used as an unsupervised technique for anomaly detection, than for data cleaning'''



'''from sklearn.ensemble import IsolationForest

df = df.drop(['Class'], axis=1)

Number of estimators: n_estimators refers to the number of base estimators or trees in the ensemble (the number of trees that will get built in the forest). This is an optional integer parameter. The default value is 100.

Max samples: max_samples is the number of samples to be drawn to train each base estimator. The default value of max_samples is 'auto' (256): If max_samples is larger than the number of samples provided, all samples will be used for all trees (no sampling).

Contamination: it refers to the expected proportion of outliers in the data set (i.e. the proportion of outliers in the data set). This is used when fitting to define the threshold on the scores of the samples. The default value is 'auto'. If ‘auto’, the threshold value will be determined as in the original paper of Isolation Forest. If float, the contamination should be in the range (0, 0.5].

Max features: All the base estimators are not trained with all the features available in the dataset. It is the number of features to draw from the total features to train each base estimator or tree.The default value of max features is 1.

model=IsolationForest(n_estimators=150, max_samples='auto', contamination=float(0.1), max_features=1.0)
model.fit(df)'''



'''from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# scale data first
X = StandardScaler().fit_transform(df.values)

db = DBSCAN(eps=3.0, min_samples=10).fit(X)
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print('The number of clusters in dataset is:', n_clusters_)
The number of clusters in dataset is: n_clusters_

The number of clusters does not include outliers/noise in the dataset.

Labels are the labels of the clusters. If the label is -1, then the observation is an outlier/noise.

To check value count for each label : pd.Series(labels).value_counts()'''