import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt


def resumetable(df):
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values    
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values

    return summary


def CalcOutliers(df_num): 

    # calculating mean and std of the array
    data_mean, data_std = np.mean(df_num), np.std(df_num)

    # seting the cut line to both higher and lower values
    # You can change this value
    cut = data_std * 3

    #Calculating the higher and lower cut values
    lower, upper = data_mean - cut, data_mean + cut

    # creating an array of lower, higher and total outlier values 
    outliers_lower = [x for x in df_num if x < lower]
    outliers_higher = [x for x in df_num if x > upper]
    outliers_total = [x for x in df_num if x < lower or x > upper]

    # array without outlier values
    outliers_removed = [x for x in df_num if x > lower and x < upper]
    
    print('Identified lowest outliers: %d' % len(outliers_lower)) 
    print('Identified upper outliers: %d' % len(outliers_higher)) 
    print('Total outlier observations: %d' % len(outliers_total)) 
    print('Non-outlier observations: %d' % len(outliers_removed)) 
    print("Total percentual of Outliers: ", round((len(outliers_total) / len(outliers_removed) )*100, 4)) 
    
    return

def plot_dist(data, col):

    total = len(data)

    plt.figure(figsize=(16,6))
    plt.suptitle(f"{col} Distribution", fontsize = 20)
    plt.subplot(1,2,1)
    p1 = sns.countplot(data= data, x = col, hue = col)
    p1.set_title(f"{col} Dist")
    p1.set_xlabel(f"{col} Categroy Name")
    for p in p1.patches:
        height = p.get_height()
        if height > 0:
            plt.text(p.get_x() + p.get_width()/2.,
                    height + 3,
                    f"{height/total*100:1.2f}%",
                    ha = "center", fontsize = 12)

    plt.subplot(1,2,2)
    p2 = sns.countplot(data= data, x = col, hue= "isFraud")
    p2.set_title(f"{col} Dist by Target")
    p2.set_xlabel(f"{col} Categroy Name")
    for p in p2.patches:
        height = p.get_height()
        if height > 0:
            plt.text(p.get_x() + p.get_width()/2.,
                    height + 3,
                    f"{height/total*100:1.2f}%",
                    ha = "center", fontsize = 12)
            
    tmp = pd.crosstab(data[col], data['isFraud'], normalize='index') * 100
    tmp = tmp.reset_index()
    tmp.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)

    pt = p2.twinx()
    pt = sns.pointplot(data = tmp, x = col, y = "Fraud", color= "black", alpha = 0.5)
    pt.set_ylabel("% of Fraud")