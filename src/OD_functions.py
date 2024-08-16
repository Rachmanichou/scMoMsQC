import pandas as pd
import numpy as np
from statsmodels.robust.scale import qn_scale
from sklearn.neighbors import LocalOutlierFactor


#From Cafaro et al. As if creating a gaussian distribution for each qc metric based on robust estimators (median, qn) and checks which points have very low probablity of happening
def zscore(df, threshold = 3):
    outliers = pd.DataFrame(data=False, index=df.index, columns=df.columns)
    outliers.index = df.index
    for column in df:
       M, q = df[column].median(), qn_scale(df[column])
       outliers[column] = df[column].apply(lambda x: abs(x-M) > threshold*q)
    return outliers

def LOF(df):
    #k parameter is lower bounded by the minimal amount of points in a cluster and upper bounded by the maximum number of neighbors which can be outliers Breunig et al (2000). The maximum proportion of outliers being 50%, we resolve to use half of the number of samples as the k parameter.
    nb_neighbors = int(df.shape[0]/2)
    outliers = pd.DataFrame(data=False, index=df.index, columns=df.columns)
    lof = LocalOutlierFactor(n_neighbors=nb_neighbors).fit_predict(df)
    j = 0
    for ind,row in df.iterrows():
        outliers.loc[ind] = lof[j] == -1
        j+=1
    return outliers

#Dixon Q Test with 95% confidence using Rorabacher's tables (1991) and Dixon's recommandations for test type by sample size
def dixon_q_test(df):

    r10_q95 = [0.97, 0.829, 0.71, 0.625, 0.568]
    r11_q95 = [0.615, 0.57, 0.534]
    r21_q95 = [0.625, 0.592, 0.565]
    r22_q95 = [0.59, 0.568, 0.548, 0.531, 0.516, 0.503, 0.491, 0.48, 0.47, 0.461, 0.452, 0.445, 0.438, 0.432, 0.426, 0.419, 0.414]

    R10_Q95 = {n:q for n,q in zip(range(3,len(r10_q95)+3), r10_q95)}
    R11_Q95 = {n:q for n,q in zip(range(8,len(r11_q95)+8), r11_q95)}
    R21_Q95 = {n:q for n,q in zip(range(11,len(r21_q95)+11), r21_q95)}
    R22_Q95 = {n:q for n,q in zip(range(14,len(r22_q95)+14), r22_q95)}

    outliers = pd.DataFrame(data=False, index=df.index, columns=df.columns)

    #Each time test both upper and lower extrema
    if df.shape[0] < 3:
        return outliers
        
    elif df.shape[0] < 8:
        #r10 = (x1 - x0) / (xn-1 - x0) or (xn-1 - xn-2) / (xn-1 - x0) as here the index is zero based
        outliers = compute_dixon_q_test(df, 0, 0, 0, R10_Q95)
         
    elif df.shape[0] < 11:
        #r11 = (x1 - x0) / (xn-2 - x0) or (xn-1 - xn-2) / (xn-1 - x1)
        outliers = compute_dixon_q_test(df, 0, 0, 1, R11_Q95)

    elif df.shape[0] < 14:
        #r21 = (x2 - x0) / (xn-2 - x0) or (xn-1 - xn-3) / (xn-1 - x1)
        outliers = compute_dixon_q_test(df, 0, 1, 1, R21_Q95)
         
    else:
        #r22 = (x2 - x0) / (xn-3 - x-0) or (xn-1 - xn-3) / (xn-1 - x0)
        outliers = compute_dixon_q_test(df, 0, 1, 2, R22_Q95)
         
    return outliers

#Check that argsort works as expected with negative input

#Parameters are entered as in the left outlier case but both will be computed. The avoidR and L offsets are the number of values which are suspected to be outliers on each end respectively.
def compute_dixon_q_test(df, outlier, avoidL_offset, avoidR_offset, test_dict):
        df = df.reset_index(drop=True)
        
        outliers = pd.DataFrame(data=False, index=df.index, columns=df.columns)
        
        outlierL = df.apply(lambda col:
                                ( ( (col[np.argsort(col)[outlier+1+avoidL_offset]] - col[np.argsort(col)[outlier]]) / (col[np.argsort(col)[df.shape[0]-1-avoidR_offset]] - col[np.argsort(col)[outlier]]) ) > test_dict[df.shape[0]] ) if (col[np.argsort(col)[df.shape[0]-1-avoidR_offset]] - col[np.argsort(col)[outlier]]) != 0 else False
                           )
        outlierR = df.apply(lambda col:
                                ( ( (col[np.argsort(col)[df.shape[0]-1]] - col[np.argsort(col)[df.shape[0]-2-avoidL_offset]]) / (col[np.argsort(col)[df.shape[0]-1]] - col[np.argsort(col)[outlier+avoidR_offset]]) ) > test_dict[df.shape[0]] ) if (col[np.argsort(col)[df.shape[0]-1]] - col[np.argsort(col)[outlier+avoidR_offset]]) != 0 else False
                           ) #Returns the columns which contain an upper outlier
    
        for col in outliers[outlierL.index[outlierL]]:
            for i in range(avoidL_offset+1):
                outliers.loc[np.argsort(df[col])[i], col] = True

        for col in outliers[outlierR.index[outlierR]]:
            for i in range(avoidR_offset+1):
                outliers.loc[np.argsort(df[col])[df.shape[0]-i-1], col] = True

        return outliers
