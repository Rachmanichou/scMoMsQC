import pandas as pd
import numpy as np
import multiprocessing as mp
import glob
import os
import time

from statsmodels.robust.scale import qn_scale
from sklearn.neighbors import LocalOutlierFactor

from OD_functions import zscore, LOF, dixon_q_test

class Metrics_Matrices:
    
    cores = mp.cpu_count()
    
    def __init__(self, inpath, nb_decimals, func):
        self.inpath = inpath
        self.nb_decimals = nb_decimals
        self.func = func
    
    
    def to_df(self): 
        #By default, LOF, zscore and Dixon Q tests will be used and one func which the metric dataframe as its only argument can be added
        metrics_df = self.clean_df( self.merge_samples() )
        scaled_df = self.scale_data( metrics_df )
        alert_df = self.alert_generator( metrics_df )

        zscore_df = zscore(metrics_df)
        dixon_df = dixon_q_test(scaled_df)
        LOF_df = LOF(metrics_df)

        zscore_df.OD_method = "Z-Score"
        dixon_df.OD_method = "Dixon Q Test"
        LOF_df.OD_method = "LOF"

        if self.func != None:
            func_df = self.func(scaled_df)
            func_df.OD_method = self.func.__name__

            return metrics_df, scaled_df, alert_df, zscore_df, dixon_df, LOF_df, func_df

        return metrics_df, scaled_df, alert_df, zscore_df, dixon_df, LOF_df
    
    
    def compute_performances(self):
        print("computing performances...")
        
        metrics_df = self.clean_df( self.merge_samples() )
        scaled_df = self.scale_data( metrics_df )
        
        performances_dict = {
            "Z-Score F1 score" : self.od_performance(metrics_df,7,9,zscore),
            "Dixon Q Test F1 score" : self.od_performance(scaled_df,7,9,dixon_q_test),
            "LOF F1 score" : self.od_performance(metrics_df,7,9,LOF)
        }

        if self.func != None:
            performances_dict[self.func.__name__+" F1 score"] = self.od_performance(scaled_df,7,9,self.func)
            
        print("performances done!")

        return performances_dict


    def alert_generator(self, df):
        d = {
            "GEX Reads mapped confidently to genome":[0.3,1],
            "GEX Reads mapped antisense to gene":[0,1],
            "Estimated number of cells":[500,10000],
            "GEX Valid UMIs":[0.75,1],
            "GEX Valid barcodes":[0.75,1],
            "ATAC Valid barcodes":[0.75,1],
            "GEX Q30 bases in UMI":[0.75,1],
            "ATAC Mean raw read pairs per cell":[10000,np.inf],
            "GEX Mean raw reads per cell":[20000,np.inf],
            "ATAC Q30 bases in read 1":[0.65,1],
            "ATAC Q30 bases in read 2":[0.65,1],
            "GEX Q30 bases in read 2":[0.65,1]
            }

        thresh_df = pd.DataFrame(data=d,index=["lower","upper"])
        thresh_col = list(thresh_df.columns)
        alert_df = pd.DataFrame(data=False, index=df.index, columns=df.columns)
        alert_df.index = df.index

        for column in df:
            if column in thresh_col:
                alert_df[column] = df[column].apply(lambda x: x >= thresh_df[column].loc["upper"] or x <= thresh_df[column].loc["lower"])
                thresh_col.remove(column)
            else:
                alert_df[column] = False

        return alert_df
    
    
    def clean_df(self, df):
        #Match index and ID & Drop off unwanted columns for matrix
        df["Sample ID"] = df["Sample ID"].map(lambda sample: sample.replace("Sample_",""))
        df.index = df["Sample ID"] #therefore loc and at will be based on the ID
        df = df.drop(columns=["Sample ID","Genome","Pipeline version"]).dropna(axis=0)
        #Round to certain number of decimals
        return df.apply(lambda x: round(x, self.nb_decimals))
    
    
    def scale_data(self, df):
        scaled_df = pd.DataFrame(data=0, index=df.index, columns=df.columns)
        scaled_df.index = df.index
        for column in df:
            center = df[column].median()
            scale = qn_scale(df[column])
            scaled_df[column] = df[column].apply(lambda x: (x-center) ) if scale == 0 else df[column].apply(lambda x: (x-center)/scale )
        return scaled_df
    
    
    def merge_samples(self):
            try:
                summaries = glob.glob(os.path.join(self.inpath,'**/outs/summary.csv'), recursive = True)
                df = pd.concat((pd.read_csv(p) for p in summaries), ignore_index=True)
                return df
            except ValueError:
                print("File not found")
                return
    

    #Apply the Monte Carlo method to compute the score of each of the OD methods. The number of iterations is determined as recommended by Oberle when the variance of the results distribution is known
    def od_performance(self, df, scalar_factor, loc_factor, func):
        print(func.__name__)
        rnd_scale, rnd_loc = np.random.randint(low=1, high=scalar_factor, size = self.cores), np.random.randint(low=1, high=loc_factor, size = self.cores)
        start = time.time()
        with mp.Pool(self.cores) as pool:
            performances = np.array( pool.starmap( self.compute_f1, [(df, i, j, func) for i,j in zip(rnd_scale,rnd_loc)] ) )
        time_for_cores = time.time() - start
        
        #Making the assumption that execution time grows linearly: to achieve n iterations, t = time for x cores * n/x cores
        #This function is set not to exceed 10 seconds
        max_n = int(10*self.cores/(time_for_cores))
        
        #n_iterations = (z*std/fraction of error*mean estimate)**2
        if max_n > 30-self.cores:
            rnd_scale, rnd_loc = np.random.randint(low=1, high=scalar_factor, size = 30-self.cores), np.random.randint(low=1, high=loc_factor, size = 30-self.cores)
            with mp.Pool(self.cores) as pool:
                performances = np.append( performances, pool.starmap( self.compute_f1, [(df, i, j, func) for i,j in zip(rnd_scale, rnd_loc)] ) )
            std, average = np.std(performances), np.mean(performances)
            n_iterations = int( (1.96*std) / (0.02*average) )**2 - 30 if average != 0 else 1
            n_iterations = min(n_iterations, max_n) #Limit the execution time to ~10 seconds
        else:
            n_iterations = max_n
            
        rnd_scale, rnd_loc = np.random.randint(low=1, high=scalar_factor, size = n_iterations), np.random.randint(low=1, high=loc_factor, size = n_iterations)

        with mp.Pool(self.cores) as pool:
            performances = np.append( performances, pool.starmap( self.compute_f1, [(df, i, j, func) for i,j in zip(rnd_scale, rnd_loc)] ) )

        return np.mean(performances)
    
    
    def compute_f1(self, df, scalar_factor, loc_factor, func):
        df, out_df = self.toy_data(df, scalar_factor, loc_factor)
        diff = out_df.compare(func(df), align_axis=0)
        
        false_neg = 0 if diff.shape[0] == 0 else diff.xs('self', level=1, axis=0).to_numpy(na_value=False).sum()
        false_pos = 0 if diff.shape[0] == 0 else diff.xs('other', level=1, axis=0).to_numpy(na_value=False).sum()
        true_pos = out_df.to_numpy(na_value=False).sum()-false_neg

        return 2*true_pos/(2*true_pos + false_pos + false_neg)
    
    
    def toy_data(self, df, scalar_factor, loc_factor): #Create a toy dataset based on the input data to test the performances of the OD methods
        toy_df = df.reset_index(drop=True)
        toy_df_outliers = pd.DataFrame(data=False, columns=toy_df.columns, index=toy_df.index)
        #Select random QC Metric to introduce a random number (< half of the data) of random location and scalar errors as in Dixon (1953)
        outlier_cols = np.random.choice(toy_df.columns, size=np.random.randint(low=1, high=toy_df.shape[1]), replace=False)

        for col in toy_df.columns:
            #Make a robust estimation of a normal distribution underlying the input data
            center_estim, std_estim = df[col].median(), qn_scale(df[col])
            toy_df[col] = np.random.normal(loc=center_estim, scale=std_estim, size=toy_df.shape[0])

            if col in outlier_cols:
                #Generate points from an outlier distribution and keep only the ones beyond the extrema. These will be labeled outliers.
                loc_factor *= np.random.choice([-1,1])
                outlying = np.random.normal(loc=center_estim+loc_factor, scale=std_estim*scalar_factor, size=int(toy_df.shape[0]/2))
                outlying = outlying[(outlying >= toy_df[col].max()) | (outlying <= toy_df[col].min())]
                out_indx = np.random.choice(toy_df.index, size=outlying.size)
                toy_df.loc[out_indx, col] = outlying
                toy_df_outliers.loc[out_indx, col] = True

        return toy_df, toy_df_outliers          
        
    
