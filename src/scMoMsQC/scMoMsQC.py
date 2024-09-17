from .Metrics_Matrices import Metrics_Matrices
from .Summary_Figures import Summary_Figures
from .HTML_Generator import HTML_Generator


class QC_Summary:
    
    '''
        A class for creating a summary from QC values of multiomics or RNAseq. It writes an HTML file which wraps up
        information about the quality of the sequencing and the presence of outliers, which could be excluded from
        the further analysis.
        
        The input arguments are:
            inpath : the path leading to the qc summary files. It must terminate on the samples containing folder. The
                summaries should be contained in an /outs subfile.
            outpath : the path leading to where the HTML summary will be written.
            nb_decimals : the maximum number of decimals a QC metric can have. This can have great influence on outlier
                detection. Default is set to 3.
            func : a user defined function for outlier detection. It should take as only input a pandas.DataFrame containing
                the metrics and return a pandas.DataFrame of the same dimensions where outlying values are represented by 
                True while inlyiers are represented as False. Default is None.
                
        After instantiation, the create_summary method should be used without any arguments. It will not return anything,
        but will write the summary to the output destination.
    '''
    
    def __init__(self, inpath, outpath, nb_decimals=3, func=None):
        self.inpath = inpath
        self.outpath = outpath
        self.nb_decimals = nb_decimals
        self.func = func
        self.mm = Metrics_Matrices(inpath, nb_decimals, func)

    #Writes the summary output HTML in the destination folder    
    def create_summary(self):
        summary = open(self.outpath+r"/qc_summary.html", 'w') 
        mm_tuple = self.mm.to_df()
        sum_fig = Summary_Figures( *mm_tuple[:3] )
        
        summary.write(
            HTML_Generator(
                sum_fig.create_summary_table(*mm_tuple[3:]),
                sum_fig.create_qc_table(*mm_tuple[3:]),
                sum_fig.create_violinplot(*mm_tuple[3:]),
                sum_fig.draw_pca(*mm_tuple[3:]),
                sum_fig.create_clustermap(*mm_tuple[3:])
            ).display_summary()
        )
        summary.close()
     
    #Returns a dict of estimated F1 scores for each of the OD algorithms   
    def compute_performances(self):
    	return self.mm.compute_performances()
     
        
        
        
        
    
