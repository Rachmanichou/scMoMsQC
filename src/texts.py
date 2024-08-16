od_script = '''function toggleHelp (txtid) { 
                                        const textField = document.getElementById(txtid); 
                                        textField.style.display = textField.style.display === "none" ? "block" : "none"; 
                                        }'''    

intro_text = "This QC Metrics Summary puts together the QC metrics returned by the Cell Ranger pipeline with multiomics, single cell or bulk sequencing data. It helps the user visualize the distribution of the QC metrics, helping identify outliers and batch effects. It also proposes different outlier detection methods."

zscore_text = "The Z-Score outlier detection applyied here is based on rejecting the points that are too far from the median value. Here, we consider outliers all values which lie more than three Qn scale (a robust estimator of the standard deviation) estimators away from the median. It returns the number of outliers for every metric of every sample individually."

dixon_text = "The Dixon Q test rejects extreme values which lie too far from their direct neighbor. The number of extreme values tested depends on population size. It ranges from 1 for a small number of samples (< 8) to 4 (> 13). Like the Z-Score, it is based on the assumption of a normally distributed population."

LOF_text = "The LOF method is a density-based outlier detection algorithm lying on the KNN principle. Grossly, it considers a value to be outlying if there are too few values close to it. It returns whether a sample is an outlier, but gives no information as to what metrics drove this classification the most. This is why every outlier's metric will appear as outlying on the plots."

OD_note_text = "Please take note that the outlier identification is based on the input data. That is, if only low quality samples are given, no outliers may be detected. The alert system serves as an objective quality control."

qc_text = "A table summarizing the QC Metrics. The values written in red are the outlying ones. The cells colored in orange are the ones for which Cell Ranger signals an alert."

violinplot_text = "A violin plot summarizing the QC Metrics. Each metric has its own violin. The data points represent a sample's value for this metric. Hovering the cursor on it will display its name, if Cell Ranger has issued any alerts for the sample, if any of its metrics is an outlier, and finally if the oulier detection methods detect this metrics to be outlying for this sample. Please note that the values are scaled using robust estimators (Qn scale and Median)."

pca_text = "A Principal Component Analysis on the samples retaining the first three principal components. Each point is a sample. You may use this to identify clusters. Please note that the values were scaled using robust estimators before performing the analysis."

clustermap_text = "A cluster map for the samples and the metrics. Darker tiles indicate higher values. On hover, the sample's ID, the metric, if it has any alerts and if it is an outlier by the different OD methods will be displayed. You may use the clustermap to identify clusters."
