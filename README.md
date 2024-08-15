[scMoMsQC logo](/image/scMoMsQC.png)

## Introduction

**scMoMsQC** or <ins>s</ins>ingle-<ins>c</ins>ell <ins>M</ins>ulti<ins>o</ins>me <ins>M</ins>ulti<ins>s</ins>ample <ins>Q</ins>uality <ins>C</ins>ontrol is a Python tool for controlling the quality of single cell Multiomics and RNAseq using QC Metrics. It takes multiple multiomics summary files and returns an HTML file with analysis and visualization tools of the samples' quality.

## User Guide

The pipeline can be used simply by running:
```
from scMoMsQC import QC_Summary

QC_Summary(inpath, outpath).create_summary()
```

### Pipeline Input

The `QC_Summary` class is constructed as follows:

```
def __init__(self, inpath, outpath, nb_decimals=3, func=None):
```
- The `inpath` argument is the path to the folder containing the sample folders. The sample folders should contain an `/outs` subfile which itself should contain the `summary.csv` containing all QC metrics for this sample.

```
inpath/
├── ...
├── ...
├── ...
├── ...
│   └── outs
│       ├── ...
│       │
...     └── summary.csv
└── ...
    └── outs
        ├── ...
        └── summary.csv
```

- The `outpath` is simply the folder where you want the `qc_summary.html` to be written. More information about it is provided in the following section.
- The `nb_decimals` argument specifies the number of digits . This can have great influence on the outlier detection. If 1e-10 precision is provided, a deviation of this amount can produce an outlying value. This parameter is tuned according to the precision you deem to be relevant. Default value is `nb_decimals=3`.
- The `func` argument specifies an user-defined outlier detection function. More information below. Default is `func=None`.

### Pipeline Output

The pipeline writes an HTML file called `qc_summary.html`. It displays a **summary table** containing the number of alerts returned by CellRanger for each sample, if this sample is an outlier according to each of the outlier detection methds and the number of outlying QC metrics. It also returns a **table of all QC metrics' values**, a **violin plot**, a **clustermap** and a **PCA**. All this plots also provide information on outliers and alerts. An example can be found in the `Example` folder.

### Adding Your Own Outlier Detection Algorithm

The `func` argument can specify a custom outlier detection algorithm. It should satisfy the following constraints:
- The function should take as only argument a metrics `pandas.DataFrame`.
- The function should return a dataframe of booleans of the same shape, index and column labels as the metrics one, with `True` for outliers and `False` for inliers.
For example:

```
from QC_Summary import QC_Summary

def test(df):
	return df.apply(lambda x: (x > 3) | (x < -3))
	
QC_Summary(inpath, path, func=test).create_summary()
```

The output can be found in the `Example` folder under the name `qc_summary_func`. You may notice that the whole summary now displays this function's results, and that these results are rather close to that of the Z-Score function. That is because `func` is fed scaled and centered data using robust estimators and therefore acts like a Z-Score function. You may if course implement more complex functions, as is done with the Local Outlier Factor from scikit-learn in `/src/OD_functions.py`.

### Estimating The Outlier Detection Algorithms' Performances

The `QC_Summary` class proposes a function for estimating the performances of the outlier detection algorithms to help the user decide which algorithm to trust more. 
- A synthetic dataset is created based on the input one: these have the same number of metrics and the same number of samples as the original. Each metric follows a normal distribution with as center the median of the original's distribution for this metric and for scale the Qn scale. A random number of values is replaced by extreme values. These are considered outliers.
- The outlier detection algorithms are ran on this synthetic dataset.
- Their results are compared to the ground truth and a F1 score is computed.
- Repeat as many time as needed to obtain a representative F1 value within a maximum of 10 seconds.
- A dictionnary of `method : F1` is returned.
For example:

```
from QC_Summary import QC_Summary

def test(df):
	return df.apply(lambda x: (x > 3) | (x < -3))
	
perf = QC_Summary(inpath, outpath, func=test).compute_performances()
print(perf)

computing performances...
zscore
dixon_q_test
LOF
test
performances done!
{'Z-Score F1 score': np.float64(0.8081092732671629), 'Dixon Q Test F1 score': np.float64(0.13345971470191592), 'LOF F1 score': np.float64(0.13989874292457125), 'test F1 score': np.float64(0.8558519694745618)}
```

> [!NOTE]
> You may object that Dixon and LOF have a very low score. For LOF, this is probably due to the normal assumption that is made for the synthetic datasets and the sparcity of the generated outliers. You may inspect the relevance of this assumption on the violin plot. As for Dixon, the way it estimates the number of values to be inspected based on the input size may be the cause of this low performance which may not necessarily be representative.

## Code Overview

1. Fetch and merge all `summary.csv` files produced by the **10xGenomics Cellranger** pipeline in RNAseq and Multiomics singe-cell context.
2. Clean the resulting `pandas.DataFrame`, robustly scale it, compute its potential outliers with every dedicated algorithm, and generate alerts using [Cell Ranger's thresholds](https://cdn.10xgenomics.com/image/upload/v1660261286/support-documents/CG000329_TechnicalNote_InterpretingCellRangerWebSummaryFiles_RevA.pdf). The default outlier detection algorithms are robust Z-Score[^1], Dixon Q Test[^2],[^3] and Local Outlier Factor[^4]. This step mainly relies on Pandas[^5], Numpy[^6], Statsmodels[^7] and Scikit-learn[^8].
3. Generate the interactive plots in HTML using Plotly[^9].
4. Integrate these plots into a full-fledged HTML using Yattag[^10].
5. Generate the final HTML.

> [!WARNING]
> The outlier detection is based on the input data. If all samples are low quality, that is outliers with regard to a normal set of samples, none will be detected as outliers. You may control the outlier detection relevance using the `Alerts` count.

## Requirements

The pipeline runs on Python >= 3.9 and uses the matching distributions of numpy, pandas, scikit-learn, statsmodels, plotly and yattag.

## Bibliography

[^1]: [M. Daszykowski, K. Kaczmarek, Y. Vander Heyden, B. Walczak, Robust statistics in data analysis — A review: Basic concepts, Chemometrics and Intelligent Laboratory Systems, Volume 85, Issue 2, 2007, Pages 203-219, ISSN 0169-7439, https://doi.org/10.1016/j.chemolab.2006.06.016.](https://www.sciencedirect.com/science/article/pii/S0169743906001493)

[^2]: [Dixon, W. J. “Analysis of Extreme Values.” The Annals of Mathematical Statistics 21, no. 4 (1950): 488–506. http://www.jstor.org/stable/2236602.](https://www.jstor.org/stable/2236602)
[^3]: [Rorabacher, David B.. “Statistical treatment for rejection of deviant values: critical values of Dixon's "Q" parameter and related subrange ratios at the 95% confidence level.” Analytical Chemistry 63 (1991): 139-146.](https://www.semanticscholar.org/paper/Statistical-treatment-for-rejection-of-deviant-of-Rorabacher/f72157d3683fd5df5af65e816a211e8aef6cab23)
[^4]: [Markus M. Breunig, Hans-Peter Kriegel, Raymond T. Ng, and Jörg Sander. 2000. LOF: identifying density-based local outliers. In Proceedings of the 2000 ACM SIGMOD international conference on Management of data (SIGMOD '00). Association for Computing Machinery, New York, NY, USA, 93–104. https://doi.org/10.1145/342009.335388](https://dl.acm.org/doi/10.1145/335191.335388)
[^5]: The pandas development team. (2024). pandas-dev/pandas: Pandas (v2.2.2). Zenodo. https://doi.org/10.5281/zenodo.10957263
[^6]: Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). DOI: 10.1038/s41586-020-2649-2. 
[^7]: Seabold, S., & Perktold, J. (2010). statsmodels: Econometric and statistical modeling with python. In 9th Python in Science Conference.
[^8]: Scikit-learn: Machine Learning in Python, Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, Jake Vanderplas, Alexandre Passos, David Cournapeau, Matthieu Brucher, Matthieu Perrot, Édouard Duchesnay; 12(85):2825−2830, 2011.
[^9]: Plotly Technologies Inc. Collaborative data science. Montréal, QC, 2015. https://plot.ly.
[^10]: [Yattag, LeForestier](https://github.com/leforestier/yattag)

## Acknowledgements

This pipeline was realized under the direction of Dr. Vikas Bansal and Dr. Mohammed Dehestani in the group Biomedical Datascience at the Deutsches Zentrum für Neurodegenerative Erkrankungen (DZNE) in Tübingen.
# scMoMsQC
