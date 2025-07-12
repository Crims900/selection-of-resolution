# selection-of-resolution

Code and Datasets for the manuscript of  "An entropy-based approach for specifying the resolution parameter and cluster number in spatially resolved transcriptomics analysis"

## Requirements

anndata==0.10.2




annotated-types==0.6.0



GraphST==1.1.1



igraph==0.10.8



ipython==8.23.0



louvain==0.8.1



matplotlib==3.8.0



matplotlib-inline==0.1.6



networkx==3.2



numba==0.58.1



numpy==1.23.2



opencv-python==4.8.1.78



pandas==2.0.3



Pillow==9.5.0



python-igraph==0.10.8



python-louvain==0.16



scanpy==1.9.5



scikit-learn==1.3.2



scipy==1.11.3



seaborn==0.13.0



sklearn==0.0.post10



SpaGCN==1.2.7



STAGATE-pyG==1.0.0



stlearn==0.4.12



torch==2.1.0+cu118



torchtoolbox==0.1.8.2





## Overview

We outline the workflow of our approach for stabilization and selection of the appropriate resolution, which is suitable for a series of spatial clustering methods based on graph networks. We take the SpaGCN method as an example. Due to inherent randomness, SpaGCN generates different results with varying random seeds. First, we evaluate the amount of clustering information contained in the results based on the clarity of the clustering. Next, for each selected random result, we calculate the similarity degree between pairs of sample points. Finally, we fuse the similarity matrices and determine an appropriate resolution parameter using the cluster-entropy-based selection strategy. Then we can obtain stable clustering results using the Louvain algorithm with the selected resolution parameter.

For similar methods other than SpaGCN, such as GraphST and STAGATE, the above strategy can't be used due to the differences in the implementations of the Louvain algorithm. However, the cluster entropy derived from the SpaGCN clustering results can still serve as a reasonable reference for these methods. Therefore, we are able to determine the appropriate resolution parameter for these methods. 



## Datasets

For DLPFC dataset, see [http://spatial.libd.org/spatialLIBD](http://spatial.libd.org/spatialLIBD).



For Stereo-seq mouse embryo dataset, see [https://db.cngb.org/stomics/mosta](https://db.cngb.org/stomics/mosta/).



For other datasets, see "Datasets" folder.
