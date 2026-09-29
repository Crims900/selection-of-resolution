import os,csv,re
import pandas as pd
import numpy as np
import scanpy as sc
import math
from scipy.sparse import issparse
import random, torch
import warnings
warnings.filterwarnings("ignore")
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import numba

import cv2
from sklearn.metrics import adjusted_rand_score
from scanpy import read_10x_h5
from pathlib import Path
import time

PROJECT_ROOT = "/Data/Programs/SpaGCN_stabilization/stabilization_part/"
DATASET_ROOT = "/Data/Datasets/"
import SpaGCN_1000_random_seeds as spg
import anndata
from sklearn.metrics import adjusted_rand_score
import scanpy as sc
import SpaGCN_1000_random_seeds as spg

def initialization_load_adata(index):
    result_dir = PROJECT_ROOT + index + "/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    base_dir =  result_dir + "1000results/"

    if index in ["MERFISH_0.04","MERFISH_0.09","MERFISH_0.14","MERFISH_0.19","MERFISH_0.24"]:
        dataset_dir = DATASET_ROOT + "MERFISH/"
        adata = sc.read_h5ad(dataset_dir + index + ".h5ad")

    if index in ["E9.5_E1S1.MOSTA","E9.5_E2S1.MOSTA","E9.5_E2S2.MOSTA","E9.5_E2S3.MOSTA","E9.5_E2S4.MOSTA"]:
        dataset_dir = DATASET_ROOT + "Stereo_seq/"
        adata = sc.read_h5ad(dataset_dir + index + ".h5ad")

    if index in ["151507", "151508", "151509", "151510", "151669", "151670",
                 "151671", "151672", "151673", "151674", "151675","151676"]:
        dataset_dir = DATASET_ROOT + "SpatialLIBD/" + index + "/"
        adata = sc.read_h5ad(dataset_dir + "processed_adata.h5ad")

    if index in ["Slice_1","Slice_2","Slice_3"]:
        dataset_dir = DATASET_ROOT + "BARISTAseq/mouse_primary_visual_cortex/"
        adata = sc.read_h5ad(dataset_dir + index + ".h5ad")

    if index in ["BZ5","BZ14","BZ97","STARmap_BY3_1k"]:
        dataset_dir = DATASET_ROOT + "STARmap/"
        adata = sc.read_h5ad(dataset_dir + index + ".h5ad")

    print(adata)
    if index == "Slice_3":
        # Normalization
        adata.var_names_make_unique()
        spg.prefilter_genes(adata, min_cells=3)  # avoiding all genes are zeros
        spg.prefilter_specialgenes(adata)
        # Normalize and take log for UMI
        sc.pp.normalize_per_cell(adata)
        sc.pp.log1p(adata)

    return result_dir,base_dir,adata

def calculate_ari(adata,idx,base_dir):
    gcn_label = pd.read_csv(base_dir+str(idx)+"_y_pred"+".txt",header = None)
    gcn_label = gcn_label[0].tolist()
    try:
        truth = adata.obs["ground_truth"].tolist()
    except:
        truth = adata.obs["annotation"].tolist()

    ari = adjusted_rand_score(truth,gcn_label)
    return ari

def calculate_1000_ari(adata,base_dir,result_dir):
    ari_list = []
    for i in range(1,1001):
        ari = calculate_ari(adata,i,base_dir)
        ari_list.append(ari)
    ari_list = np.array(ari_list)
    np.savetxt(result_dir + "ari_1000_list.txt", ari_list, delimiter=',')

if __name__ == '__main__':
    for index in ["151507","151508","151509","151510","151669","151670","151671","151672","151673","151674","151675","151676"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#for index in ["Slice_1","Slice_2","Slice_3"]:
#for index in ["MERFISH_0.04","MERFISH_0.09","MERFISH_0.14","MERFISH_0.19","MERFISH_0.24"]:
#for index in ["BZ5","BZ14","BZ97","STARmap_BY3_1k"]:
#for index in ["E9.5_E1S1.MOSTA","E9.5_E2S1.MOSTA","E9.5_E2S2.MOSTA","E9.5_E2S3.MOSTA","E9.5_E2S4.MOSTA"]:

        time1 = time.time()
        result_dir,base_dir,adata = initialization_load_adata(index)
        calculate_1000_ari(adata,base_dir,result_dir)
        time2 = time.time()
        print("Time for " + index + " is: " + str(time2-time1))
