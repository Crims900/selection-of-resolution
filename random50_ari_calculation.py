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
import SpaGCN_1000_random_seeds as spg
import anndata
from sklearn.metrics import adjusted_rand_score
import scanpy as sc
import csv

PROJECT_ROOT = "/Data/Programs/SpaGCN_stabilization/stabilization_part/"
DATASET_ROOT = "/Data/Datasets/"

RESOLUTIONS = [round(float(res), 2) for res in np.linspace(1.03, 1.12, 10)]

def initialization_load_adata(index):
    result_dir = PROJECT_ROOT + index + "/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    base_dir = result_dir + "50_random/louvain_results/"
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

    if index == "Slice_3":
        # Normalization
        adata.var_names_make_unique()
        spg.prefilter_genes(adata, min_cells=3)  # avoiding all genes are zeros
        spg.prefilter_specialgenes(adata)
        # Normalize and take log for UMI
        sc.pp.normalize_per_cell(adata)
        sc.pp.log1p(adata)

    return result_dir,base_dir,adata

# Generate a 50x10 data frame: rows are random batches and columns are resolutions.
def calculate_ari(adata,idx,res,base_dir):
    gcn_label = pd.read_csv(base_dir+"clustering_"+str(idx)+"_"+str(res)+".csv")
    gcn_label = gcn_label.cluster.tolist()
    try:
        truth = adata.obs["ground_truth"].tolist()
    except:
        truth = adata.obs["annotation"].tolist()
    ari = adjusted_rand_score(truth,gcn_label)
    return ari

def calculate_random_ari(adata,base_dir,result_dir):
    with open(result_dir + "ari_list.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["idx"] + [f"{res:.2f}" for res in RESOLUTIONS])
        for i in range(1,51):
            ari_list = []
            for res in RESOLUTIONS:
                ari = calculate_ari(adata,i,res,base_dir)
                ari_list.append(ari)
                print("idx: "+str(i)+" res: "+str(res)+" done!")
            writer.writerow([i]+ari_list)


if __name__ == '__main__':
    for index in ["MERFISH_0.04","MERFISH_0.09","MERFISH_0.14","MERFISH_0.19","MERFISH_0.24",
                  "E9.5_E1S1.MOSTA","E9.5_E2S1.MOSTA","E9.5_E2S2.MOSTA","E9.5_E2S3.MOSTA","E9.5_E2S4.MOSTA",
                  "151507", "151508", "151509", "151510", "151669", "151670","151671", "151672", "151673", "151674", "151675","151676",
                  "Slice_1","Slice_2","Slice_3","BZ5","BZ14","BZ97","STARmap_BY3_1k"]:
        time1 = time.time()
        result_dir,base_dir,adata = initialization_load_adata(index)
        calculate_random_ari(adata,base_dir,result_dir)
        time2 = time.time()
        print(index + " cost time: " + str(time2 - time1))
