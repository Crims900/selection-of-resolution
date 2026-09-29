import os

import numpy as np
import torch
import pandas as pd
import scanpy as sc
from sklearn import metrics
import multiprocessing as mp
from GraphST import GraphST
import csv
import time
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
from sklearn.metrics.cluster import entropy
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score
import SpaGCN_1000_random_seeds as spg
from skimage.metrics import variation_of_information
from matplotlib import gridspec
import warnings

PROJECT_ROOT = "/Data/Programs/SpaGCN_stabilization/stabilization_part/"
DATASET_ROOT = "/Data/Datasets/"
SUMMARY_ROOT = PROJECT_ROOT + "summary/"
GRAPHST_ROOT = PROJECT_ROOT + "GraphST/"
STAGATE_ROOT = PROJECT_ROOT + "STAGATE/"




def refine(sample_id, pred, if_main, dis, shape="hexagon"):
    refined_pred=[]
    pred=pd.DataFrame({"pred": pred,"if_main": if_main}, index=sample_id)
    dis_df=pd.DataFrame(dis, index=sample_id, columns=sample_id)
    if shape=="hexagon":
        num_nbs=6
    elif shape=="square":
        num_nbs=4
    else:
        raise ValueError("Shape not recognized; use 'hexagon' for Visium data or 'square' for ST data.")
    for i in range(len(sample_id)):
        index=sample_id[i]
        if pred["if_main"].loc[index]==1:
            refined_pred.append(pred.loc[index, "pred"])
            continue
        dis_tmp=dis_df.loc[index, if_main].sort_values()
        nbs=dis_tmp[0:num_nbs+1]
        nbs_pred=pred.loc[nbs.index, "pred"]
        v_c=nbs_pred.value_counts()
        refined_pred.append(v_c.idxmax())
    return refined_pred

def initialization_SpaGCN_SpatialLIBD(index):
    dataset_dir = DATASET_ROOT + "SpatialLIBD/" + index + "/"
    base_dir = PROJECT_ROOT + index + "/first100/louvain_results/"
    #The format is asw100_fusion_Euclid_median_res0.1.csv
    #index,cluster are the columns
    prefix = "asw100_fusion_Euclid_median_res"
    res_list = np.linspace(1.0,1.2,201)
    result_dir = SUMMARY_ROOT + "SpaGCN/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    adata = sc.read_h5ad(dataset_dir+"processed_adata.h5ad")

    return base_dir,result_dir,adata,prefix,res_list

def initialization_SpaGCN_Stereoseq(index):
    dataset_dir = DATASET_ROOT + "Stereo_seq/"
    base_dir = PROJECT_ROOT + index + "/first100/louvain_results/"
    #The format is asw100_fusion_Euclid_median_res0.1.csv
    #index,cluster are the columns
    prefix = "asw100_fusion_Euclid_median_res"
    res_list = np.linspace(1.0,1.2,201)
    result_dir = SUMMARY_ROOT + "SpaGCN/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    adata = sc.read_h5ad(dataset_dir + index + ".h5ad")
    adata.obs["ground_truth"] = adata.obs["annotation"]

    return base_dir,result_dir,adata,prefix,res_list

def initialization_GraphST_SpatialLIBD(index):
    dataset_dir = DATASET_ROOT + "SpatialLIBD/" + index + "/"
    # dataset_dir = "/Data/Datasets/SpatialLIBD/" + index + "/"
    base_dir = GRAPHST_ROOT + index + "/first100/louvain_results/"
    # base_dir = "/Data/Programs/SpaGCN_stabilization/GraphST_calculation/" + index + "_first100/"+"louvain_results/"
    #The 0.1_res.csv
    suffix = "_res"
    res_list = np.linspace(0.3,1.5,1201)
    result_dir = SUMMARY_ROOT + "GraphST/"
    # result_dir = "/Data/Programs/SpaGCN_stabilization/asw_results/GraphST_method/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    adata = sc.read_h5ad(dataset_dir + "processed_adata.h5ad")

    return base_dir,result_dir,adata,suffix,res_list

def initialization_SpaGCN_BARISTAseq(index):
    dataset_dir = DATASET_ROOT + "BARISTAseq/mouse_primary_visual_cortex/"
    base_dir = PROJECT_ROOT + index + "/first100/louvain_results/"
    #The format is asw100_fusion_Euclid_median_res0.1.csv
    #index,cluster are the columns
    prefix = "asw100_fusion_Euclid_median_res"
    res_list = np.linspace(1.0,1.2,201)
    result_dir = SUMMARY_ROOT + "SpaGCN/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    adata = sc.read_h5ad(dataset_dir + index + ".h5ad")
    # Normalization
    adata.var_names_make_unique()
    spg.prefilter_genes(adata, min_cells=3)  # avoiding all genes are zeros
    spg.prefilter_specialgenes(adata)
    # Normalize and take log for UMI
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)

    return base_dir,result_dir,adata,prefix,res_list

def initialization_STAGATE_SpatialLIBD(index):
    dataset_dir = DATASET_ROOT + "SpatialLIBD/" + index + "/"
    base_dir = STAGATE_ROOT + index + "/first100/louvain_results/"
    #The 0.1_res.csv
    suffix = "_res"
    res_list = np.linspace(0.3,1.5,1201)
    result_dir = SUMMARY_ROOT + "STAGATE/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    adata = sc.read_h5ad(dataset_dir + "processed_adata.h5ad")

    return base_dir,result_dir,adata,suffix,res_list

def initialization_STAGATE_Stereoseq(index):
    dataset_dir = DATASET_ROOT + "Stereo_seq/"
    base_dir = STAGATE_ROOT + index + "/first100/louvain_results/"
    #The 0.1_res.csv
    suffix = "_res"
    res_list = np.linspace(0.3,1.5,1201)
    result_dir = SUMMARY_ROOT + "STAGATE/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    adata = sc.read_h5ad(dataset_dir + index + ".h5ad")
    adata.obs["ground_truth"] = adata.obs["annotation"]

    return base_dir,result_dir,adata,suffix,res_list



def initialization_GraphST_Stereoseq(index):
    dataset_dir = DATASET_ROOT + "Stereo_seq/"
    base_dir = GRAPHST_ROOT + index + "/first100/louvain_results/"
    #The 0.1_res.csv
    suffix = "_res"
    res_list = np.linspace(0.3,1.5,1201)
    result_dir = SUMMARY_ROOT + "GraphST/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    adata = sc.read_h5ad(dataset_dir + index + ".h5ad")
    adata.obs["ground_truth"] = adata.obs["annotation"]

    return base_dir,result_dir,adata,suffix,res_list

def initialization_GraphST_BARISTAseq(index):
    dataset_dir = DATASET_ROOT + "BARISTAseq/mouse_primary_visual_cortex/"
    base_dir = GRAPHST_ROOT + index + "/first100/louvain_results/"
    #The 0.1_res.csv
    suffix = "_res"
    res_list = np.linspace(0.3,1.5,1201)
    result_dir = SUMMARY_ROOT + "GraphST/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    adata = sc.read_h5ad(dataset_dir + index + ".h5ad")

    # adata.var_names_make_unique()
    # spg.prefilter_genes(adata, min_cells=3)  # avoiding all genes are zeros
    # spg.prefilter_specialgenes(adata)
    # # Normalize and take log for UMI
    # sc.pp.normalize_per_cell(adata)
    # sc.pp.log1p(adata)

    return base_dir,result_dir,adata,suffix,res_list

def initialization_STAGATE_BARISTAseq(index):
    dataset_dir = DATASET_ROOT + "BARISTAseq/mouse_primary_visual_cortex/"
    base_dir = STAGATE_ROOT + index + "/first100/louvain_results/"
    #The 0.1_res.csv
    suffix = "_res"
    res_list = np.linspace(0.3,1.5,1201)
    result_dir = SUMMARY_ROOT + "STAGATE/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    adata = sc.read_h5ad(dataset_dir + index + ".h5ad")

    # adata.var_names_make_unique()
    # spg.prefilter_genes(adata, min_cells=3)  # avoiding all genes are zeros
    # spg.prefilter_specialgenes(adata)
    # # Normalize and take log for UMI
    # sc.pp.normalize_per_cell(adata)
    # sc.pp.log1p(adata)

    return base_dir,result_dir,adata,suffix,res_list

def initialization_STAGATE_STARmap(index):
    dataset_dir = DATASET_ROOT + "STARmap/"
    base_dir = STAGATE_ROOT + index + "/first100/louvain_results/"
    suffix = "_res"
    res_list = np.linspace(0.3, 1.5, 1201)
    result_dir = SUMMARY_ROOT + "STAGATE/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    adata = sc.read_h5ad(dataset_dir + index + ".h5ad")

    return base_dir, result_dir, adata, suffix, res_list

def initialization_SpaGCN_MERFISH(index):
    dataset_dir = DATASET_ROOT + "MERFISH/"
    base_dir = PROJECT_ROOT + index + "/first100/louvain_results/"
    prefix = "asw100_fusion_Euclid_median_res"
    res_list = np.linspace(1.0, 1.199, 200)
    result_dir = SUMMARY_ROOT + "SpaGCN/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    adata = sc.read_h5ad(dataset_dir + index + ".h5ad")

    return base_dir, result_dir, adata, prefix, res_list

def initialization_STAGATE_MERFISH(index):
    dataset_dir = DATASET_ROOT + "MERFISH/"
    base_dir = STAGATE_ROOT + index + "/first100/louvain_results/"
    suffix = "_res"
    res_list = np.linspace(0.3, 1.5, 1201)
    result_dir = SUMMARY_ROOT + "STAGATE/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    adata = sc.read_h5ad(dataset_dir + index + ".h5ad")

    return base_dir, result_dir, adata, suffix, res_list

def initialization_GraphST_STARmap(index):
    dataset_dir = DATASET_ROOT + "STARmap/"
    base_dir = GRAPHST_ROOT + index + "/first100/louvain_results/"
    suffix = "_res"
    res_list = np.linspace(0.3, 1.5, 1201)
    result_dir = SUMMARY_ROOT + "GraphST/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    adata = sc.read_h5ad(dataset_dir + index + ".h5ad")
    return base_dir, result_dir, adata, suffix, res_list

def initialization_GraphST_MERFISH(index):
    dataset_dir = DATASET_ROOT + "MERFISH/"
    base_dir = GRAPHST_ROOT + index + "/first100/louvain_results/"
    suffix = "_res"
    res_list = np.linspace(0.3, 1.5, 1201)
    result_dir = SUMMARY_ROOT + "GraphST/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    adata = sc.read_h5ad(dataset_dir + index + ".h5ad")
    return base_dir, result_dir, adata, suffix, res_list

def initialization_SpaGCN_STARmap(index):
    dataset_dir = DATASET_ROOT + "STARmap/"
    base_dir = PROJECT_ROOT + index + "/first100/louvain_results/"
    prefix = "asw100_fusion_Euclid_median_res"
    res_list = np.linspace(1.0, 1.199, 200)
    result_dir = SUMMARY_ROOT + "SpaGCN/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    adata = sc.read_h5ad(dataset_dir + index + ".h5ad")

    return base_dir, result_dir, adata, prefix, res_list



def refine_a_res(res,adata,base_dir,prefix,adj_2d,if_Spa):
    adata.obs["x_pixel"] = adata.obsm["spatial"][:, 1]
    adata.obs["y_pixel"] = adata.obsm["spatial"][:, 0]
    res = round(res, 3)
    if res == 1:
        res_str = "1.0"
    else:
        res_str = str(res)
    if if_Spa == True:
        file = base_dir + prefix + res_str + ".csv"
    else:
        file = base_dir + res_str + prefix + ".csv"
    result = pd.read_csv(file, index_col=0)
    adata.obs["louvain"] = result['cluster'].tolist()
    adata.obs["louvain"] = adata.obs["louvain"].astype("category")

    refined_pred = spg.refine(sample_id=adata.obs.index.tolist(), pred=adata.obs["louvain"].tolist(), dis=adj_2d,
                              shape="hexagon")
    adata.obs["louvain"] = refined_pred
    adata.obs["louvain"] = adata.obs["louvain"].astype("category")

    # Tell whether refinement is needed
    cluster_num = sorted(list(Counter(adata.obs["louvain"]).values()), reverse=True)
    tmp = np.array(cluster_num)
    tmp = tmp / tmp.sum()
    entropy_list = (-1) * np.log(tmp) * tmp
    entropy = entropy_list.sum()
    entropy_percent = entropy_list / entropy
    number = adata.obs["louvain"].nunique()

    if_ref = 0
    percent_tail = 0
    num_tail = 0
    for percent in sorted(entropy_percent):
        if percent < 0.05:
            percent_tail += percent
            num_tail += 1
        else:
            break
    if percent_tail < 0.05:
        if_ref = 1

    # If refinement is needed, do refinement
    if if_ref == 1:
        main_counter = Counter(adata.obs["louvain"]).most_common()[:number - num_tail]
        main_cluster = [x[0] for x in main_counter]
        if_main = []
        for cluster in result['cluster'].tolist():
            if cluster in main_cluster:
                if_main.append(True)
            else:
                if_main.append(False)

        refined_pred = refine(sample_id=adata.obs.index.tolist(), pred=adata.obs["louvain"].tolist(), if_main=if_main,
                              dis=adj_2d,
                              shape="hexagon")
        adata.obs["louvain_ref"] = refined_pred

    return adata,if_ref


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    res_df = pd.read_excel(SUMMARY_ROOT + "summary_res.xlsx", index_col=0)
    save_dir = SUMMARY_ROOT + "figures/modify_figs/"
    index_list = ['151507',            '151508',            '151509',
                  '151510',            '151669',            '151670',
                  '151671',            '151672',            '151673',
                  '151674',            '151675',            '151676',
       'E9.5_E1S1.MOSTA', 'E9.5_E2S1.MOSTA', 'E9.5_E2S2.MOSTA',
       'E9.5_E2S3.MOSTA', 'E9.5_E2S4.MOSTA',         'Slice_1',
               'Slice_2',         'Slice_3',    'MERFISH_0.04',
          'MERFISH_0.09',    'MERFISH_0.14',    'MERFISH_0.19',
          'MERFISH_0.24',            'BZ14',             'BZ5',
                  'BZ97',          'STARmap_BY3_1k']
    # plot_color = glasbey.create_palette(palette_size=20)
    plot_color = ['#d21820', '#1869ff', '#008a00', '#f36dff', '#710079',
                  '#aafb00', '#00bec2', '#ffa235', '#5d3d04', '#08008a',
                  '#005d5d', '#9a7d82', '#a2aeff', '#96b675', '#9e28ff',
                  '#4d0014', '#ffaebe', '#ce0092', '#00ffb6', '#002d00','#9e7500', '#3d3541','#f3eb92','#65618a','#8a3d4d']



    # for i in [10]:
    #     fig = plt.figure(figsize=(26.8, 10.5))
    #     plt.subplots_adjust(wspace=0.6)
    #     plt.subplots_adjust(hspace=0.5)
    #     ax0 = plt.subplot2grid((10, 25), (0, 0), rowspan=9, colspan=9)
    #     ax1 = plt.subplot2grid((10, 25), (0, 9), rowspan=5, colspan=5)
    #     ax2 = plt.subplot2grid((10, 25), (0, 14), rowspan=5, colspan=5)
    #     ax3 = plt.subplot2grid((10, 25), (0, 19), rowspan=5, colspan=5)
    #     ax4 = plt.subplot2grid((10, 25), (5, 9), rowspan=5, colspan=5)
    #     ax5 = plt.subplot2grid((10, 25), (5, 14), rowspan=5, colspan=5)
    #     ax6 = plt.subplot2grid((10, 25), (5, 19), rowspan=5, colspan=5)
    #
    #
    #
    #     index = index_list[i]
    #     print("Start: "+index)
    #     res_list = res_df.iloc[i,:]
    #     for j in range(2):
    #         base_dir, _, adata, prefix, _ = initialization_SpaGCN_SpatialLIBD(index)
    #         x_pixel = adata.obsm["spatial"][:, 1].tolist()
    #         y_pixel = adata.obsm["spatial"][:, 0].tolist()
    #         adj_2d = spg.calculate_adj_matrix(x=x_pixel, y=y_pixel, histology=False)
    #         adata,if_ref = refine_a_res(res_list[j],adata,base_dir,prefix,adj_2d,if_Spa=True)
    #         if if_ref == 0:
    #             print("Something is wrong")
    #         if j == 0:
    #             # plot the ground truth
    #             domains = "ground_truth"
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             num_celltype = len(adata.obs[domains].unique())
    #             adata.uns[domains + "_colors"] = list(plot_color[:num_celltype])
    #             axs = ax0
    #             sc.pl.scatter(adata, alpha=1, x="y_pixel", y="x_pixel", color=domains, title="", ax=axs,
    #                                show=False, size=50000 / adata.shape[0] * 5)
    #             axs.set_aspect('equal', 'box')
    #             axs.set_yticklabels(axs.get_yticklabels(), rotation=90,ha="center",va="center",fontsize=7)
    #             axs.legend(ncol=5, markerscale=2.5, bbox_to_anchor=[0.5, -0.065], loc='upper center')
    #             axs.legend_.set_title("Domains")
    #             axs.axes.invert_yaxis()
    #
    #         if j%2 == 0:
    #             domains = "louvain"
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             adata.obs[domains] = adata.obs[domains].cat.codes + 1
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             axs = ax1
    #             sc.pl.scatter(adata, alpha=1, x="y_pixel", y="x_pixel", color=domains,title="",
    #                                    palette=plot_color, ax=axs, show=False, size=50000 / adata.shape[0] * 1.5, legend_fontsize='x-small')
    #             axs.set_xticklabels(axs.get_xticklabels(), fontsize=7)
    #             axs.set_yticklabels(axs.get_yticklabels(), rotation=90,ha="center",va="center",fontsize=7)
    #             axs.set_aspect('equal', 'box')
    #             axs.set_xlabel("")
    #             axs.annotate(
    #                 f"SpaGCN(ASW)",  # Title text
    #                 xy=(0.98, 0.99),  # Relative axes coordinates, slightly inside the plot area
    #                 xycoords="axes fraction",  # Use axes-fraction coordinates (0-1 range)
    #                 ha="right",  # Horizontal alignment
    #                 va="top",  # Vertical alignment: bottom
    #                 fontsize=10,  # Font size
    #             )
    #             axs.axes.invert_yaxis()
    #
    #
    #         if j%2 == 1:
    #             domains = "louvain_ref"
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             adata.obs[domains] = adata.obs[domains].cat.codes + 1
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             # adata.uns[domains + "_colors"] = list(plot_color[:num_celltype])
    #             axs = ax4
    #             sc.pl.scatter(adata, alpha=1, x="y_pixel", y="x_pixel", color=domains,title="",
    #                                palette=plot_color,ax=axs, show=False, size=50000 / adata.shape[0]* 1.5, legend_fontsize='x-small')
    #             axs.set_xticklabels(axs.get_xticklabels(), fontsize=7)
    #             axs.set_yticklabels(axs.get_yticklabels(), rotation=90,ha="center",va="center",fontsize=7)
    #             axs.set_aspect('equal', 'box')
    #             axs.annotate(
    #                 f"SpaGCN(Entropy)",  # Title text
    #                 xy=(0.98, 0.99),  # Relative axes coordinates, slightly inside the plot area
    #                 xycoords="axes fraction",  # Use axes-fraction coordinates (0-1 range)
    #                 ha="right",  # Horizontal alignment
    #                 va="top",  # Vertical alignment: bottom
    #                 fontsize=10,  # Font size
    #             )
    #             axs.axes.invert_yaxis()
    #
    #     for j in range(2,4):
    #         base_dir, _, adata, prefix, _ = initialization_GraphST_SpatialLIBD(index)
    #         x_pixel = adata.obsm["spatial"][:, 1].tolist()
    #         y_pixel = adata.obsm["spatial"][:, 0].tolist()
    #         adj_2d = spg.calculate_adj_matrix(x=x_pixel, y=y_pixel, histology=False)
    #         adata, if_ref = refine_a_res(res_list[j], adata, base_dir, prefix, adj_2d, if_Spa=False)
    #         if if_ref == False:
    #             print("Something is wrong")
    #
    #         if j % 2 == 0:
    #             domains = "louvain"
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             adata.obs[domains] = adata.obs[domains].cat.codes + 1
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             axs = ax2
    #             sc.pl.scatter(adata, alpha=1, x="y_pixel", y="x_pixel", color=domains, title="",
    #                           palette=plot_color, ax=axs, show=False, size=50000 / adata.shape[0]* 1.5, legend_fontsize='x-small')
    #             axs.set_xticklabels(axs.get_xticklabels(), fontsize=7)
    #             axs.set_yticklabels(axs.get_yticklabels(), rotation=90,ha="center",va="center",fontsize=7)
    #             axs.set_aspect('equal', 'box')
    #             axs.annotate(
    #                 f"GraphST(ASW)",  # Title text
    #                 xy=(0.98, 0.99),  # Relative axes coordinates, slightly inside the plot area
    #                 xycoords="axes fraction",  # Use axes-fraction coordinates (0-1 range)
    #                 ha="right",  # Horizontal alignment
    #                 va="top",  # Vertical alignment: bottom
    #                 fontsize=10,  # Font size
    #             )
    #             axs.set_ylabel("")
    #             axs.set_xlabel("")
    #             axs.axes.invert_yaxis()
    #
    #         if j % 2 == 1:
    #             domains = "louvain_ref"
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             adata.obs[domains] = adata.obs[domains].cat.codes + 1
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             # adata.uns[domains + "_colors"] = list(plot_color[:num_celltype])
    #             axs = ax5
    #             sc.pl.scatter(adata, alpha=1, x="y_pixel", y="x_pixel", color=domains, title="",
    #                                palette=plot_color,ax=axs, show=False, size=50000 / adata.shape[0]* 1.5, legend_fontsize='x-small')
    #             axs.set_xticklabels(axs.get_xticklabels(), fontsize=7)
    #             axs.set_yticklabels(axs.get_yticklabels(), rotation=90,ha="center",va="center",fontsize=7)
    #             axs.set_aspect('equal', 'box')
    #             axs.annotate(
    #                 f"GraphST(Entropy)",  # Title text
    #                 xy=(0.98, 0.99),  # Relative axes coordinates, slightly inside the plot area
    #                 xycoords="axes fraction",  # Use axes-fraction coordinates (0-1 range)
    #                 ha="right",  # Horizontal alignment
    #                 va="top",  # Vertical alignment: bottom
    #                 fontsize=10,  # Font size
    #             )
    #             axs.set_ylabel("")
    #             axs.axes.invert_yaxis()
    #
    #     for j in range(4,6):
    #         base_dir, _, adata, prefix, _ = initialization_STAGATE_SpatialLIBD(index)
    #         x_pixel = adata.obsm["spatial"][:, 1].tolist()
    #         y_pixel = adata.obsm["spatial"][:, 0].tolist()
    #         adj_2d = spg.calculate_adj_matrix(x=x_pixel, y=y_pixel, histology=False)
    #         adata, if_ref = refine_a_res(res_list[j], adata, base_dir, prefix, adj_2d, if_Spa=False)
    #         if if_ref == False:
    #             print("Something is wrong")
    #
    #         if j % 2 == 0:
    #             domains = "louvain"
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             adata.obs[domains] = adata.obs[domains].cat.codes + 1
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             axs = ax3
    #             sc.pl.scatter(adata, alpha=1, x="y_pixel", y="x_pixel", color=domains, title="",
    #                           palette=plot_color, ax=axs, show=False, size=50000 / adata.shape[0]* 1.5, legend_fontsize='x-small')
    #             axs.set_xticklabels(axs.get_xticklabels(), fontsize=7)
    #             axs.set_yticklabels(axs.get_yticklabels(), rotation=90,ha="center",va="center",fontsize=7)
    #             axs.set_ylabel("")
    #             axs.set_xlabel("")
    #             axs.annotate(
    #                 f"STAGATE(ASW)",  # Title text
    #                 xy=(0.98, 0.99),  # Relative axes coordinates, slightly inside the plot area
    #                 xycoords="axes fraction",  # Use axes-fraction coordinates (0-1 range)
    #                 ha="right",  # Horizontal alignment
    #                 va="top",  # Vertical alignment: bottom
    #                 fontsize=10,  # Font size
    #             )
    #             axs.set_aspect('equal', 'box')
    #             axs.axes.invert_yaxis()
    #
    #         if j % 2 == 1:
    #             domains = "louvain_ref"
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             adata.obs[domains] = adata.obs[domains].cat.codes + 1
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             # adata.uns[domains + "_colors"] = list(plot_color[:num_celltype])
    #             axs = ax6
    #             sc.pl.scatter(adata, alpha=1, x="y_pixel", y="x_pixel", color=domains, title="",
    #                                palette=plot_color, ax=axs, show=False, size=50000 / adata.shape[0]* 1.5, legend_fontsize='x-small')
    #             axs.set_xticklabels(axs.get_xticklabels(), fontsize=7)
    #             axs.set_yticklabels(axs.get_yticklabels(), rotation=90,ha="center",va="center",fontsize=7)
    #             axs.set_ylabel("")
    #             axs.annotate(
    #                 f"STAGATE(Entropy)",  # Title text
    #                 xy=(0.98, 0.99),  # Relative axes coordinates, slightly inside the plot area
    #                 xycoords="axes fraction",  # Use axes-fraction coordinates (0-1 range)
    #                 ha="right",  # Horizontal alignment
    #                 va="top",  # Vertical alignment: bottom
    #                 fontsize=10,  # Font size
    #             )
    #             axs.set_aspect('equal', 'box')
    #             axs.axes.invert_yaxis()
    #
    #     plt.tight_layout()
    #     plt.savefig(save_dir+index+".png",dpi = 600, bbox_inches='tight',format='png')
    #     plt.close()

    # for i in [15]:
    #     fig = plt.figure(figsize=(31, 10.1))
    #     plt.subplots_adjust(wspace=2)
    #     plt.subplots_adjust(hspace=0.1)
    #     ax0 = plt.subplot2grid((10, 25), (0, 0), rowspan=9, colspan=9)
    #     ax1 = plt.subplot2grid((10, 25), (0, 9), rowspan=5, colspan=5)
    #     ax2 = plt.subplot2grid((10, 25), (0, 14), rowspan=5, colspan=5)
    #     ax3 = plt.subplot2grid((10, 25), (0, 19), rowspan=5, colspan=5)
    #     ax4 = plt.subplot2grid((10, 25), (5, 9), rowspan=5, colspan=5)
    #     ax5 = plt.subplot2grid((10, 25), (5, 14), rowspan=5, colspan=5)
    #     ax6 = plt.subplot2grid((10, 25), (5, 19), rowspan=5, colspan=5)
    #     index = index_list[i]
    #     print("Start: " + index)
    #     res_list = res_df.iloc[i, :]
    #     for j in range(2):
    #         base_dir, _, adata, prefix, _ = initialization_SpaGCN_Stereoseq(index)
    #         x_pixel = adata.obsm["spatial"][:, 1].tolist()
    #         y_pixel = adata.obsm["spatial"][:, 0].tolist()
    #         adj_2d = spg.calculate_adj_matrix(x=x_pixel, y=y_pixel, histology=False)
    #         adata, if_ref = refine_a_res(res_list[j], adata, base_dir, prefix, adj_2d, if_Spa=True)
    #         ground = adata.obs["ground_truth"].tolist()
    #         ground = ['BA' if x == 'Branchial arch' else x for x in ground]
    #         ground = ['Mes.' if x == 'Mesenchyme' else x for x in ground]
    #         ground = ['HD' if x == 'Head mesenchyme' else x for x in ground]
    #         ground = ['LP' if x == 'Lung primordium' else x for x in ground]
    #         ground = ['Noto.' if x == 'Notochord' else x for x in ground]
    #         ground = ['Sc.' if x == 'Sclerotome' else x for x in ground]
    #         ground = ['PP' if x == 'Pancreas primordium' else x for x in ground]
    #         ground = ['SC' if x == 'Spinal cord' else x for x in ground]
    #         ground = ['SE' if x == 'Surface ectoderm' else x for x in ground]
    #         ground = ['PGT' if x == 'Primitive gut tube' else x for x in ground]
    #         ground = ['CT' if x == 'Connective tissue' else x for x in ground]
    #         ground = ['Derm.' if x == 'Dermomyotome' else x for x in ground]
    #         ground = ['NC' if x == 'Neural crest' else x for x in ground]
    #
    #         adata.obs["ground_truth"] = ground
    #         adata.obs["ground_truth"] = adata.obs["ground_truth"].astype("category")
    #         if if_ref == 0:
    #             print("Something is wrong")
    #         if j == 0:
    #             # plot the ground truth
    #             domains = "ground_truth"
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             num_celltype = len(adata.obs[domains].unique())
    #             adata.uns[domains + "_colors"] = list(plot_color[:num_celltype])
    #             axs = ax0
    #             sc.pl.scatter(adata, alpha=1, x="y_pixel", y="x_pixel", color=domains, title="", ax=axs,
    #                           show=False, size=50000 / adata.shape[0] * 5)
    #             axs.set_aspect('equal', 'box')
    #             axs.set_yticklabels(axs.get_yticklabels(), rotation=90, ha="center", va="center", fontsize=7)
    #             axs.legend(ncol=7, markerscale=2.5, bbox_to_anchor=[0.5, -0.065], loc='upper center', fontsize='small')
    #             axs.legend_.set_title("Domains")
    #             axs.axes.invert_yaxis()
    #
    #         if j % 2 == 0:
    #             domains = "louvain"
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             adata.obs[domains] = adata.obs[domains].cat.codes + 1
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             axs = ax1
    #             sc.pl.scatter(adata, alpha=1, x="y_pixel", y="x_pixel", color=domains, title="",
    #                           palette=plot_color, ax=axs, show=False, size=50000 / adata.shape[0] * 1.5,
    #                           legend_fontsize='x-small')
    #             axs.set_xticklabels(axs.get_xticklabels(), fontsize=7)
    #             axs.set_yticklabels(axs.get_yticklabels(), rotation=90, ha="center", va="center", fontsize=7)
    #             axs.set_aspect('equal', 'box')
    #             axs.set_xlabel("")
    #             axs.annotate(
    #                 f"SpaGCN(ASW)",  # Title text
    #                 xy=(0.98, 0.99),  # Relative axes coordinates, slightly inside the plot area
    #                 xycoords="axes fraction",  # Use axes-fraction coordinates (0-1 range)
    #                 ha="right",  # Horizontal alignment
    #                 va="top",  # Vertical alignment: bottom
    #                 fontsize=10,  # Font size
    #             )
    #             axs.axes.invert_yaxis()
    #
    #         if j % 2 == 1:
    #             domains = "louvain_ref"
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             adata.obs[domains] = adata.obs[domains].cat.codes + 1
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             # adata.uns[domains + "_colors"] = list(plot_color[:num_celltype])
    #             axs = ax4
    #             sc.pl.scatter(adata, alpha=1, x="y_pixel", y="x_pixel", color=domains, title="",
    #                           palette=plot_color, ax=axs, show=False, size=50000 / adata.shape[0] * 1.5,
    #                           legend_fontsize='x-small')
    #             axs.set_xticklabels(axs.get_xticklabels(), fontsize=7)
    #             axs.set_yticklabels(axs.get_yticklabels(), rotation=90, ha="center", va="center", fontsize=7)
    #             axs.set_aspect('equal', 'box')
    #             axs.annotate(
    #                 f"SpaGCN(Entropy)",  # Title text
    #                 xy=(0.98, 0.99),  # Relative axes coordinates, slightly inside the plot area
    #                 xycoords="axes fraction",  # Use axes-fraction coordinates (0-1 range)
    #                 ha="right",  # Horizontal alignment
    #                 va="top",  # Vertical alignment: bottom
    #                 fontsize=10,  # Font size
    #             )
    #             axs.axes.invert_yaxis()
    #
    #     for j in range(2, 4):
    #         base_dir, _, adata, prefix, _ = initialization_GraphST_Stereoseq(index)
    #         x_pixel = adata.obsm["spatial"][:, 1].tolist()
    #         y_pixel = adata.obsm["spatial"][:, 0].tolist()
    #         adj_2d = spg.calculate_adj_matrix(x=x_pixel, y=y_pixel, histology=False)
    #         adata, if_ref = refine_a_res(res_list[j], adata, base_dir, prefix, adj_2d, if_Spa=False)
    #         if if_ref == False:
    #             print("Something is wrong")
    #
    #         if j % 2 == 0:
    #             domains = "louvain"
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             adata.obs[domains] = adata.obs[domains].cat.codes + 1
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             axs = ax2
    #             sc.pl.scatter(adata, alpha=1, x="y_pixel", y="x_pixel", color=domains, title="",
    #                           palette=plot_color, ax=axs, show=False, size=50000 / adata.shape[0] * 1.5,
    #                           legend_fontsize='x-small')
    #             axs.set_xticklabels(axs.get_xticklabels(), fontsize=7)
    #             axs.set_yticklabels(axs.get_yticklabels(), rotation=90, ha="center", va="center", fontsize=7)
    #             axs.set_aspect('equal', 'box')
    #             axs.set_ylabel("")
    #             axs.set_xlabel("")
    #             axs.annotate(
    #                 f"GraphST(ASW)",  # Title text
    #                 xy=(0.98, 0.99),  # Relative axes coordinates, slightly inside the plot area
    #                 xycoords="axes fraction",  # Use axes-fraction coordinates (0-1 range)
    #                 ha="right",  # Horizontal alignment
    #                 va="top",  # Vertical alignment: bottom
    #                 fontsize=10,  # Font size
    #             )
    #             axs.axes.invert_yaxis()
    #
    #         if j % 2 == 1:
    #             domains = "louvain_ref"
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             adata.obs[domains] = adata.obs[domains].cat.codes + 1
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             # adata.uns[domains + "_colors"] = list(plot_color[:num_celltype])
    #             axs = ax5
    #             sc.pl.scatter(adata, alpha=1, x="y_pixel", y="x_pixel", color=domains, title="",
    #                           palette=plot_color, ax=axs, show=False, size=50000 / adata.shape[0] * 1.5,
    #                           legend_fontsize='x-small')
    #             axs.set_xticklabels(axs.get_xticklabels(), fontsize=7)
    #             axs.set_yticklabels(axs.get_yticklabels(), rotation=90, ha="center", va="center", fontsize=7)
    #             axs.set_aspect('equal', 'box')
    #             axs.set_ylabel("")
    #             axs.annotate(
    #                 f"GraphST(Entropy)",  # Title text
    #                 xy=(0.98, 0.99),  # Relative axes coordinates, slightly inside the plot area
    #                 xycoords="axes fraction",  # Use axes-fraction coordinates (0-1 range)
    #                 ha="right",  # Horizontal alignment
    #                 va="top",  # Vertical alignment: bottom
    #                 fontsize=10,  # Font size
    #             )
    #             axs.axes.invert_yaxis()
    #
    #     for j in range(4, 6):
    #         base_dir, _, adata, prefix, _ = initialization_STAGATE_Stereoseq(index)
    #         x_pixel = adata.obsm["spatial"][:, 1].tolist()
    #         y_pixel = adata.obsm["spatial"][:, 0].tolist()
    #         adj_2d = spg.calculate_adj_matrix(x=x_pixel, y=y_pixel, histology=False)
    #         adata, if_ref = refine_a_res(res_list[j], adata, base_dir, prefix, adj_2d, if_Spa=False)
    #         if if_ref == False:
    #             print("Something is wrong")
    #
    #         if j % 2 == 0:
    #             domains = "louvain"
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             adata.obs[domains] = adata.obs[domains].cat.codes + 1
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             axs = ax3
    #             sc.pl.scatter(adata, alpha=1, x="y_pixel", y="x_pixel", color=domains, title="",
    #                           palette=plot_color, ax=axs, show=False, size=50000 / adata.shape[0] * 1.5,
    #                           legend_fontsize='x-small',legend_fontweight='light')
    #             axs.set_xticklabels(axs.get_xticklabels(), fontsize=7)
    #             axs.set_yticklabels(axs.get_yticklabels(), rotation=90, ha="center", va="center", fontsize=7)
    #             axs.set_ylabel("")
    #             axs.set_xlabel("")
    #             axs.set_aspect('equal', 'box')
    #             axs.annotate(
    #                 f"STAGATE(ASW)",  # Title text
    #                 xy=(0.98, 0.99),  # Relative axes coordinates, slightly inside the plot area
    #                 xycoords="axes fraction",  # Use axes-fraction coordinates (0-1 range)
    #                 ha="right",  # Horizontal alignment
    #                 va="top",  # Vertical alignment: bottom
    #                 fontsize=10,  # Font size
    #             )
    #             axs.axes.invert_yaxis()
    #
    #         if j % 2 == 1:
    #             domains = "louvain_ref"
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             adata.obs[domains] = adata.obs[domains].cat.codes + 1
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             # adata.uns[domains + "_colors"] = list(plot_color[:num_celltype])
    #             axs = ax6
    #             sc.pl.scatter(adata, alpha=1, x="y_pixel", y="x_pixel", color=domains, title="",
    #                           palette=plot_color, ax=axs, show=False, size=50000 / adata.shape[0] * 1.5,
    #                           legend_fontsize='x-small')
    #             axs.set_xticklabels(axs.get_xticklabels(), fontsize=7)
    #             axs.set_yticklabels(axs.get_yticklabels(), rotation=90, ha="center", va="center", fontsize=7)
    #             axs.set_ylabel("")
    #             axs.set_aspect('equal', 'box')
    #             axs.annotate(
    #                 f"STAGATE(Entropy)",  # Title text
    #                 xy=(0.98, 0.99),  # Relative axes coordinates, slightly inside the plot area
    #                 xycoords="axes fraction",  # Use axes-fraction coordinates (0-1 range)
    #                 ha="right",  # Horizontal alignment
    #                 va="top",  # Vertical alignment: bottom
    #                 fontsize=10,  # Font size
    #             )
    #             axs.axes.invert_yaxis()
    #
    #     # plt.tight_layout()
    #     plt.savefig(save_dir + index + ".png", dpi=600, bbox_inches='tight',format='png')
    #     plt.close()
    #
    for i in range(17,20):
        fig = plt.figure(figsize=(23.8,8.5))
        plt.subplots_adjust(wspace=1.6)
        plt.subplots_adjust(hspace=0.5)
        ax0 = plt.subplot2grid((8, 19), (0, 0), rowspan=7, colspan=7)
        ax1 = plt.subplot2grid((8, 19), (0, 7), rowspan=4, colspan=4)
        ax2 = plt.subplot2grid((8, 19), (0, 11), rowspan=4, colspan=4)
        ax3 = plt.subplot2grid((8, 19), (0, 15), rowspan=4, colspan=4)
        ax4 = plt.subplot2grid((8, 19), (4, 7), rowspan=4, colspan=4)
        ax5 = plt.subplot2grid((8, 19), (4, 11), rowspan=4, colspan=4)
        ax6 = plt.subplot2grid((8, 19), (4, 15), rowspan=4, colspan=4)
        index = index_list[i]
        print("Start: "+index)
        res_list = res_df.iloc[i,:]
        for j in range(2):
            base_dir, _, adata, prefix, _ = initialization_SpaGCN_BARISTAseq(index)
            x_pixel = adata.obsm["spatial"][:, 1].tolist()
            y_pixel = adata.obsm["spatial"][:, 0].tolist()
            adj_2d = spg.calculate_adj_matrix(x=x_pixel, y=y_pixel, histology=False)
            adata,if_ref = refine_a_res(res_list[j],adata,base_dir,prefix,adj_2d,if_Spa=True)
            if if_ref == 0:
                print("Something is wrong")
            if j == 0:
                # plot the ground truth
                domains = "ground_truth"
                adata.obs[domains] = adata.obs[domains].astype("category")
                num_celltype = len(adata.obs[domains].unique())
                adata.uns[domains + "_colors"] = list(plot_color[:num_celltype])
                axs = ax0
                sc.pl.scatter(adata, alpha=1, x="y_pixel", y="x_pixel", color=domains, title="", ax=axs,
                                   show=False, size=50000 / adata.shape[0] * 5)
                axs.set_aspect('equal', 'box')
                axs.set_yticklabels(axs.get_yticklabels(), rotation=90,ha="center",va="center",fontsize=7)
                axs.legend(ncol=3, markerscale=1.5, bbox_to_anchor=[0.5, -0.07], loc='upper center')
                axs.legend_.set_title("Domains")
                axs.axes.invert_yaxis()

            if j%2 == 0:
                domains = "louvain"
                adata.obs[domains] = adata.obs[domains].astype("category")
                adata.obs[domains] = adata.obs[domains].cat.codes + 1
                adata.obs[domains] = adata.obs[domains].astype("category")
                axs = ax1
                sc.pl.scatter(adata, alpha=1, x="y_pixel", y="x_pixel", color=domains,title="",
                                       palette=plot_color, ax=axs, show=False, size=50000 / adata.shape[0] * 1.5, legend_fontsize='x-small')
                axs.set_xticklabels(axs.get_xticklabels(), fontsize=7)
                axs.set_yticklabels(axs.get_yticklabels(), rotation=90,ha="center",va="center",fontsize=7)
                axs.set_aspect('equal', 'box')
                axs.set_xlabel("")
                axs.annotate(
                    f"SpaGCN(ASW)",  # Title text
                    xy=(0.98, 0.02),  # Relative axes coordinates, slightly inside the plot area
                    xycoords="axes fraction",  # Use axes-fraction coordinates (0-1 range)
                    ha="right",  # Horizontal alignment
                    va="bottom",  # Vertical alignment: bottom
                    fontsize=10,  # Font size
                )
                axs.axes.invert_yaxis()


            if j%2 == 1:
                domains = "louvain_ref"
                adata.obs[domains] = adata.obs[domains].astype("category")
                adata.obs[domains] = adata.obs[domains].cat.codes + 1
                adata.obs[domains] = adata.obs[domains].astype("category")
                # adata.uns[domains + "_colors"] = list(plot_color[:num_celltype])
                axs = ax4
                sc.pl.scatter(adata, alpha=1, x="y_pixel", y="x_pixel", color=domains,title="",
                                   palette=plot_color,ax=axs, show=False, size=50000 / adata.shape[0]* 1.5, legend_fontsize='x-small')
                axs.set_xticklabels(axs.get_xticklabels(), fontsize=7)
                axs.set_yticklabels(axs.get_yticklabels(), rotation=90,ha="center",va="center",fontsize=7)
                axs.set_aspect('equal', 'box')
                axs.annotate(
                    f"SpaGCN(Entropy)",  # Title text
                    xy=(0.98, 0.02),  # Relative axes coordinates, slightly inside the plot area
                    xycoords="axes fraction",  # Use axes-fraction coordinates (0-1 range)
                    ha="right",  # Horizontal alignment
                    va="bottom",  # Vertical alignment: bottom
                    fontsize=10,  # Font size
                )
                axs.axes.invert_yaxis()


        for j in range(2,4):
            base_dir, _, adata, prefix, _ = initialization_GraphST_BARISTAseq(index)
            x_pixel = adata.obsm["spatial"][:, 1].tolist()
            y_pixel = adata.obsm["spatial"][:, 0].tolist()
            adj_2d = spg.calculate_adj_matrix(x=x_pixel, y=y_pixel, histology=False)
            adata, if_ref = refine_a_res(res_list[j], adata, base_dir, prefix, adj_2d, if_Spa=False)
            if if_ref == False:
                print("Something is wrong")

            if j % 2 == 0:
                domains = "louvain"
                adata.obs[domains] = adata.obs[domains].astype("category")
                adata.obs[domains] = adata.obs[domains].cat.codes + 1
                adata.obs[domains] = adata.obs[domains].astype("category")
                axs = ax2
                sc.pl.scatter(adata, alpha=1, x="y_pixel", y="x_pixel", color=domains, title="",
                              palette=plot_color, ax=axs, show=False, size=50000 / adata.shape[0]* 1.5, legend_fontsize='x-small')
                axs.set_xticklabels(axs.get_xticklabels(), fontsize=7)
                axs.set_yticklabels(axs.get_yticklabels(), rotation=90,ha="center",va="center",fontsize=7)
                axs.set_aspect('equal', 'box')
                axs.set_ylabel("")
                axs.set_xlabel("")
                axs.annotate(
                    f"GraphST(ASW)",  # Title text
                    xy=(0.98, 0.02),  # Relative axes coordinates, slightly inside the plot area
                    xycoords="axes fraction",  # Use axes-fraction coordinates (0-1 range)
                    ha="right",  # Horizontal alignment
                    va="bottom",  # Vertical alignment: bottom
                    fontsize=10,  # Font size
                )
                axs.axes.invert_yaxis()

            if j % 2 == 1:
                domains = "louvain_ref"
                adata.obs[domains] = adata.obs[domains].astype("category")
                adata.obs[domains] = adata.obs[domains].cat.codes + 1
                adata.obs[domains] = adata.obs[domains].astype("category")
                # adata.uns[domains + "_colors"] = list(plot_color[:num_celltype])
                axs = ax5
                sc.pl.scatter(adata, alpha=1, x="y_pixel", y="x_pixel", color=domains, title="",
                                   palette=plot_color,ax=axs, show=False, size=50000 / adata.shape[0]* 1.5, legend_fontsize='x-small')
                axs.set_xticklabels(axs.get_xticklabels(), fontsize=7)
                axs.set_yticklabels(axs.get_yticklabels(), rotation=90,ha="center",va="center",fontsize=7)
                axs.set_aspect('equal', 'box')
                axs.set_ylabel("")
                axs.annotate(
                    f"GraphST(Entropy)",  # Title text
                    xy=(0.98, 0.02),  # Relative axes coordinates, slightly inside the plot area
                    xycoords="axes fraction",  # Use axes-fraction coordinates (0-1 range)
                    ha="right",  # Horizontal alignment
                    va="bottom",  # Vertical alignment: bottom
                    fontsize=10,  # Font size
                )
                axs.axes.invert_yaxis()

        for j in range(4,6):
            base_dir, _, adata, prefix, _ = initialization_STAGATE_BARISTAseq(index)
            x_pixel = adata.obsm["spatial"][:, 1].tolist()
            y_pixel = adata.obsm["spatial"][:, 0].tolist()
            adj_2d = spg.calculate_adj_matrix(x=x_pixel, y=y_pixel, histology=False)
            adata, if_ref = refine_a_res(res_list[j], adata, base_dir, prefix, adj_2d, if_Spa=False)
            if if_ref == False:
                print("Something is wrong")

            if j % 2 == 0:
                domains = "louvain"
                adata.obs[domains] = adata.obs[domains].astype("category")
                adata.obs[domains] = adata.obs[domains].cat.codes + 1
                adata.obs[domains] = adata.obs[domains].astype("category")
                axs = ax3
                sc.pl.scatter(adata, alpha=1, x="y_pixel", y="x_pixel", color=domains, title="",
                              palette=plot_color, ax=axs, show=False, size=50000 / adata.shape[0]* 1.5, legend_fontsize='x-small')
                axs.set_xticklabels(axs.get_xticklabels(), fontsize=7)
                axs.set_yticklabels(axs.get_yticklabels(), rotation=90,ha="center",va="center",fontsize=7)
                axs.set_ylabel("")
                axs.set_xlabel("")
                axs.set_aspect('equal', 'box')
                axs.annotate(
                    f"STAGATE(ASW)",  # Title text
                    xy=(0.98, 0.02),  # Relative axes coordinates, slightly inside the plot area
                    xycoords="axes fraction",  # Use axes-fraction coordinates (0-1 range)
                    ha="right",  # Horizontal alignment
                    va="bottom",  # Vertical alignment: bottom
                    fontsize=10,  # Font size
                )
                axs.axes.invert_yaxis()

            if j % 2 == 1:
                domains = "louvain_ref"
                adata.obs[domains] = adata.obs[domains].astype("category")
                adata.obs[domains] = adata.obs[domains].cat.codes + 1
                adata.obs[domains] = adata.obs[domains].astype("category")
                # adata.uns[domains + "_colors"] = list(plot_color[:num_celltype])
                axs = ax6
                sc.pl.scatter(adata, alpha=1, x="y_pixel", y="x_pixel", color=domains, title="",
                                   palette=plot_color, ax=axs, show=False, size=50000 / adata.shape[0]* 1.5, legend_fontsize='x-small')
                axs.set_xticklabels(axs.get_xticklabels(), fontsize=7)
                axs.set_yticklabels(axs.get_yticklabels(), rotation=90,ha="center",va="center",fontsize=7)
                axs.set_ylabel("")
                axs.set_aspect('equal', 'box')
                axs.annotate(
                    f"STAGATE(Entropy)",  # Title text
                    xy=(0.98, 0.02),  # Relative axes coordinates, slightly inside the plot area
                    xycoords="axes fraction",  # Use axes-fraction coordinates (0-1 range)
                    ha="right",  # Horizontal alignment
                    va="bottom",  # Vertical alignment: bottom
                    fontsize=10,  # Font size
                )
                axs.axes.invert_yaxis()

        plt.tight_layout()
        plt.savefig(save_dir+index+".png",dpi = 600, bbox_inches='tight',format='png')
        plt.close()


    #
    # for i in range(24, 25):
    #     fig = plt.figure(figsize=(22.6,8.8))
    #     plt.subplots_adjust(wspace=0.7)
    #     plt.subplots_adjust(hspace=1.8)
    #     ax0 = plt.subplot2grid((16, 41), (0, 0), rowspan=14, colspan=14)
    #     ax1 = plt.subplot2grid((16, 41), (0, 15), rowspan=8, colspan=8)
    #     ax2 = plt.subplot2grid((16, 41), (0, 24), rowspan=8, colspan=8)
    #     ax3 = plt.subplot2grid((16, 41), (0, 33), rowspan=8, colspan=8)
    #     ax4 = plt.subplot2grid((16, 41), (8, 15), rowspan=8, colspan=8)
    #     ax5 = plt.subplot2grid((16, 41), (8, 24), rowspan=8, colspan=8)
    #     ax6 = plt.subplot2grid((16, 41), (8, 33), rowspan=8, colspan=8)
    #     index = index_list[i]
    #     print("Start: "+index)
    #     res_list = res_df.iloc[i,:]
    #     for j in range(2):
    #         base_dir, _, adata, prefix, _ = initialization_SpaGCN_MERFISH(index)
    #         x_pixel = adata.obsm["spatial"][:, 1].tolist()
    #         y_pixel = adata.obsm["spatial"][:, 0].tolist()
    #         adj_2d = spg.calculate_adj_matrix(x=x_pixel, y=y_pixel, histology=False)
    #         adata,if_ref = refine_a_res(res_list[j],adata,base_dir,prefix,adj_2d,if_Spa=True)
    #         if if_ref == 0:
    #             print("Something is wrong")
    #         if j == 0:
    #             # plot the ground truth
    #             domains = "ground_truth"
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             num_celltype = len(adata.obs[domains].unique())
    #             adata.uns[domains + "_colors"] = list(plot_color[:num_celltype])
    #             axs = ax0
    #             sc.pl.scatter(adata, alpha=1, x="y_pixel", y="x_pixel", color=domains, title="", ax=axs,
    #                                show=False, size=50000 / adata.shape[0] * 5)
    #             axs.set_aspect('equal', 'box')
    #             axs.set_yticklabels(axs.get_yticklabels(), rotation=90,ha="center",va="center",fontsize=7)
    #             axs.legend(ncol=4, markerscale=2.5, bbox_to_anchor=[0.5, -0.075], loc='upper center')
    #             axs.legend_.set_title("Domains")
    #             axs.axes.invert_yaxis()
    # 
    #         if j%2 == 0:
    #             domains = "louvain"
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             adata.obs[domains] = adata.obs[domains].cat.codes + 1
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             axs = ax1
    #             sc.pl.scatter(adata, alpha=1, x="y_pixel", y="x_pixel", color=domains,title="",
    #                                    palette=plot_color, ax=axs, show=False, size=50000 / adata.shape[0] * 1.5, legend_fontsize='x-small')
    #             axs.set_xticklabels(axs.get_xticklabels(), fontsize=7)
    #             axs.set_yticklabels(axs.get_yticklabels(), rotation=90,ha="center",va="center",fontsize=7)
    #             axs.set_aspect('equal', 'box')
    #             axs.set_xlabel("")
    #             axs.annotate(
    #                 f"SpaGCN(ASW)",  # Title text
    #                 xy=(0.65, 0.99),  # Relative axes coordinates, slightly inside the plot area
    #                 xycoords="axes fraction",  # Use axes-fraction coordinates (0-1 range)
    #                 ha="right",  # Horizontal alignment
    #                 va="top",  # Vertical alignment: bottom
    #                 fontsize=9,  # Font size
    #             )
    #             axs.axes.invert_yaxis()
    # 
    # 
    #         if j%2 == 1:
    #             domains = "louvain_ref"
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             adata.obs[domains] = adata.obs[domains].cat.codes + 1
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             # adata.uns[domains + "_colors"] = list(plot_color[:num_celltype])
    #             axs = ax4
    #             sc.pl.scatter(adata, alpha=1, x="y_pixel", y="x_pixel", color=domains,title="",
    #                                palette=plot_color,ax=axs, show=False, size=50000 / adata.shape[0]* 1.5, legend_fontsize='x-small')
    #             axs.set_xticklabels(axs.get_xticklabels(), fontsize=7)
    #             axs.set_yticklabels(axs.get_yticklabels(), rotation=90,ha="center",va="center",fontsize=7)
    #             axs.set_aspect('equal', 'box')
    #             axs.annotate(
    #                 f"SpaGCN(Entropy)",  # Title text
    #                 xy=(0.65, 0.99),  # Relative axes coordinates, slightly inside the plot area
    #                 xycoords="axes fraction",  # Use axes-fraction coordinates (0-1 range)
    #                 ha="right",  # Horizontal alignment
    #                 va="top",  # Vertical alignment: bottom
    #                 fontsize=9,  # Font size
    #             )
    #             axs.axes.invert_yaxis()
    # 
    #     for j in range(2,4):
    #         base_dir, _, adata, prefix, _ = initialization_GraphST_MERFISH(index)
    #         x_pixel = adata.obsm["spatial"][:, 1].tolist()
    #         y_pixel = adata.obsm["spatial"][:, 0].tolist()
    #         adj_2d = spg.calculate_adj_matrix(x=x_pixel, y=y_pixel, histology=False)
    #         adata, if_ref = refine_a_res(res_list[j], adata, base_dir, prefix, adj_2d, if_Spa=False)
    #         if if_ref == False:
    #             print("Something is wrong")
    # 
    #         if j % 2 == 0:
    #             domains = "louvain"
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             adata.obs[domains] = adata.obs[domains].cat.codes + 1
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             axs = ax2
    #             sc.pl.scatter(adata, alpha=1, x="y_pixel", y="x_pixel", color=domains, title="",
    #                           palette=plot_color, ax=axs, show=False, size=50000 / adata.shape[0]* 1.5, legend_fontsize='x-small')
    #             axs.set_xticklabels(axs.get_xticklabels(), fontsize=7)
    #             axs.set_yticklabels(axs.get_yticklabels(), rotation=90,ha="center",va="center",fontsize=7)
    #             axs.set_aspect('equal', 'box')
    #             axs.set_ylabel("")
    #             axs.set_xlabel("")
    #             axs.annotate(
    #                 f"GraphST(ASW)",  # Title text
    #                 xy=(0.65, 0.99),  # Relative axes coordinates, slightly inside the plot area
    #                 xycoords="axes fraction",  # Use axes-fraction coordinates (0-1 range)
    #                 ha="right",  # Horizontal alignment
    #                 va="top",  # Vertical alignment: bottom
    #                 fontsize=9,  # Font size
    #             )
    #             axs.axes.invert_yaxis()
    # 
    #         if j % 2 == 1:
    #             domains = "louvain_ref"
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             adata.obs[domains] = adata.obs[domains].cat.codes + 1
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             # adata.uns[domains + "_colors"] = list(plot_color[:num_celltype])
    #             axs = ax5
    #             sc.pl.scatter(adata, alpha=1, x="y_pixel", y="x_pixel", color=domains, title="",
    #                                palette=plot_color,ax=axs, show=False, size=50000 / adata.shape[0]* 1.5, legend_fontsize='x-small')
    #             axs.set_xticklabels(axs.get_xticklabels(), fontsize=7)
    #             axs.set_yticklabels(axs.get_yticklabels(), rotation=90,ha="center",va="center",fontsize=7)
    #             axs.set_aspect('equal', 'box')
    #             axs.set_ylabel("")
    #             axs.annotate(
    #                 f"GraphST(Entropy)",  # Title text
    #                 xy=(0.65, 0.99),  # Relative axes coordinates, slightly inside the plot area
    #                 xycoords="axes fraction",  # Use axes-fraction coordinates (0-1 range)
    #                 ha="right",  # Horizontal alignment
    #                 va="top",  # Vertical alignment: bottom
    #                 fontsize=9,  # Font size
    #             )
    #             axs.axes.invert_yaxis()
    # 
    #     for j in range(4,6):
    #         base_dir, _, adata, prefix, _ = initialization_STAGATE_MERFISH(index)
    #         x_pixel = adata.obsm["spatial"][:, 1].tolist()
    #         y_pixel = adata.obsm["spatial"][:, 0].tolist()
    #         adj_2d = spg.calculate_adj_matrix(x=x_pixel, y=y_pixel, histology=False)
    #         adata, if_ref = refine_a_res(res_list[j], adata, base_dir, prefix, adj_2d, if_Spa=False)
    #         if if_ref == False:
    #             print("Something is wrong")
    # 
    #         if j % 2 == 0:
    #             domains = "louvain"
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             adata.obs[domains] = adata.obs[domains].cat.codes + 1
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             axs = ax3
    #             sc.pl.scatter(adata, alpha=1, x="y_pixel", y="x_pixel", color=domains, title="",
    #                           palette=plot_color, ax=axs, show=False, size=50000 / adata.shape[0]* 1.5, legend_fontsize='x-small')
    #             axs.set_xticklabels(axs.get_xticklabels(), fontsize=7)
    #             axs.set_yticklabels(axs.get_yticklabels(), rotation=90,ha="center",va="center",fontsize=7)
    #             axs.set_ylabel("")
    #             axs.set_xlabel("")
    #             axs.set_aspect('equal', 'box')
    #             axs.annotate(
    #                 f"STAGATE(ASW)",  # Title text
    #                 xy=(0.65, 0.99),  # Relative axes coordinates, slightly inside the plot area
    #                 xycoords="axes fraction",  # Use axes-fraction coordinates (0-1 range)
    #                 ha="right",  # Horizontal alignment
    #                 va="top",  # Vertical alignment: bottom
    #                 fontsize=9,  # Font size
    #             )
    #             axs.axes.invert_yaxis()
    # 
    #         if j % 2 == 1:
    #             domains = "louvain_ref"
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             adata.obs[domains] = adata.obs[domains].cat.codes + 1
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             # adata.uns[domains + "_colors"] = list(plot_color[:num_celltype])
    #             axs = ax6
    #             sc.pl.scatter(adata, alpha=1, x="y_pixel", y="x_pixel", color=domains, title="",
    #                                palette=plot_color, ax=axs, show=False, size=50000 / adata.shape[0]* 1.5, legend_fontsize='x-small')
    #             axs.set_xticklabels(axs.get_xticklabels(), fontsize=7)
    #             axs.set_yticklabels(axs.get_yticklabels(), rotation=90,ha="center",va="center",fontsize=7)
    #             axs.set_ylabel("")
    #             axs.set_aspect('equal', 'box')
    #             axs.annotate(
    #                 f"STAGATE(Entropy)",  # Title text
    #                 xy=(0.65, 0.99),  # Relative axes coordinates, slightly inside the plot area
    #                 xycoords="axes fraction",  # Use axes-fraction coordinates (0-1 range)
    #                 ha="right",  # Horizontal alignment
    #                 va="top",  # Vertical alignment: bottom
    #                 fontsize=9,  # Font size
    #             )
    #             axs.axes.invert_yaxis()
    # 
    #     plt.tight_layout()
    #     plt.savefig(save_dir+index+".png",dpi = 600, bbox_inches='tight',format='png')
    #     plt.close()

    # for i in [28]:
    #     fig = plt.figure(figsize=(27,5.1))
    #     # plt.subplots_adjust(wspace=1)
    #     plt.subplots_adjust(hspace=0.3)
    #     ax0 = plt.subplot2grid((15, 81), (0, 0), rowspan=14, colspan=30)
    #     ax1 = plt.subplot2grid((15, 81), (0, 32), rowspan=7, colspan=15)
    #     ax2 = plt.subplot2grid((15, 81), (0, 49), rowspan=7, colspan=15)
    #     ax3 = plt.subplot2grid((15, 81), (0, 66), rowspan=7, colspan=15)
    #     ax4 = plt.subplot2grid((15, 81), (8, 32), rowspan=7, colspan=15)
    #     ax5 = plt.subplot2grid((15, 81), (8, 49), rowspan=7, colspan=15)
    #     ax6 = plt.subplot2grid((15, 81), (8, 66), rowspan=7, colspan=15)
    #     index = index_list[i]
    #     print("Start: "+index)
    #     res_list = res_df.iloc[i,:]
    #     for j in range(2):
    #         base_dir, _, adata, prefix, _ = initialization_SpaGCN_STARmap(index)
    #         x_pixel = adata.obsm["spatial"][:, 1].tolist()
    #         y_pixel = adata.obsm["spatial"][:, 0].tolist()
    #         adj_2d = spg.calculate_adj_matrix(x=x_pixel, y=y_pixel, histology=False)
    #         adata,if_ref = refine_a_res(res_list[j],adata,base_dir,prefix,adj_2d,if_Spa=True)
    #         if if_ref == 0:
    #             print("Something is wrong")
    #         if j == 0:
    #             # plot the ground truth
    #             domains = "ground_truth"
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             num_celltype = len(adata.obs[domains].unique())
    #             adata.uns[domains + "_colors"] = list(plot_color[:num_celltype])
    #             axs = ax0
    #             sc.pl.scatter(adata, alpha=1, x="y_pixel", y="x_pixel", color=domains, title="", ax=axs,
    #                                show=False, size=50000 / adata.shape[0] * 5)
    #             axs.set_aspect('equal', 'box')
    #             axs.set_yticklabels(axs.get_yticklabels(), rotation=90,ha="center",va="center",fontsize=7)
    #             axs.legend(ncol=7, markerscale=1.0, bbox_to_anchor=[0.5, -0.12], loc='upper center', fontsize='x-small')
    #             axs.legend_.set_title("Domains")
    #             axs.axes.invert_yaxis()
    #
    #         if j%2 == 0:
    #             domains = "louvain"
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             adata.obs[domains] = adata.obs[domains].cat.codes + 1
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             axs = ax1
    #             sc.pl.scatter(adata, alpha=1, x="y_pixel", y="x_pixel", color=domains,title="",
    #                                    palette=plot_color, ax=axs, show=False, size=50000 / adata.shape[0] * 1.5, legend_fontsize='xx-small')
    #             axs.set_xticklabels(axs.get_xticklabels(), fontsize=7)
    #             axs.set_yticklabels(axs.get_yticklabels(), rotation=90,ha="center",va="center",fontsize=7)
    #             axs.set_aspect('equal', 'box')
    #             axs.set_xlabel("")
    #             axs.annotate(
    #                 f"SpaGCN(ASW)",  # Title text
    #                 xy=(0.01, 0.99),  # Relative axes coordinates, slightly inside the plot area
    #                 xycoords="axes fraction",  # Use axes-fraction coordinates (0-1 range)
    #                 ha="left",  # Horizontal alignment
    #                 va="top",  # Vertical alignment: bottom
    #                 fontsize=7,  # Font size
    #                 rotation=-90  # Rotation angle (positive clockwise, negative counterclockwise)
    #             )
    #             axs.axes.invert_yaxis()
    #
    #
    #         if j%2 == 1:
    #             domains = "louvain_ref"
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             adata.obs[domains] = adata.obs[domains].cat.codes + 1
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             # adata.uns[domains + "_colors"] = list(plot_color[:num_celltype])
    #             axs = ax4
    #             sc.pl.scatter(adata, alpha=1, x="y_pixel", y="x_pixel", color=domains,title="",
    #                                palette=plot_color,ax=axs, show=False, size=50000 / adata.shape[0]* 1.5, legend_fontsize='xx-small')
    #             axs.set_xticklabels(axs.get_xticklabels(), fontsize=7)
    #             axs.set_yticklabels(axs.get_yticklabels(), rotation=90,ha="center",va="center",fontsize=7)
    #             axs.set_aspect('equal', 'box')
    #             axs.annotate(
    #                 f"SpaGCN(Entropy)",  # Title text
    #                 xy=(0.01, 0.99),  # Relative axes coordinates, slightly inside the plot area
    #                 xycoords="axes fraction",  # Use axes-fraction coordinates (0-1 range)
    #                 ha="left",  # Horizontal alignment
    #                 va="top",  # Vertical alignment: bottom
    #                 fontsize=7,  # Font size
    #                 rotation=-90  # Rotation angle (positive clockwise, negative counterclockwise)
    #             )
    #             axs.axes.invert_yaxis()
    #
    #     for j in range(2,4):
    #         base_dir, _, adata, prefix, _ = initialization_GraphST_STARmap(index)
    #         x_pixel = adata.obsm["spatial"][:, 1].tolist()
    #         y_pixel = adata.obsm["spatial"][:, 0].tolist()
    #         adj_2d = spg.calculate_adj_matrix(x=x_pixel, y=y_pixel, histology=False)
    #         adata, if_ref = refine_a_res(res_list[j], adata, base_dir, prefix, adj_2d, if_Spa=False)
    #         if if_ref == False:
    #             print("Something is wrong")
    #
    #         if j % 2 == 0:
    #             domains = "louvain"
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             adata.obs[domains] = adata.obs[domains].cat.codes + 1
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             axs = ax2
    #             sc.pl.scatter(adata, alpha=1, x="y_pixel", y="x_pixel", color=domains, title="",
    #                           palette=plot_color, ax=axs, show=False, size=50000 / adata.shape[0]* 1.5, legend_fontsize='xx-small')
    #             axs.set_xticklabels(axs.get_xticklabels(), fontsize=7)
    #             axs.set_yticklabels(axs.get_yticklabels(), rotation=90,ha="center",va="center",fontsize=7)
    #             axs.set_aspect('equal', 'box')
    #             axs.set_ylabel("")
    #             axs.set_xlabel("")
    #             axs.annotate(
    #                 f"GraphST(ASW)",  # Title text
    #                 xy=(0.01, 0.99),  # Relative axes coordinates, slightly inside the plot area
    #                 xycoords="axes fraction",  # Use axes-fraction coordinates (0-1 range)
    #                 ha="left",  # Horizontal alignment
    #                 va="top",  # Vertical alignment: bottom
    #                 fontsize=7,  # Font size
    #                 rotation=-90  # Rotation angle (positive clockwise, negative counterclockwise)
    #             )
    #             axs.axes.invert_yaxis()
    #
    #         if j % 2 == 1:
    #             domains = "louvain_ref"
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             adata.obs[domains] = adata.obs[domains].cat.codes + 1
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             # adata.uns[domains + "_colors"] = list(plot_color[:num_celltype])
    #             axs = ax5
    #             sc.pl.scatter(adata, alpha=1, x="y_pixel", y="x_pixel", color=domains, title="",
    #                                palette=plot_color,ax=axs, show=False, size=50000 / adata.shape[0]* 1.5, legend_fontsize='xx-small')
    #             axs.set_xticklabels(axs.get_xticklabels(), fontsize=7)
    #             axs.set_yticklabels(axs.get_yticklabels(), rotation=90,ha="center",va="center",fontsize=7)
    #             axs.set_aspect('equal', 'box')
    #             axs.set_ylabel("")
    #             axs.annotate(
    #                 f"GraphST(Entropy)",  # Title text
    #                 xy=(0.01, 0.99),  # Relative axes coordinates, slightly inside the plot area
    #                 xycoords="axes fraction",  # Use axes-fraction coordinates (0-1 range)
    #                 ha="left",  # Horizontal alignment
    #                 va="top",  # Vertical alignment: bottom
    #                 fontsize=7,  # Font size
    #                 rotation=-90  # Rotation angle (positive clockwise, negative counterclockwise)
    #             )
    #             axs.axes.invert_yaxis()
    #
    #     for j in range(4,6):
    #         base_dir, _, adata, prefix, _ = initialization_STAGATE_STARmap(index)
    #         x_pixel = adata.obsm["spatial"][:, 1].tolist()
    #         y_pixel = adata.obsm["spatial"][:, 0].tolist()
    #         adj_2d = spg.calculate_adj_matrix(x=x_pixel, y=y_pixel, histology=False)
    #         adata, if_ref = refine_a_res(res_list[j], adata, base_dir, prefix, adj_2d, if_Spa=False)
    #         if if_ref == False:
    #             print("Something is wrong")
    #
    #         if j % 2 == 0:
    #             domains = "louvain"
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             adata.obs[domains] = adata.obs[domains].cat.codes + 1
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             axs = ax3
    #             sc.pl.scatter(adata, alpha=1, x="y_pixel", y="x_pixel", color=domains, title="",
    #                           palette=plot_color, ax=axs, show=False, size=50000 / adata.shape[0]* 1.5, legend_fontsize='xx-small')
    #             axs.set_xticklabels(axs.get_xticklabels(), fontsize=7)
    #             axs.set_yticklabels(axs.get_yticklabels(), rotation=90,ha="center",va="center",fontsize=7)
    #             axs.set_ylabel("")
    #             axs.set_xlabel("")
    #             axs.set_aspect('equal', 'box')
    #             axs.annotate(
    #                 f"STAGATE(ASW)",  # Title text
    #                 xy=(0.01, 0.99),  # Relative axes coordinates, slightly inside the plot area
    #                 xycoords="axes fraction",  # Use axes-fraction coordinates (0-1 range)
    #                 ha="left",  # Horizontal alignment
    #                 va="top",  # Vertical alignment: bottom
    #                 fontsize=7,  # Font size
    #                 rotation=-90  # Rotation angle (positive clockwise, negative counterclockwise)
    #             )
    #             axs.axes.invert_yaxis()
    #
    #         if j % 2 == 1:
    #             domains = "louvain_ref"
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             adata.obs[domains] = adata.obs[domains].cat.codes + 1
    #             adata.obs[domains] = adata.obs[domains].astype("category")
    #             # adata.uns[domains + "_colors"] = list(plot_color[:num_celltype])
    #             axs = ax6
    #             sc.pl.scatter(adata, alpha=1, x="y_pixel", y="x_pixel", color=domains, title="",
    #                                palette=plot_color, ax=axs, show=False, size=50000 / adata.shape[0]* 1.5, legend_fontsize='xx-small')
    #             axs.set_xticklabels(axs.get_xticklabels(), fontsize=7)
    #             axs.set_yticklabels(axs.get_yticklabels(), rotation=90,ha="center",va="center",fontsize=7)
    #             axs.set_ylabel("")
    #             axs.set_aspect('equal', 'box')
    #             axs.annotate(
    #                 f"STAGATE(Entropy)",  # Title text
    #                 xy=(0.01, 0.99),  # Relative axes coordinates, slightly inside the plot area
    #                 xycoords="axes fraction",  # Use axes-fraction coordinates (0-1 range)
    #                 ha="left",  # Horizontal alignment
    #                 va="top",  # Vertical alignment: bottom
    #                 fontsize=6,  # Font size
    #                 rotation=-90  # Rotation angle (positive clockwise, negative counterclockwise)
    #             )
    #             axs.axes.invert_yaxis()
    #
    #
    #     plt.tight_layout()
    #     plt.savefig(save_dir+index+".png",dpi = 600, bbox_inches='tight',format='png')
    #     plt.close()
