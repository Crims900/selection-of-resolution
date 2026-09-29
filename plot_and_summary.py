import pandas as pd
import os
import scanpy as sc
import SpaGCN_1000_random_seeds as spg
import csv
from sklearn.metrics import adjusted_rand_score
from collections import Counter
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import anndata
import time

PROJECT_ROOT = "/Data/Programs/SpaGCN_stabilization/stabilization_part/"
DATASET_ROOT = "/Data/Datasets/"

#TODO:BARISTAseq

# base_dir = PROJECT_ROOT + "Slice_1" + "/first100/"
# dataset_dir = DATASET_ROOT + "BARISTAseq/mouse_primary_visual_cortex/"
# adata = sc.read_h5ad(dataset_dir+"Slice_1.h5ad")
# x_pixel = adata.obsm["spatial"][:,1].tolist()
# y_pixel = adata.obsm["spatial"][:,0].tolist()
#
# adj_2d=spg.calculate_adj_matrix(x=x_pixel,y=y_pixel, histology=False)
# #
# #
# if os.path.exists(base_dir + "louvain_results/") == False:
#     os.mkdir(base_dir + "louvain_results/")
#
# with open(base_dir + "louvain_results/" + 'asw100_fusion_BARISTAseq_summary_pre.csv', 'a+', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(["file", "ari_pre", "ari_ref","cluster_number","cluster1","cluster2","cluster3","cluster4","cluster5","cluster6","cluster7","cluster8","cluster9",
#                      "cluster10","cluster11","cluster12","cluster13","cluster14","cluster15","cluster16","cluster17","cluster18","cluster19","cluster20"])
# with open(base_dir + "louvain_results/" + 'asw100_fusion_BARISTAseq_summary_ref.csv', 'a+', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(["file", "ari_pre", "ari_ref","cluster_number","cluster1","cluster2","cluster3","cluster4","cluster5","cluster6","cluster7","cluster8","cluster9",
#                      "cluster10","cluster11","cluster12","cluster13","cluster14","cluster15","cluster16","cluster17","cluster18","cluster19","cluster20"])
#
#
# for file in os.listdir(base_dir):
#     if file.endswith('.csv'):
#         result = pd.read_csv(base_dir+file,index_col=0)
#         adata.obs["pred"] = result['cluster'].tolist()
#         adata.obs["pred"] = adata.obs["pred"].astype('category')
#
#         refined_pred = spg.refine(sample_id=adata.obs.index.tolist(), pred=adata.obs["pred"].tolist(), dis=adj_2d,
#                                   shape="hexagon")
#         adata.obs["refined_pred"] = refined_pred
#         adata.obs["refined_pred"] = adata.obs["refined_pred"].astype('category')
#
#         domains = "pred"
#         label = adata.obs[domains].to_list()
#         truth = adata.obs['ground_truth'].to_list()
#         ari_pre = adjusted_rand_score(label, truth)
#         domains = "refined_pred"
#         label = adata.obs[domains].to_list()
#         truth = adata.obs['ground_truth'].to_list()
#         ari_ref = adjusted_rand_score(label, truth)
#
#
#
#         with open(base_dir + "louvain_results/" + 'asw100_fusion_BARISTAseq_summary_pre.csv', 'a+', newline='') as csvfile:
#             writer = csv.writer(csvfile)
#             writer.writerow([file, ari_pre, ari_ref, result['cluster'].nunique()]+sorted(list(Counter(result['cluster']).values()),reverse=True))
#
#         with open(base_dir + "louvain_results/" + 'asw100_fusion_BARISTAseq_summary_ref.csv', 'a+', newline='') as csvfile:
#             writer = csv.writer(csvfile)
#             writer.writerow([file, ari_pre, ari_ref, adata.obs["refined_pred"].nunique()]+sorted(list(Counter(result['cluster']).values()),reverse=True))

# Calculate the entropy contribution of each cluster.
# with open(base_dir  + 'asw100_fusion_BARISTAseq_summary_detailed.csv', 'a+',newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(["file", "ari_pre", "ari_ref", "cluster_number", "entropy", "cluster1", "cluster2", "cluster3", "cluster4","cluster5","cluster6", "cluster7", "cluster8", "cluster9","cluster10", "cluster11", "cluster12", "cluster13", "cluster14", "cluster15", "cluster16", "cluster17","cluster18", "cluster19", "cluster20"])
#
# for res in np.linspace(1.0, 1.2, 201):
#     res = round(res, 3)
#     result = pd.read_csv(base_dir + "asw100_fusion_Euclid_median_res"+str(res)+".csv", index_col=0)
#
#     adata.obs["pred"] = result['cluster'].tolist()
#     adata.obs["pred"] = adata.obs["pred"].astype('category')
#
#     refined_pred = spg.refine(sample_id=adata.obs.index.tolist(), pred=adata.obs["pred"].tolist(), dis=adj_2d,
#                               shape="hexagon")
#     adata.obs["refined_pred"] = refined_pred
#     adata.obs["refined_pred"] = adata.obs["refined_pred"].astype('category')
#
#     domains = "pred"
#     label = adata.obs[domains].to_list()
#     truth = adata.obs['ground_truth'].to_list()
#     ari_pre = adjusted_rand_score(label, truth)
#     domains = "refined_pred"
#     label = adata.obs[domains].to_list()
#     truth = adata.obs['ground_truth'].to_list()
#     ari_ref = adjusted_rand_score(label, truth)
#
#     cluster_num = sorted(list(Counter(adata.obs["refined_pred"]).values()), reverse=True)
#     cluster_num = np.array(cluster_num)
#     cluster_num = cluster_num / cluster_num.sum()
#     entropy_list = (-1) * np.log(cluster_num) * cluster_num
#     entropy_total = entropy_list.sum()
#     entropy_percent = entropy_list / entropy_total
#     entropy_cumsum_percent = entropy_percent.cumsum()
#
#     with open(base_dir  + 'asw100_fusion_BARISTAseq_summary_detailed.csv', 'a+',
#               newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow([res, ari_pre, ari_ref, adata.obs["refined_pred"].nunique(), entropy_total] + sorted(
#             list(Counter(adata.obs["refined_pred"]).values()), reverse=True))
#         writer.writerow([""] + [""] + [""] + [""] + [""] + [
#             str(round(entropy_list[i], 4)) + " " + "{:.2%}".format(entropy_percent[i]) + " " + "{:.2%}".format(
#                 entropy_cumsum_percent[i]) for i in range(len(entropy_list))])
#     print("res is ",res," finished!")

# Generate plots of resolution, cluster number, and entropy.
# res_list = []
# entropy_list = []
# number_list = []
# for res in np.linspace(1.0, 1.2, 201):
#     res = round(res, 3)
#     result = pd.read_csv(base_dir + "asw100_fusion_Euclid_median_res"+str(res)+".csv", index_col=0)
#     adata.obs["pred"] = result['cluster'].tolist()
#     adata.obs["pred"] = adata.obs["pred"].astype('category')
#
#     refined_pred = spg.refine(sample_id=adata.obs.index.tolist(), pred=adata.obs["pred"].tolist(), dis=adj_2d,
#                               shape="hexagon")
#     adata.obs["refined_pred"] = refined_pred
#     adata.obs["refined_pred"] = adata.obs["refined_pred"].astype('category')
#     list_tmp = list(Counter(result['cluster']).values())
#     np_tmp = np.array(list_tmp)
#     np_tmp = np_tmp/np_tmp.sum()
#     # res_list.append(float(file.split("res")[1].split(".csv")[0]))
#     res_list.append(res)
#     entropy_tmp = entropy(np_tmp)
#     entropy_list.append(entropy_tmp)
#     number_list.append(result['cluster'].nunique())
#     print("res is ",res," finished!")

# label = np.zeros_like(entropy_list)
# label[[0,1,10]]=1
# plt.figure(dpi=600,figsize=(8,6))
# plt.subplot(221)
# plt.scatter(res_list,entropy_list,s=5,alpha=0.5)
# plt.title("res vs entropy")
# plt.subplot(222)
# plt.scatter(res_list,number_list,s=5,alpha=0.5)
# plt.title("res vs number")
# plt.subplot(223)
# plt.scatter(entropy_list,number_list,s=5,alpha=0.5)
# plt.title("entropy vs number")
# plt.savefig(base_dir + "louvain_results/plot_new.png")
#
# entropy_diff_list = [entropy_list[1:][i] - entropy_list[:-1][i] for i in range(len(entropy_list)-1)]
# res_diff_list = [res_list[1:][i] - res_list[:-1][i] for i in range(len(res_list)-1)]
# k_list = [entropy_diff_list[i]/res_diff_list[i] for i in range(len(res_diff_list))]
# plt.figure(dpi=600,figsize=(8,6))
# plt.scatter(res_list[1:],k_list,s=5,alpha=0.5)
# plt.title("entropy/res - res")
# plt.savefig(base_dir + "louvain_results/k_new.png")
# save = pd.DataFrame({"res":res_list[1:],"k":k_list})
# save.to_csv(base_dir+"louvain_results/k_new.csv")

# df = pd.read_csv(base_dir+"louvain_results/k_new.csv",index_col=0)
# df = df.loc[df.k.sort_values(ascending=False)[:10].index]
# print(df)

#TODO:151673

# base_dir = PROJECT_ROOT + "151673" + "/first100/"
# dataset_dir = DATASET_ROOT + "SpatialLIBD/151673/"
# adata = sc.read_visium(dataset_dir)
# adata.obs["x_array"] = adata.obs["array_row"]
# adata.obs["y_array"] = adata.obs["array_col"]
# adata.obs["x_pixel"] = adata.obsm['spatial'][:, 1]
# adata.obs["y_pixel"] = adata.obsm['spatial'][:, 0]
# adata.obs["x_pixel"] = np.array(adata.obs["x_pixel"]).round().astype("int")
# adata.obs["y_pixel"] = np.array(adata.obs["y_pixel"]).round().astype("int")
#
# truth_dir = dataset_dir + "Annotation_train_test_split.csv"
# df = pd.read_csv(truth_dir, index_col=0)
# df = df.reindex(adata.obs.index)
# adata.obs["ground"] = df["Cluster"].to_list()
# adata.obs["ground"] = adata.obs["ground"].astype('category')
#
# x_pixel = adata.obs["x_pixel"].tolist()
# y_pixel = adata.obs["y_pixel"].tolist()
# x_array = adata.obs["x_array"].tolist()
# y_array = adata.obs["y_array"].tolist()
#
#
# adj_2d=spg.calculate_adj_matrix(x=x_array,y=y_array, histology=False)
#
# if os.path.exists(base_dir + "louvain_results/") == False:
#     os.mkdir(base_dir + "louvain_results/")
#
# with open(base_dir + "louvain_results/" + 'asw100_fusion_151673_summary_pre.csv', 'a+', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(["file", "ari_pre", "ari_ref","cluster_number","cluster1","cluster2","cluster3","cluster4","cluster5","cluster6","cluster7","cluster8","cluster9",
#                      "cluster10","cluster11","cluster12","cluster13","cluster14","cluster15","cluster16","cluster17","cluster18","cluster19","cluster20"])
#
# with open(base_dir + "louvain_results/" + 'asw100_fusion_151673_summary_ref.csv', 'a+', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(["file", "ari_pre", "ari_ref","cluster_number","cluster1","cluster2","cluster3","cluster4","cluster5","cluster6","cluster7","cluster8","cluster9",
#                      "cluster10","cluster11","cluster12","cluster13","cluster14","cluster15","cluster16","cluster17","cluster18","cluster19","cluster20"])
#
# for file in os.listdir(base_dir):
#     if file.endswith('.csv'):
#         result = pd.read_csv(base_dir+file,index_col=0)
#         adata.obs["pred"] = result['cluster'].tolist()
#         adata.obs["pred"] = adata.obs["pred"].astype('category')
#
#         refined_pred = spg.refine(sample_id=adata.obs.index.tolist(), pred=adata.obs["pred"].tolist(), dis=adj_2d,
#                                   shape="hexagon")
#         adata.obs["refined_pred"] = refined_pred
#         adata.obs["refined_pred"] = adata.obs["refined_pred"].astype('category')
#
#         domains = "pred"
#         label = adata.obs[domains].to_list()
#         truth = adata.obs['ground'].to_list()
#         ari_pre = adjusted_rand_score(label, truth)
#         domains = "refined_pred"
#         label = adata.obs[domains].to_list()
#         truth = adata.obs['ground'].to_list()
#         ari_ref = adjusted_rand_score(label, truth)
#
#         with open(base_dir + "louvain_results/" + 'asw100_fusion_151673_summary_pre.csv', 'a+',
#                   newline='') as csvfile:
#             writer = csv.writer(csvfile)
#             writer.writerow([file, ari_pre, ari_ref, result['cluster'].nunique()] + sorted(
#                 list(Counter(result['cluster']).values()), reverse=True))
#
#         with open(base_dir + "louvain_results/" + 'asw100_fusion_151673_summary_ref.csv', 'a+',
#                   newline='') as csvfile:
#             writer = csv.writer(csvfile)
#             writer.writerow([file, ari_pre, ari_ref, adata.obs["refined_pred"].nunique()] + sorted(
#                 list(Counter(result['cluster']).values()), reverse=True))

# Calculate the entropy contribution of each cluster.
# with open(base_dir + "louvain_results/" + 'asw100_fusion_151673_summary_ref_with_detailed_entropy.csv', 'a+',
#           newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(
#         ["file", "ari_pre", "ari_ref", "cluster_number", "entropy", "cluster1", "cluster2", "cluster3", "cluster4",
#          "cluster5",
#          "cluster6", "cluster7", "cluster8", "cluster9",
#          "cluster10", "cluster11", "cluster12", "cluster13", "cluster14", "cluster15", "cluster16", "cluster17",
#          "cluster18", "cluster19", "cluster20"])
#
# for res in np.linspace(1.0, 1.2, 201):
#     res = round(res, 3)
#     result = pd.read_csv(base_dir + "asw100_fusion_Euclid_median_res"+str(res)+".csv", index_col=0)
#     adata.obs["pred"] = result['cluster'].tolist()
#     adata.obs["pred"] = adata.obs["pred"].astype('category')
#
#     refined_pred = spg.refine(sample_id=adata.obs.index.tolist(), pred=adata.obs["pred"].tolist(), dis=adj_2d,
#                               shape="hexagon")
#     adata.obs["refined_pred"] = refined_pred
#     adata.obs["refined_pred"] = adata.obs["refined_pred"].astype('category')
#
#     domains = "pred"
#     label = adata.obs[domains].to_list()
#     truth = adata.obs['ground'].to_list()
#     ari_pre = adjusted_rand_score(label, truth)
#     domains = "refined_pred"
#     label = adata.obs[domains].to_list()
#     truth = adata.obs['ground'].to_list()
#     ari_ref = adjusted_rand_score(label, truth)
#
#     cluster_num = sorted(list(Counter(adata.obs["refined_pred"]).values()), reverse=True)
#     cluster_num = np.array(cluster_num)
#     cluster_num = cluster_num / cluster_num.sum()
#     entropy_list = (-1) * np.log(cluster_num) * cluster_num
#     entropy = entropy_list.sum()
#     entropy_percent = entropy_list / entropy
#     entropy_cumsum_percent = entropy_percent.cumsum()
#
#     with open(base_dir + "louvain_results/" + 'asw100_fusion_151673_summary_ref_with_detailed_entropy.csv', 'a+',
#               newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow([res, ari_pre, ari_ref, adata.obs["refined_pred"].nunique(), entropy] + sorted(
#             list(Counter(adata.obs["refined_pred"]).values()), reverse=True))
#         writer.writerow([""] + [""] + [""] + [""] + [""] + [
#             str(round(entropy_list[i], 4)) + " " + "{:.2%}".format(entropy_percent[i]) + " " + "{:.2%}".format(
#                 entropy_cumsum_percent[i]) for i in range(len(entropy_list))])
#     print("res is ",res," finished!")

# Generate plots of resolution, cluster number, and entropy.
# res_list = []
# entropy_list = []
# number_list = []
# for file in os.listdir(base_dir):
#     if file.endswith('.csv') & ("_Euclid_median" in file):
# for res in np.linspace(1.0, 1.2, 201):
#     res = round(res, 3)
#     result = pd.read_csv(base_dir + "asw100_fusion_Euclid_median_res"+str(res)+".csv", index_col=0)
#     adata.obs["pred"] = result['cluster'].tolist()
#     adata.obs["pred"] = adata.obs["pred"].astype('category')
#
#     refined_pred = spg.refine(sample_id=adata.obs.index.tolist(), pred=adata.obs["pred"].tolist(), dis=adj_2d,
#                               shape="hexagon")
#     adata.obs["refined_pred"] = refined_pred
#     adata.obs["refined_pred"] = adata.obs["refined_pred"].astype('category')
#     list_tmp = list(Counter(result['cluster']).values())
#     np_tmp = np.array(list_tmp)
#     np_tmp = np_tmp/np_tmp.sum()
#     # res_list.append(float(file.split("res")[1].split(".csv")[0]))
#     res_list.append(res)
#     entropy_tmp = entropy(np_tmp)
#     entropy_list.append(entropy_tmp)
#     number_list.append(result['cluster'].nunique())
#     print("res is ",res," finished!")
# #
# label = ["blue" for n in range(len(entropy_list))]
# label[0]="red"
# label[1]="red"
# label[2]="red"
# label[15]="red"
# plt.figure(dpi=600,figsize=(8,6))
# plt.subplot(221)
# plt.scatter(res_list,entropy_list,s=5,alpha=0.5)
# plt.title("res vs entropy")
# plt.subplot(222)
# plt.scatter(res_list,number_list,s=5,alpha=0.5)
# plt.title("res vs number")
# plt.subplot(223)
# plt.scatter(entropy_list,number_list,s=5,alpha=0.5)
# plt.title("entropy vs number")
# plt.savefig(base_dir + "louvain_results/plot_new.png")
# #
# entropy_diff_list = [entropy_list[1:][i] - entropy_list[:-1][i] for i in range(len(entropy_list)-1)]
# res_diff_list = [res_list[1:][i] - res_list[:-1][i] for i in range(len(res_list)-1)]
# k_list = [entropy_diff_list[i]/res_diff_list[i] for i in range(len(res_diff_list))]
# plt.figure(dpi=600,figsize=(8,6))
# plt.scatter(res_list[1:],k_list,s=5,alpha=0.5)
# plt.title("entropy/res - res")
# plt.savefig(base_dir + "louvain_results/k-151673-new.png")
#
# save = pd.DataFrame({"res":res_list[1:],"k":k_list})
# save.to_csv(base_dir+"louvain_results/k_151673_new.csv")


# Calculate mutual information.

# res_list=[]
# for file in os.listdir(base_dir):
#     if file.endswith('.csv') & ("_Euclid_median" in file):
#         res_list.append(file.split(".csv")[0].split("res")[1])
# res_list = sorted(res_list)
# with open(base_dir+"louvain_results/"+"151673_ARI_info.csv",'a+', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(["res","ARI_info"])
# with open(base_dir+"louvain_results/"+"151673_mutual_info.csv",'a+', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(["res","mutual_info"])
# with open(base_dir+"louvain_results/"+"151673_normalized_mutual_info.csv",'a+', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(["res","normalized_mutual_info"])
#
#
# for i in range(len(res_list) - 1):
#     df1 = pd.read_csv(base_dir + "asw100_fusion_Euclid_median_res" + res_list[i] + ".csv", index_col=0).cluster.values
#     df2 = pd.read_csv(base_dir + "asw100_fusion_Euclid_median_res" + res_list[i + 1] + ".csv", index_col=0).cluster.values
#     mi1 = metrics.adjusted_rand_score(df1, df2)
#     mi2 = metrics.normalized_mutual_info_score(df1,df2)
#     mi3 = metrics.mutual_info_score(df1,df2)
#     with open(base_dir+"louvain_results/"+"151673_ARI_info.csv",'a+', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow([res_list[i], mi1])
#     with open(base_dir+"louvain_results/"+"151673_normalized_mutual_info.csv",'a+', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow([res_list[i], mi2])
#     with open(base_dir+"louvain_results/"+"151673_mutual_info.csv",'a+', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow([res_list[i], mi3])

# Plot mutual information.
# df = pd.read_csv(base_dir+"louvain_results/"+"151673_normalized_mutual_info.csv")
# res_list = df.res.values
# res_list = [float(i) for i in res_list]
# mi_list = df.normalized_mutual_info.values
# plt.scatter(res_list,mi_list,s=5,alpha=0.5)
# plt.title("res and normalized_mutual_info")
# plt.savefig(base_dir+"louvain_results/normalized_mutual_info.png",dpi=600)
# plt.close()
# plt.scatter(res_list[:-5],mi_list[:-5],s=5,alpha=0.5)
# plt.title("res and normalized_mutual_info_modified")
# plt.savefig(base_dir+"louvain_results/normalized_mutual_info_modified.png",dpi=600)
# plt.close()
#
#
# df = pd.read_csv(base_dir+"louvain_results/"+"151673_mutual_info.csv")
# res_list = df.res.values
# res_list = [float(i) for i in res_list]
# mi_list = df.mutual_info.values
# plt.scatter(res_list,mi_list,s=5,alpha=0.5)
# plt.title("res and mutual_info")
# plt.savefig(base_dir+"louvain_results/mutual_info.png",dpi=600)
# plt.close()
# plt.scatter(res_list[:-5],mi_list[:-5],s=5,alpha=0.5)
# plt.title("res and mutual_info_modified")
# plt.savefig(base_dir+"louvain_results/mutual_info_modified.png",dpi=600)
# plt.close()
#
#
# df = pd.read_csv(base_dir+"louvain_results/"+"151673_ARI_info.csv")
# res_list = df.res.values
# res_list = [float(i) for i in res_list]
# mi_list = df.ARI_info.values
# plt.scatter(res_list,mi_list,s=5,alpha=0.5)
# plt.title("res and ARI_info")
# plt.savefig(base_dir+"louvain_results/ARI_info.png",dpi=600)
# plt.close()
# plt.scatter(res_list[:-5],mi_list[:-5],s=5,alpha=0.5)
# plt.title("res and ARI_info_modified")
# plt.savefig(base_dir+"louvain_results/ARI_info_modified.png",dpi=600)
# plt.close()



#TODO:SpatialLIBD

# Dataset
# for index in ["151674","151675","151676"]:
#
#     base_dir = PROJECT_ROOT + index + "/first100/"
#     dataset_dir = DATASET_ROOT + "SpatialLIBD/" + index + "/"
#     adata = sc.read_h5ad(dataset_dir+"processed_adata.h5ad")
#
#     adata.obs["x_array"] = adata.obs["array_row"]
#     adata.obs["y_array"] = adata.obs["array_col"]
#     x_array = adata.obs["x_array"].tolist()
#     y_array = adata.obs["y_array"].tolist()
#     adj_2d=spg.calculate_adj_matrix(x=x_array,y=y_array, histology=False)
#
#     # Calculate the entropy contribution of each cluster.
#     with open(base_dir  + 'asw100_fusion_'+index+'_summary_detailed.csv', 'a+', newline='') as csvfile:
#             writer = csv.writer(csvfile)
#             writer.writerow(
#                 ["file", "ari_pre", "ari_ref", "cluster_number", "entropy","cluster1", "cluster2", "cluster3", "cluster4", "cluster5",
#                  "cluster6", "cluster7", "cluster8", "cluster9",
#                  "cluster10", "cluster11", "cluster12", "cluster13", "cluster14", "cluster15", "cluster16", "cluster17",
#                  "cluster18", "cluster19", "cluster20"])
#
#     res_list_save = []
#     entropy_list_save = []
#     number_list_save = []
#
#     for file in os.listdir(base_dir+"louvain_results/"):
#         if file.endswith('.csv') and ("_Euclid_median" in file):
#             time1 = time.time()
#             res = file.split("res")[1].split(".csv")[0]
#             result = pd.read_csv(base_dir+"louvain_results/"+file,index_col=0)
#             adata.obs["pred"] = result['cluster'].tolist()
#             adata.obs["pred"] = adata.obs["pred"].astype('category')
#
#             refined_pred = spg.refine(sample_id=adata.obs.index.tolist(), pred=adata.obs["pred"].tolist(), dis=adj_2d,
#                                       shape="hexagon")
#             adata.obs["refined_pred"] = refined_pred
#             adata.obs["refined_pred"] = adata.obs["refined_pred"].astype('category')
#
#             domains = "pred"
#             label = adata.obs[domains].to_list()
#             truth = adata.obs['ground_truth'].to_list()
#             ari_pre = adjusted_rand_score(label, truth)
#             domains = "refined_pred"
#             label = adata.obs[domains].to_list()
#             truth = adata.obs['ground_truth'].to_list()
#             ari_ref = adjusted_rand_score(label, truth)
#
#             cluster_num = sorted(list(Counter(adata.obs["refined_pred"]).values()),reverse=True)
#             cluster_num = np.array(cluster_num)
#             cluster_num = cluster_num/cluster_num.sum()
#             entropy_list = (-1)*np.log(cluster_num)*cluster_num
#             entropy = entropy_list.sum()
#             entropy_percent = entropy_list/entropy
#             entropy_cumsum_percent = entropy_percent.cumsum()
#
#             entropy_list_save.append(entropy)
#             number_list_save.append(result['cluster'].nunique())
#             res_list_save.append(float(res))
#
#
#             with open(base_dir  + 'asw100_fusion_'+index+'_summary_detailed.csv', 'a+', newline='') as csvfile:
#                 writer = csv.writer(csvfile)
#                 writer.writerow([res, ari_pre, ari_ref, adata.obs["refined_pred"].nunique(),entropy]+sorted(list(Counter(adata.obs["refined_pred"]).values()),reverse=True))
#                 writer.writerow([""]+[""]+[""]+[""]+[""]+[str(round(entropy_list[i],4))+" "+"{:.2%}".format(entropy_percent[i])+" "+"{:.2%}".format(entropy_cumsum_percent[i]) for i in range(len(entropy_list))])
#             time2 = time.time()
#             print("res {} is finished".format(file)+"using time :{}".format(time2-time1))
#
#     # Generate plots of resolution, cluster number, and entropy.
#     # save = pd.read_csv(base_dir+"res_entropy_number.csv",index_col=0)
#     # res_list_save = save.res.values
#     # entropy_list_save = save.entropy.values
#     # number_list_save = save.number.values
#
#     fig, ax = plt.subplots(3, 3, figsize=(10,10), dpi=600)
#     fig.tight_layout(h_pad=2)
#
#     ax[0, 0].scatter(res_list_save, entropy_list_save, s=5, alpha=0.5)
#     ax[0, 0].set_title(index+"_  res vs entropy", fontsize=8)
#     ax[0, 1].scatter(res_list_save,number_list_save, s=5, alpha=0.5)
#     ax[0, 1].set_title(index+"_  res vs number",fontsize=8)
#     ax[0, 2].scatter(entropy_list_save, number_list_save, s=5, alpha=0.5)
#     ax[0, 2].set_title(index+"_  entropy vs number", fontsize=8)
#
#     ax[1, 0].scatter(res_list_save[0:100], entropy_list_save[0:100], s=5, alpha=0.5)
#     ax[1, 0].set_title(index+"_  res vs entropy", fontsize=8)
#     ax[1, 1].scatter(res_list_save[0:100],number_list_save[0:100], s=5, alpha=0.5)
#     ax[1, 1].set_title(index+"_  res vs number",fontsize=8)
#     ax[1, 2].scatter(entropy_list_save[0:100], number_list_save[0:100], s=5, alpha=0.5)
#     ax[1, 2].set_title(index+"_  entropy vs number", fontsize=8)
#
#     ax[2, 0].scatter(res_list_save[50:100], entropy_list_save[50:100], s=5, alpha=0.5)
#     ax[2, 0].set_title(index+"_  res vs entropy", fontsize=8)
#     ax[2, 1].scatter(res_list_save[50:100],number_list_save[50:100], s=5, alpha=0.5)
#     ax[2, 1].set_title(index+"_  res vs number",fontsize=8)
#     ax[2, 2].scatter(entropy_list_save[50:100], number_list_save[50:100], s=5, alpha=0.5)
#     ax[2, 2].set_title(index+"_  entropy vs number", fontsize=8)
#
#     plt.savefig(base_dir + index+"_res_entropy_number.png")
#
#     save = pd.DataFrame({"res":res_list_save,"entropy":entropy_list_save,"number":number_list_save})
#     save.to_csv(base_dir+"res_entropy_number.csv")

# entropy_diff_list = [entropy_list[1:][i] - entropy_list[:-1][i] for i in range(len(entropy_list)-1)]
# res_diff_list = [res_list[1:][i] - res_list[:-1][i] for i in range(len(res_list)-1)]
# k_list = [entropy_diff_list[i]/res_diff_list[i] for i in range(len(res_diff_list))]
# save = pd.DataFrame({"res":res_list[1:],"k":k_list})
# save.to_csv(base_dir+"louvain_results/k_151507_new.csv")
# plt.figure(dpi=600,figsize=(8,6))
# plt.scatter(res_list[1:],k_list,s=5,alpha=0.5,)
# plt.title("entropy/res - res")
# plt.savefig(base_dir + "louvain_results/k_151507_new.png")



#TODO:Stereoseq

# Stereo-seq dataset


# Dataset
for index in ["E9.5_E1S1.MOSTA",'E9.5_E2S1.MOSTA','E9.5_E2S2.MOSTA','E9.5_E2S3.MOSTA','E9.5_E2S4.MOSTA']:
    base_dir = PROJECT_ROOT + index + "/first100/"
    dataset_dir = DATASET_ROOT + "Stereo_seq/"
    adata = sc.read_h5ad(dataset_dir + index + ".h5ad")
    # adata = sc.read_visium(dataset_dir)
    adata.obs["x_pixel"] = adata.obsm['spatial'][:, 1]
    adata.obs["y_pixel"] = adata.obsm['spatial'][:, 0]
    adata.obs["x_pixel"] = np.array(adata.obs["x_pixel"]).round().astype("int")
    adata.obs["y_pixel"] = np.array(adata.obs["y_pixel"]).round().astype("int")
    adata.obs["ground_truth"] = adata.obs["annotation"]

    x_pixel = adata.obs["x_pixel"].tolist()
    y_pixel = adata.obs["y_pixel"].tolist()

    adata.var_names_make_unique()
    spg.prefilter_genes(adata, min_cells=3)  # avoiding all genes are zeros
    spg.prefilter_specialgenes(adata)
    # Normalize and take log for UMI
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)

    print("data preprocess completed")

    adj = spg.calculate_adj_matrix(x=x_pixel, y=y_pixel, histology=False)
    p = 0.5
    l = spg.search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100)
    adj_2d = adj
    print("start 1")



    # Calculate the entropy contribution of each cluster.
    with open(base_dir  + 'asw100_fusion_'+index+'_summary_detailed.csv', 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                ["file", "ari_pre", "ari_ref", "cluster_number", "entropy","cluster1", "cluster2", "cluster3", "cluster4", "cluster5",
                 "cluster6", "cluster7", "cluster8", "cluster9",
                 "cluster10", "cluster11", "cluster12", "cluster13", "cluster14", "cluster15", "cluster16", "cluster17",
                 "cluster18", "cluster19", "cluster20"])

    res_list_save = []
    entropy_list_save = []
    number_list_save = []
    print("start 2")

    for file in os.listdir(base_dir+"louvain_results/"):
        if file.endswith('.csv') and ("_Euclid_median" in file):
            time1 = time.time()
            res = file.split("res")[1].split(".csv")[0]
            result = pd.read_csv(base_dir+"louvain_results/"+file,index_col=0)
            adata.obs["pred"] = result['cluster'].tolist()
            adata.obs["pred"] = adata.obs["pred"].astype('category')

            refined_pred = spg.refine(sample_id=adata.obs.index.tolist(), pred=adata.obs["pred"].tolist(), dis=adj_2d,
                                      shape="hexagon")
            adata.obs["refined_pred"] = refined_pred
            adata.obs["refined_pred"] = adata.obs["refined_pred"].astype('category')

            domains = "pred"
            label = adata.obs[domains].to_list()
            truth = adata.obs['ground_truth'].to_list()
            ari_pre = adjusted_rand_score(label, truth)
            domains = "refined_pred"
            label = adata.obs[domains].to_list()
            truth = adata.obs['ground_truth'].to_list()
            ari_ref = adjusted_rand_score(label, truth)

            cluster_num = sorted(list(Counter(adata.obs["refined_pred"]).values()),reverse=True)
            cluster_num = np.array(cluster_num)
            cluster_num = cluster_num/cluster_num.sum()
            entropy_list = (-1)*np.log(cluster_num)*cluster_num
            entropy = entropy_list.sum()
            entropy_percent = entropy_list/entropy
            entropy_cumsum_percent = entropy_percent.cumsum()

            entropy_list_save.append(entropy)
            number_list_save.append(result['cluster'].nunique())
            res_list_save.append(float(res))


            with open(base_dir  + 'asw100_fusion_'+index+'_summary_detailed.csv', 'a+', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([res, ari_pre, ari_ref, adata.obs["refined_pred"].nunique(),entropy]+sorted(list(Counter(adata.obs["refined_pred"]).values()),reverse=True))
                writer.writerow([""]+[""]+[""]+[""]+[""]+[str(round(entropy_list[i],4))+" "+"{:.2%}".format(entropy_percent[i])+" "+"{:.2%}".format(entropy_cumsum_percent[i]) for i in range(len(entropy_list))])
            time2 = time.time()
            print("res {} is finished".format(file)+"using time :{}".format(time2-time1))

    pd.DataFrame({"res":res_list_save,"entropy":entropy_list_save,"number":number_list_save}).to_csv(base_dir+"res_entropy_number.csv")

    # Generate plots of resolution, cluster number, and entropy.
    # save = pd.read_csv(base_dir+"res_entropy_number.csv",index_col=0)
    # res_list_save = save.res.values
    # entropy_list_save = save.entropy.values
    # number_list_save = save.number.values

    fig, ax = plt.subplots(3, 3, figsize=(10,10), dpi=600)
    fig.tight_layout(h_pad=2)

    ax[0, 0].scatter(res_list_save, entropy_list_save, s=5, alpha=0.5)
    ax[0, 0].set_title(index+"_  res vs entropy", fontsize=8)
    ax[0, 1].scatter(res_list_save,number_list_save, s=5, alpha=0.5)
    ax[0, 1].set_title(index+"_  res vs number",fontsize=8)
    ax[0, 2].scatter(entropy_list_save, number_list_save, s=5, alpha=0.5)
    ax[0, 2].set_title(index+"_  entropy vs number", fontsize=8)

    ax[1, 0].scatter(res_list_save[0:100], entropy_list_save[0:100], s=5, alpha=0.5)
    ax[1, 0].set_title(index+"_  res vs entropy", fontsize=8)
    ax[1, 1].scatter(res_list_save[0:100],number_list_save[0:100], s=5, alpha=0.5)
    ax[1, 1].set_title(index+"_  res vs number",fontsize=8)
    ax[1, 2].scatter(entropy_list_save[0:100], number_list_save[0:100], s=5, alpha=0.5)
    ax[1, 2].set_title(index+"_  entropy vs number", fontsize=8)

    ax[2, 0].scatter(res_list_save[50:100], entropy_list_save[50:100], s=5, alpha=0.5)
    ax[2, 0].set_title(index+"_  res vs entropy", fontsize=8)
    ax[2, 1].scatter(res_list_save[50:100],number_list_save[50:100], s=5, alpha=0.5)
    ax[2, 1].set_title(index+"_  res vs number",fontsize=8)
    ax[2, 2].scatter(entropy_list_save[50:100], number_list_save[50:100], s=5, alpha=0.5)
    ax[2, 2].set_title(index+"_  entropy vs number", fontsize=8)

    plt.savefig(base_dir + index+"_res_entropy_number.png")

# entropy_diff_list = [entropy_list[1:][i] - entropy_list[:-1][i] for i in range(len(entropy_list)-1)]
# res_diff_list = [res_list[1:][i] - res_list[:-1][i] for i in range(len(res_list)-1)]
# k_list = [entropy_diff_list[i]/res_diff_list[i] for i in range(len(res_diff_list))]
# save = pd.DataFrame({"res":res_list[1:],"k":k_list})
# save.to_csv(base_dir+"louvain_results/k_151507_new.csv")
# plt.figure(dpi=600,figsize=(8,6))
# plt.scatter(res_list[1:],k_list,s=5,alpha=0.5,)
# plt.title("entropy/res - res")
# plt.savefig(base_dir + "louvain_results/k_151507_new.png")
