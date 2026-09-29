"""

End-to-end pipeline that reuses the functions defined in the uploaded
project scripts to select an OPTIMAL LOUVAIN RESOLUTION for one dataset
(example here: SpatialLIBD sample 151673).


    PROJECT_ROOT/151673/
        1000results/                    <- step 1 output (1000 x _prob.txt, _y_pred.txt, _mu.txt)
        first100/
            asw_all.csv                 <- step 2 output
            asw100_fusion_Euclid_median.txt   <- step 4 output (fused distance matrix)
            louvain_results/
                asw100_fusion_Euclid_median_res<r>.csv   <- step 4 output (201 resolutions)
        ari_1000_list.txt               <- step 3 output

"""

import os
import sys
import time
import numpy as np
import pandas as pd
import scanpy as sc

# ---------------------------------------------------------------------------
import SpaGCN_1000_random_seeds as spg_main   # step 1
import select_first_100                          # step 2
import ari_1000_helper as ari100                 # step 3 (see note below)
import fusion_matrix_calculate_and_louvain as fusion   # step 4
import res_selection                             # step 5
import plot_and_summary                          # step 6
import boxplot_stabilization                     # step 7

DATASET_INDEX = "151673"   # SpatialLIBD sample used as the running example


# ===========================================================================
# STEP 1 -- Run SpaGCN with 1000 random seeds
# ===========================================================================
# Functions used (from SpaGCN_1000_random_seeds.py):
#   * initialization(index)  : loads visium data, preprocesses, computes
#                              adjacency (histology), searches l, PCA embed.
#   * multiple_experiments(index, adata, adj, l, result_dir, embed, res=0.7):
#                              trains SpaGCN with 1000 different random seeds
#                              and writes, for run i (i=1..1000):
#                                  1000results/<i>_y_pred.txt
#                                  1000results/<i>_prob.txt   (soft assignment)
#                                  1000results/<i>_mu.txt
def step1_run_1000_seeds():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    (dataset_dir, result_dir, adata, x_pixel, y_pixel,
     adj, l, adj_2d, embed) = spg_main.initialization(DATASET_INDEX)
    spg_main.multiple_experiments(DATASET_INDEX, adata, adj, l,
                                  result_dir, embed, res=0.7)


# ===========================================================================
# STEP 2 -- Rank the 1000 runs with ASW, keep the top 100
# ===========================================================================
# Function used (from select_first_100.py):
#   * t_sne_asw_all(idx, base_dir): loads 1000results/<idx>_prob.txt,
#     wraps it in an AnnData, and computes silhouette_score(prob, y_pred).
#     The driver loop appends rows [idx, asw] to first100/asw_all.csv.
# Later, get_first100_index() (step 4) sorts this file and takes idx[:100].
def step2_compute_asw():
    result_dir, base_dir = select_first_100.initialization_load_adata(DATASET_INDEX)
    for i in range(1, 1001):
        asw = select_first_100.t_sne_asw_all(i, base_dir)
        with open(result_dir + "asw_all.csv", "a+", newline="") as csvfile:
            import csv
            csv.writer(csvfile).writerow([i, asw])


# ===========================================================================
# STEP 3 -- Stability baseline: ARI of each of the 1000 raw SpaGCN runs
# ===========================================================================
# Functions used (from 1000_ari_calculation.py):
#   * initialization_load_adata(index): loads processed_adata.h5ad
#   * calculate_1000_ari(adata, base_dir, result_dir): reads each
#     1000results/<i>_y_pred.txt, compares with ground_truth via
#     adjusted_rand_score, saves ari_1000_list.txt.
def step3_baseline_ari():
    import importlib
    mod = importlib.import_module("1000_ari_calculation")  # module starts with digit -> importlib
    result_dir, base_dir, adata = mod.initialization_load_adata(DATASET_INDEX)
    mod.calculate_1000_ari(adata, base_dir, result_dir)


# ===========================================================================
# STEP 4 -- Consensus fusion + Louvain over a fine resolution grid
# ===========================================================================
# Functions used (from fusion_matrix_calculate_and_louvain.py):
#   * initialization_load_adata(index): sets first100/ and 1000results/ dirs
#   * get_first100_index(result_dir): sorts asw_all.csv desc, returns top-100 run ids
#   * calculate_median_matrix(idx_list, base_dir, result_dir):
#         pairwise Euclidean distance of the probability matrix of each of the
#         top-100 runs -> element-wise median ->
#         first100/asw100_fusion_Euclid_median.txt
#   * main_louvain(index):
#         builds graph G with weight = exp(-distance), then runs
#         community_louvain.best_partition for every resolution in
#         FUSION_RESOLUTIONS = linspace(1.0, 1.2, 201)
#         -> first100/louvain_results/asw100_fusion_Euclid_median_res<r>.csv
def step4_fusion_and_louvain():
    result_dir, base_dir = fusion.initialization_load_adata(DATASET_INDEX)
    idx_list = fusion.get_first100_index(result_dir)
    fusion.calculate_median_matrix(idx_list, base_dir, result_dir)
    fusion.main_louvain(DATASET_INDEX)


# ===========================================================================
# STEP 5 -- Select the optimal resolution  (THE core step)
# ===========================================================================
# Function used (from res_selection.py):
#   * select_resolution(directory) -> (tau_star, n_star)
#
#   Algorithm:
#     1. Extract tau from every CSV filename in louvain_results/.
#     2. For each tau, compute cluster-size entropy
#        H(tau) = sum_k -p_k log p_k ,  p_k = size_k / N.
#     3. k(tau) = (H(tau) - H(tau-1)) / (tau - tau-1): the entropy slope.
#        Keep the top 5% of tau with the largest k  (entropy jumps =
#        resolutions where new, meaningful clusters appear).
#     4. For each candidate tau, compute the "effective cluster count":
#        sort cluster entropy contributions descending, drop the tail whose
#        normalized contributions are < 5%, and require that the remaining
#        tail entropy is >= 5% (else the partition is degenerate).
#     5. tau* = the LARGEST candidate tau that passes the effective-cluster
#        rule; n_star = its effective number of clusters.
def step5_select_resolution():
    louvain_dir = ("/Data/Programs/SpaGCN_stabilization/stabilization_part/"
                   + DATASET_INDEX + "/first100/louvain_results/")
    tau_star, n_star = res_selection.select_resolution(louvain_dir)
    print(f"[151673] optimal resolution tau* = {tau_star}, "
          f"effective clusters N(tau*) = {n_star}")
    return tau_star, n_star


# ===========================================================================
# STEP 6 -- Evaluate tau*: refine labels and compute ARI / entropy curves
# ===========================================================================
# Utilities reused from plot_and_summary.py (SpaGCN module imported there as
# spg) and the local refine() defined in plot_and_summary.py:
#   * spg.refine(sample_id, pred, dis, shape="hexagon"): majority-vote spatial
#     smoothing of the Louvain labels.
#   * adjusted_rand_score(label, ground_truth): accuracy of the selected
#     resolution against the manual annotation.
#   * entropy of the cluster-size distribution -> the res-vs-entropy /
#     res-vs-number / entropy-vs-number scatter grids (9 panels).
def step6_evaluate(tau_star):
    import SpaGCN_1000_random_seeds as spg
    from sklearn.metrics import adjusted_rand_score
    from collections import Counter
    from scipy.stats import entropy
    import matplotlib.pyplot as plt

    base_dir = ("/Data/Programs/SpaGCN_stabilization/stabilization_part/"
                + DATASET_INDEX + "/first100/")
    dataset_dir = "/Data/Datasets/SpatialLIBD/151673/"
    adata = sc.read_h5ad(dataset_dir + "processed_adata.h5ad")

    # spatial adjacency for refinement (hexagon grid of Visium)
    x_array = adata.obs["array_row"].tolist()
    y_array = adata.obs["array_col"].tolist()
    adj_2d = spg.calculate_adj_matrix(x=x_array, y=y_array, histology=False)

    res = round(float(tau_star), 3)
    result = pd.read_csv(base_dir
                         + "louvain_results/asw100_fusion_Euclid_median_res"
                         + str(res) + ".csv", index_col=0)
    adata.obs["pred"] = result["cluster"].tolist()
    adata.obs["pred"] = adata.obs["pred"].astype("category")

    refined_pred = spg.refine(sample_id=adata.obs.index.tolist(),
                              pred=adata.obs["pred"].tolist(),
                              dis=adj_2d, shape="hexagon")
    adata.obs["refined_pred"] = refined_pred
    adata.obs["refined_pred"] = adata.obs["refined_pred"].astype("category")

    ari_pre = adjusted_rand_score(adata.obs["pred"].tolist(),
                                  adata.obs["ground_truth"].tolist())
    ari_ref = adjusted_rand_score(adata.obs["refined_pred"].tolist(),
                                  adata.obs["ground_truth"].tolist())
    print(f"[151673] tau*={res}: ARI(raw)={ari_pre:.4f}, "
          f"ARI(refined)={ari_ref:.4f}, "
          f"n_clusters={adata.obs['refined_pred'].nunique()}")

    # ---- entropy/number curves across ALL resolutions (for the report) ----
    res_list, entropy_list, number_list = [], [], []
    for r in np.linspace(1.0, 1.2, 201):
        r = round(r, 3)
        df = pd.read_csv(base_dir + "louvain_results/"
                         "asw100_fusion_Euclid_median_res" + str(r) + ".csv",
                         index_col=0)
        sizes = np.array(sorted(Counter(df["cluster"]).values(), reverse=True))
        p = sizes / sizes.sum()
        res_list.append(r)
        entropy_list.append(entropy(p))
        number_list.append(df["cluster"].nunique())

    fig, ax = plt.subplots(1, 3, figsize=(12, 4), dpi=600)
    ax[0].scatter(res_list, entropy_list, s=5, alpha=0.5)
    ax[0].set_title("res vs entropy")
    ax[1].scatter(res_list, number_list, s=5, alpha=0.5)
    ax[1].set_title("res vs number")
    ax[2].scatter(entropy_list, number_list, s=5, alpha=0.5)
    ax[2].set_title("entropy vs number")
    for a in ax:
        a.axvline(res, color="red", ls="--") if a is ax[0] else None
    plt.tight_layout()
    plt.savefig(base_dir + DATASET_INDEX + "_res_entropy_number.png")
    plt.close()
    pd.DataFrame({"res": res_list, "entropy": entropy_list,
                  "number": number_list}).to_csv(base_dir
                                                 + "res_entropy_number.csv")




# ===========================================================================
# Main entry point
# ===========================================================================
if __name__ == "__main__":
    t0 = time.time()
    step1_run_1000_seeds()        # ~ hours on GPU; produces 1000results/
    step2_compute_asw()           # produces first100/asw_all.csv
    step3_baseline_ari()          # produces ari_1000_list.txt
    step4_fusion_and_louvain()    # produces fused matrix + 201 louvain CSVs
    tau_star, n_star = step5_select_resolution()   # <-- THE ANSWER
    step6_evaluate(tau_star)      # ARI report + entropy/number curves
    print(f"All done in {time.time() - t0:.1f}s")
