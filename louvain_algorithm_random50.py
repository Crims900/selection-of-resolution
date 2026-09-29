import numpy as np
import os
import numba
import pandas as pd
import scipy.stats
import time
import random



# Add the Louvain clustering step.
import networkx as nx
from community import community_louvain
import multiprocessing

PROJECT_ROOT = "/Data/Programs/SpaGCN_stabilization/stabilization_part/"
DATASET_ROOT = "/Data/Datasets/"

RESOLUTIONS = [round(float(res), 2) for res in np.linspace(1.03, 1.12, 10)]


def initialization_louvain(index):
    base_dir = PROJECT_ROOT + index + "/50_random/"
    result_dir = base_dir + "louvain_results/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    res_list = RESOLUTIONS
    return base_dir, result_dir, res_list

def initialization_G(idx, base_dir):
    time1 = time.time()
    weight = np.loadtxt(base_dir + str(idx) + '_random_matrix.txt', delimiter=',')
    weight = np.exp(-1 * weight)
    G = nx.from_numpy_array(weight)
    time2 = time.time()
    print('The time cost of ', idx, 'G calculation is', time2 - time1)
    return G

def louvain_batch(result_dir, res, G, idx):

    res = round(float(res), 2)
    print(res)

    if os.path.exists(result_dir + 'clustering_'   + str(idx) + '_' + str(res) + '.csv'):
        print("res", res, "exists")
        return

    time_start = time.time()

    partition1 = community_louvain.best_partition(G, resolution=res)
    result = pd.DataFrame({'index': list(partition1.keys()), 'cluster': list(partition1.values())})
    result.to_csv(result_dir + 'clustering_'   + str(idx) + '_' + str(res) + '.csv', index=False)

    del partition1

    time_end = time.time()

    print('The time cost of louvain part of', idx, "and res", res, 'is', time_end - time_start)



def main_louvain():
    for index in ["MERFISH_0.04","MERFISH_0.09","MERFISH_0.14","MERFISH_0.19","MERFISH_0.24",
              "E9.5_E1S1.MOSTA","E9.5_E2S1.MOSTA","E9.5_E2S2.MOSTA","E9.5_E2S3.MOSTA","E9.5_E2S4.MOSTA",
              "151507", "151508", "151509", "151510", "151669", "151670","151671", "151672", "151673", "151674", "151675","151676",
              "Slice_1","Slice_2","Slice_3","BZ5","BZ14","BZ97","STARmap_BY3_1k"]:
        base_dir, result_dir, res_list = initialization_louvain(index)
        for i in range(1, 51):

            time1 = time.time()
            G = initialization_G(i, base_dir)

            pool2 = multiprocessing.Pool(processes=10)
            for batch in range(10):
                pool2.apply_async(louvain_batch, (result_dir, res_list[batch], G, i))
            pool2.close()
            pool2.join()
            time2 = time.time()
            print('The time cost of ', i, 'louvain part is', time2 - time1, "for index", index)


if __name__ == "__main__":
    main_louvain()
