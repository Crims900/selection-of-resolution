import numpy as np
import os
import numba
import pandas as pd
import scipy.stats
import time
import random

PROJECT_ROOT = "/Data/Programs/SpaGCN_stabilization/stabilization_part/"
DATASET_ROOT = "/Data/Datasets/"

def initialization_load_adata(index):
    result_dir = PROJECT_ROOT + index + "/50_random/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    base_dir = PROJECT_ROOT + index + "/1000results/"
    return result_dir,base_dir



@numba.njit("f4(f4[:], f4[:])")
def euclid_dist(t1,t2):
    sum=0
    for i in range(t1.shape[0]):
       sum+=(t1[i]-t2[i])**2
    return np.sqrt(sum)

@numba.njit("f4[:,:](f4[:,:])", parallel=True, nogil=True)
def euclid_matrix(X):
    n=X.shape[0]
    adj=np.empty((n, n), dtype=np.float32)
    for i in numba.prange(n):
       for j in numba.prange(n):
          adj[i][j]=euclid_dist(X[i], X[j])
    return adj

def get_first100_index(result_dir):
    df1 = pd.read_csv(result_dir+"asw_all.csv",index_col=0,header=None)
    df1.sort_values(1,inplace=True,ascending = False)
    idx_list = df1.index[:100].tolist()
    return idx_list




# The following section calculates the median matrix.
def calculate_median_matrix(base_dir,result_dir,index):
    random.seed(100)
    for i in range(1,51):

        if os.path.exists(result_dir + str(i) + '_random_matrix.txt'):
            print("res", i, "exists")
            continue
        time1 = time.time()
        idx_list = random.sample(range(1, 1001), 100)
        Euclid_list = []
        for batch in idx_list:
            time1 = time.time()

            prob = np.loadtxt(base_dir + str(batch)+"_prob.txt",delimiter=',')
            Euclid_matrix = euclid_matrix(prob.astype(np.float32))
            Euclid_list.append(Euclid_matrix)
            del Euclid_matrix
            time2 = time.time()
            print('The time cost of batch',batch,'prob','calculation is',time2-time1)

        Euclid_matrix = np.array(Euclid_list,dtype=np.float32)
        print("Euclid_matrix composition is done!")

        Euclid_list.clear()
        del Euclid_list

        Euclid_matrix_median = np.median(Euclid_matrix,axis=0)
        del Euclid_matrix

        print("Euclid_matrix median calculation is done!")

        np.savetxt(result_dir + str(i) + '_random_matrix.txt',Euclid_matrix_median,delimiter=',')
        time2 = time.time()
        print('The time cost of batch',i,'median','calculation is',time2-time1,"for index",index)



def main():
    for index in ["MERFISH_0.04","MERFISH_0.09","MERFISH_0.14","MERFISH_0.19","MERFISH_0.24",
              "E9.5_E1S1.MOSTA","E9.5_E2S1.MOSTA","E9.5_E2S2.MOSTA","E9.5_E2S3.MOSTA","E9.5_E2S4.MOSTA",
              "151507", "151508", "151509", "151510", "151669", "151670","151671", "151672", "151673", "151674", "151675","151676",
              "Slice_1","Slice_2","Slice_3","BZ5","BZ14","BZ97","STARmap_BY3_1k"]:
        result_dir,base_dir = initialization_load_adata(index)
        calculate_median_matrix(base_dir,result_dir,index)


if __name__ == "__main__":
    main()
