
import numpy as np
import os
import numba
import pandas as pd
import time

def initialization_load_adata(index):

    result_dir = "/Data/Programs/"+index + "_first100/"
    base_dir =  "/Data/Programs"+index+ "/"

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
	df = pd.read_csv(result_dir+"asw_all.csv",index_col=0,header=None)
	df.sort_values(1,inplace=True,ascending = False)
	idx_list = df.index[:100].tolist()
	return idx_list

# Calculate the median of the Euclidean distance matrix
def calculate_median_matrix(idx_list,base_dir,result_dir):
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
	np.savetxt(result_dir +  'asw100_fusion_Euclid_median.txt',Euclid_matrix_median,delimiter=',')


# Louvain
import networkx as nx
from community import community_louvain
import multiprocessing


def initialization_louvain(index):
    base_dir = "/Data/Programs/" + index  + "_first100/"
    result_dir = base_dir + "louvain_result/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    res_list = np.linspace(1.0, 1.2, 201)
    suffix = "_Euclid_median"
    weight = np.loadtxt(base_dir + 'asw100_fusion' + suffix + '.txt', delimiter=',')
    weight = np.exp(-1 * weight)
    G = nx.from_numpy_array(weight)
    return base_dir, result_dir, res_list, suffix, G


def louvain_batch(batch, base_dir, result_dir, res_list, suffix, G):
    for res in res_list[batch * 20:(batch + 1) * 20]:
        res = round(res, 3)

        time_start = time.time()

        partition1 = community_louvain.best_partition(G, resolution=res)
        result = pd.DataFrame({'index': list(partition1.keys()), 'cluster': list(partition1.values())})
        result.to_csv(result_dir + 'asw100_fusion' + suffix + '_res' + str(res) + '.csv', index=False)

        del partition1
        time_end = time.time()

        print('The time cost of louvain part of', suffix, "and res", res, 'is', time_end - time_start)

    print("batch", batch, "end")


def main_louvain(index):
	base_dir, result_dir, res_list, suffix, G = initialization_louvain(index)
	pool2 = multiprocessing.Pool(processes=10)
	for batch in range(10):
		pool2.apply_async(louvain_batch, (batch, base_dir, result_dir, res_list, suffix, G))
	pool2.close()
	pool2.join()
	print(index + "is finished,end")

def main():
	for index in ["151507","151508","151509","151510","151669","151670","151671","151672","151673","151674","151675","151676"]:

		result_dir,base_dir = initialization_load_adata(index)
		idx_list = get_first100_index(result_dir)
		calculate_median_matrix(idx_list,base_dir,result_dir)

	for index in ["151507","151508","151509","151510","151669","151670","151671","151672","151673","151674","151675","151676"]:
		main_louvain(index)

if __name__ == "__main__":
	main()





