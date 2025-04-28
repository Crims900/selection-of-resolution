
import os,csv,re
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import time
import anndata
from sklearn.metrics import silhouette_score

def initialization_load_adata(index):

    result_dir = "/Data/Programs/"+index+"_first100/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    base_dir =  "/Data/Programs/" + index +"/"

    return result_dir,base_dir

def t_sne_asw_all(idx,base_dir):
    prob = np.loadtxt(base_dir+str(idx)+"_prob"+".txt",delimiter=",")
    prob = anndata.AnnData(X=prob)
    gcn_label = pd.read_csv(base_dir+str(idx)+"_y_pred"+".txt",header = None)
    prob.obs["ground"] = gcn_label.values
    prob.obs['ground'] = prob.obs['ground'].astype('category')
    asw = silhouette_score(prob.X, prob.obs["ground"])
    return asw

def main():
    for index in ["151507","151508","151509","151510","151669","151670","151671","151672","151673","151674","151675","151676"]:
        result_dir,base_dir = initialization_load_adata(index)
        for i in range(1,1001):
            time1 = time.time()
            asw = t_sne_asw_all(i,base_dir)
            with open (result_dir+"asw_all.csv","a+",newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([i,asw])

            print("Time for idx:",i,"is",time.time()-time1)
        print("Time for index:",index,"is",time.time()-time1)


if __name__ == "__main__":
    main()


