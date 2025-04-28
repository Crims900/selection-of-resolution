
import pandas as pd
import numpy as np
import scanpy as sc
import math
from scipy.sparse import issparse
import random, torch
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numba
import cv2
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import time
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@numba.njit("f4(f4[:], f4[:])")
def euclid_dist(t1,t2):
    sum=0
    for i in range(t1.shape[0]):
        sum+=(t1[i]-t2[i])**2
    return np.sqrt(sum)

@numba.njit("f4[:,:](f4[:,:])", parallel=True, nogil=True)
def pairwise_distance(X):
    n=X.shape[0]
    adj=np.empty((n, n), dtype=np.float32)
    for i in numba.prange(n):
        for j in numba.prange(n):
            adj[i][j]=euclid_dist(X[i], X[j])
    return adj

def prefilter_genes(adata,min_counts=None,max_counts=None,min_cells=10,max_cells=None):
    if min_cells is None and min_counts is None and max_cells is None and max_counts is None:
        raise ValueError('Provide one of min_counts, min_genes, max_counts or max_genes.')
    id_tmp=np.asarray([True]*adata.shape[1],dtype=bool)
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,min_cells=min_cells)[0]) if min_cells is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,max_cells=max_cells)[0]) if max_cells is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,min_counts=min_counts)[0]) if min_counts is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,max_counts=max_counts)[0]) if max_counts is not None  else id_tmp
    adata._inplace_subset_var(id_tmp)


def prefilter_specialgenes(adata,Gene1Pattern="ERCC",Gene2Pattern="MT-"):
    id_tmp1=np.asarray([not str(name).startswith(Gene1Pattern) for name in adata.var_names],dtype=bool)
    id_tmp2=np.asarray([not str(name).startswith(Gene2Pattern) for name in adata.var_names],dtype=bool)
    id_tmp=np.logical_and(id_tmp1,id_tmp2)
    adata._inplace_subset_var(id_tmp)


def calculate_adj_matrix(x, y, x_pixel=None, y_pixel=None, image=None, beta=49, alpha=1, histology=True):
    #x,y,x_pixel, y_pixel are lists
    if histology:
        assert (x_pixel is not None) & (x_pixel is not None) & (image is not None)
        assert (len(x)==len(x_pixel)) & (len(y)==len(y_pixel))
        print("Calculateing adj matrix using histology image...")
        #beta to control the range of neighbourhood when calculate grey vale for one spot
        #alpha to control the color scale
        beta_half=round(beta/2)
        g=[]
        for i in range(len(x_pixel)):
            max_x=image.shape[0]
            max_y=image.shape[1]
            nbs=image[max(0,x_pixel[i]-beta_half):min(max_x,x_pixel[i]+beta_half+1),max(0,y_pixel[i]-beta_half):min(max_y,y_pixel[i]+beta_half+1)]
            g.append(np.mean(np.mean(nbs,axis=0),axis=0))
        c0, c1, c2=[], [], []
        for i in g:
            c0.append(i[0])
            c1.append(i[1])
            c2.append(i[2])
        c0=np.array(c0)
        c1=np.array(c1)
        c2=np.array(c2)
        print("Var of c0,c1,c2 = ", np.var(c0),np.var(c1),np.var(c2))
        c3=(c0*np.var(c0)+c1*np.var(c1)+c2*np.var(c2))/(np.var(c0)+np.var(c1)+np.var(c2))
        c4=(c3-np.mean(c3))/np.std(c3)
        z_scale=np.max([np.std(x), np.std(y)])*alpha
        z=c4*z_scale
        z=z.tolist()
        print("Var of x,y,z = ", np.var(x),np.var(y),np.var(z))
        X=np.array([x, y, z]).T.astype(np.float32)
    else:
        print("Calculateing adj matrix using xy only...")
        X=np.array([x, y]).T.astype(np.float32)
    return pairwise_distance(X)

def calculate_p(adj, l):
    adj_exp=np.exp(-1*(adj**2)/(2*(l**2)))
    return np.mean(np.sum(adj_exp,1))-1

def search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100):
    run=0
    p_low=calculate_p(adj, start)
    p_high=calculate_p(adj, end)
    if p_low>p+tol:
        print("l not found, try smaller start point.")
        return None
    elif p_high<p-tol:
        print("l not found, try bigger end point.")
        return None
    elif  np.abs(p_low-p) <=tol:
        print("recommended l = ", str(start))
        return start
    elif  np.abs(p_high-p) <=tol:
        print("recommended l = ", str(end))
        return end
    while (p_low+tol)<p<(p_high-tol):
        run+=1
        print("Run "+str(run)+": l ["+str(start)+", "+str(end)+"], p ["+str(p_low)+", "+str(p_high)+"]")
        if run >max_run:
            print("Exact l not found, closest values are:\n"+"l="+str(start)+": "+"p="+str(p_low)+"\nl="+str(end)+": "+"p="+str(p_high))
            return None
        mid=(start+end)/2
        p_mid=calculate_p(adj, mid)
        if np.abs(p_mid-p)<=tol:
            print("recommended l = ", str(mid))
            return mid
        if p_mid<=p:
            start=mid
            p_low=p_mid
        else:
            end=mid
            p_high=p_mid


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.DoubleTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.DoubleTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class simple_GC_DEC(nn.Module):
    def __init__(self, nfeat, nhid, alpha=0.2):
        super(simple_GC_DEC, self).__init__()
        self.gc = GraphConvolution(nfeat, nhid).cuda()
        self.nhid = nhid
        # self.mu determined by the init method
        self.alpha = alpha

    def forward(self, x, adj):
        x = self.gc(x, adj)
        q = 1.0 / ((1.0 + torch.sum((x.unsqueeze(1) - self.mu.cuda()) ** 2, dim=2) / self.alpha) + 1e-8)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)
        return x, q

    def loss_function(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target * torch.log(target / (pred + 1e-6)), dim=1))

        loss = kld(p, q)
        return loss

    def target_distribution(self, q):
        # weight = q ** 2 / q.sum(0)
        # return torch.transpose((torch.transpose(weight,0,1) / weight.sum(1)),0,1)e
        p = q ** 2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    def fit(self, X, adj, lr=0.001, max_epochs=5000, update_interval=3, trajectory_interval=50, weight_decay=5e-4,
            opt="sgd", init="louvain", n_neighbors=10, res=0.4, n_clusters=10, init_spa=True, tol=1e-3):
        self.trajectory = []
        if opt == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        elif opt == "admin":
            optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        features = self.gc(X,adj)
        # ----------------------------------------------------------------
        if init == "kmeans":
            print("Initializing cluster centers with kmeans, n_clusters known")
            self.n_clusters = n_clusters
            kmeans = KMeans(self.n_clusters, n_init=20)
            if init_spa:
                # ------Kmeans use exp and spatial
                y_pred = kmeans.fit_predict(features.detach().numpy())
            else:
                # ------Kmeans only use exp info, no spatial
                y_pred = kmeans.fit_predict(X)  # Here we use X as numpy
        elif init == "louvain":
            print("Initializing cluster centers with louvain, resolution = ", res)
            if init_spa:
                adata = sc.AnnData(features.cpu().detach().numpy())
            else:
                adata = sc.AnnData(X.cpu())
            sc.pp.neighbors(adata, n_neighbors=n_neighbors)
            sc.tl.louvain(adata, resolution=res)
            y_pred = adata.obs['louvain'].astype(int).to_numpy()
            self.n_clusters = len(np.unique(y_pred))
            self.result = y_pred
        # ----------------------------------------------------------------
        y_pred_last = y_pred
        self.mu = Parameter(torch.DoubleTensor(self.n_clusters, self.nhid))
        self.trajectory.append(y_pred)
        features = pd.DataFrame(features.cpu().detach().numpy(), index=np.arange(0, features.shape[0]))
        Group = pd.Series(y_pred, index=np.arange(0, features.shape[0]), name="Group")
        Mergefeature = pd.concat([features, Group], axis=1)
        cluster_centers = np.asarray(Mergefeature.groupby("Group").mean())

        self.mu.data.copy_(torch.Tensor(cluster_centers))
        self.train()
        for epoch in range(max_epochs):
            if epoch % update_interval == 0:
                _, q = self.forward(X, adj)
                p = self.target_distribution(q).data
            if epoch % 10 == 0:
                print("Epoch ", epoch)
            optimizer.zero_grad()
            z, q = self(X, adj)
            loss = self.loss_function(p, q)
            loss.backward()
            optimizer.step()
            if epoch % trajectory_interval == 0:
                self.trajectory.append(torch.argmax(q, dim=1).data.cpu().numpy())

            # Check stop criterion
            y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float64) / X.shape[0]
            y_pred_last = y_pred
            if epoch > 0 and (epoch - 1) % update_interval == 0 and delta_label < tol:
                print('delta_label ', delta_label, '< tol ', tol)
                print("Reach tolerance threshold. Stopping training.")
                print("Total epoch:", epoch)
                break

    def fit_with_init(self, X, adj, init_y, lr=0.001, max_epochs=5000, update_interval=1, weight_decay=5e-4, opt="sgd"):
        print("Initializing cluster centers with kmeans.")
        if opt == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        elif opt == "admin":
            optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        X = torch.DoubleTensor(X)
        adj = torch.DoubleTensor(adj)
        features, _ = self.forward(X, adj)
        features = pd.DataFrame(features.detach().numpy(), index=np.arange(0, features.shape[0]))
        Group = pd.Series(init_y, index=np.arange(0, features.shape[0]), name="Group")
        Mergefeature = pd.concat([features, Group], axis=1)
        cluster_centers = np.asarray(Mergefeature.groupby("Group").mean())
        self.mu.data.copy_(torch.Tensor(cluster_centers))
        self.train()
        for epoch in range(max_epochs):
            if epoch % update_interval == 0:
                _, q = self.forward(torch.FloatTensor(X), torch.FloatTensor(adj))
                p = self.target_distribution(q).data
            X = torch.FloatTensor(X)
            adj = torch.FloatTensor(adj)
            optimizer.zero_grad()
            z, q = self(X, adj)
            loss = self.loss_function(p, q)
            loss.backward()
            optimizer.step()

    def predict(self, X, adj):
        z, q = self(X, adj)
        return z, q


class SpaGCN(object):
    def __init__(self):
        super(SpaGCN, self).__init__()
        self.l=None

    def set_l(self, l):
        self.l=l

    def train(self,embed,adj,
            num_pcs=50,
            lr=0.005,
            max_epochs=2000,
            weight_decay=0,
            opt="admin",
            init_spa=True,
            init="louvain", #louvain or kmeans
            n_neighbors=10, #for louvain
            n_clusters=None, #for kmeans
            res=0.4, #for louvain
            tol=1e-3):
        self.num_pcs=num_pcs
        self.res=res
        self.lr=lr
        self.max_epochs=max_epochs
        self.weight_decay=weight_decay
        self.opt=opt
        self.init_spa=init_spa
        self.init=init
        self.n_neighbors=n_neighbors
        self.n_clusters=n_clusters
        self.res=res
        self.tol=tol
        assert embed.shape[0]==adj.shape[0]==adj.shape[1]


        # pca = PCA(n_components=self.num_pcs)


        ###------------------------------------------###
        if self.l is None:
            raise ValueError('l should not be set before fitting the model!')
        adj_exp=torch.exp(-1*(adj**2)/(2*(self.l**2)))
        embed = torch.DoubleTensor(embed).cuda()
        #----------Train model----------

        self.model=simple_GC_DEC(embed.shape[1],embed.shape[1]).cuda()
        self.model.fit(embed,adj_exp,lr=self.lr,max_epochs=self.max_epochs,weight_decay=self.weight_decay,opt=self.opt,init_spa=self.init_spa,init=self.init,n_neighbors=self.n_neighbors,n_clusters=self.n_clusters,res=self.res, tol=self.tol)
        self.embed=embed
        self.adj_exp=adj_exp


    def predict(self):
        z,q=self.model.predict(self.embed,self.adj_exp)
        y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
        # Max probability plot
        prob=q.cpu().detach().numpy()
        return y_pred, prob

def refine(sample_id, pred, dis, shape="hexagon"):
    refined_pred=[]
    pred=pd.DataFrame({"pred": pred}, index=sample_id)
    dis_df=pd.DataFrame(dis, index=sample_id, columns=sample_id)
    if shape=="hexagon":
        num_nbs=6
    elif shape=="square":
        num_nbs=4
    else:
        print("Shape not recongized, shape='hexagon' for Visium data, 'square' for ST data.")
    for i in range(len(sample_id)):
        index=sample_id[i]
        dis_tmp=dis_df.loc[index, :].sort_values()
        nbs=dis_tmp[0:num_nbs+1]
        nbs_pred=pred.loc[nbs.index, "pred"]
        self_pred=pred.loc[index, "pred"]
        v_c=nbs_pred.value_counts()
        if (v_c.loc[self_pred]<num_nbs/2) and (np.max(v_c)>num_nbs/2):
            refined_pred.append(v_c.idxmax())
        else:
            refined_pred.append(self_pred)
    return refined_pred

def initialization(index):
    #SpatialLIBD
    dataset_dir = "/Data/SpatialLIBD/" + index + "/"
    result_dir = "/Data/Programs/" + index + "/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    adata = sc.read_visium(dataset_dir)
    adata.obs["x_array"] = adata.obs["array_row"]
    adata.obs["y_array"] = adata.obs["array_col"]
    adata.obs["x_pixel"] = adata.obsm['spatial'][:, 1]
    adata.obs["y_pixel"] = adata.obsm['spatial'][:, 0]
    adata.obs["x_pixel"] = np.array(adata.obs["x_pixel"]).round().astype("int")
    adata.obs["y_pixel"] = np.array(adata.obs["y_pixel"]).round().astype("int")

    x_pixel = adata.obs["x_pixel"].tolist()
    y_pixel = adata.obs["y_pixel"].tolist()
    x_array = adata.obs["x_array"].tolist()
    y_array = adata.obs["y_array"].tolist()

    # normalization
    adata.var_names_make_unique()
    prefilter_genes(adata, min_cells=3)  # avoiding all genes are zeros
    prefilter_specialgenes(adata)
    # Normalize and take log for UMI
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)

    print("data preprocess completed")

    img = cv2.imread(dataset_dir + index + "_full_image.tif")
    s = 1
    b = 49
    adj = calculate_adj_matrix(x=x_array, y=y_array, x_pixel=x_pixel, y_pixel=y_pixel, image=img, beta=b, alpha=s,
                                   histology=True)


    p = 0.5
    l = search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100)
    adj_2d = calculate_adj_matrix(x=x_array, y=y_array, histology=False)

    pca = PCA(n_components=50)

    if issparse(adata.X):
        pca.fit(adata.X.A)
        embed = pca.transform(adata.X.A)
    else:
        pca.fit(adata.X)
        embed = pca.transform(adata.X)

    return dataset_dir,result_dir,adata,x_pixel,y_pixel,adj,l,adj_2d,embed




def single_experiment(adata,adj,l,adj_2d,seed,result_dir,res=0.7):
    r_seed = t_seed = n_seed = seed
    res = res

    # Training
    clf = SpaGCN()
    clf.set_l(l)
    # Set seed
    random.seed(r_seed)
    torch.manual_seed(t_seed)
    np.random.seed(n_seed)
    # Run
    adj = torch.from_numpy(adj).double().cuda()
    clf.train(adata, adj, init_spa=True, init="louvain", res=res, tol=5e-3, lr=0.05, max_epochs=200)
    y_pred, prob = clf.predict()
    adata.obs["pred"]= y_pred
    adata.obs["pred"]=adata.obs["pred"].astype('category')

    refined_pred=refine(sample_id=adata.obs.index.tolist(), pred=adata.obs["pred"].tolist(), dis=adj_2d, shape="hexagon")
    adata.obs["refined_pred"]=refined_pred
    adata.obs["refined_pred"]=adata.obs["refined_pred"].astype('category')
    # Plot
    plot_color = ["#F56867", "#FEB915", "#C798EE", "#59BE86", "#7495D3", "#D1D1D1",
                  "#6D1A9C", "#15821E", "#3A84E6", "#997273", "#787878", "#DB4C6C", "#9E7A7A", "#554236", "#1F5F3C",
                  "#B3796C", "#F9BD3F", "#DAB370", "#877F6C", "#268785"]

    domains="pred"
    label = adata.obs[domains].to_list()
    truth = adata.obs['ground_truth'].to_list()
    ari = adjusted_rand_score(label,truth)
    num_celltype=len(adata.obs[domains].unique())
    adata.uns[domains+"_colors"]=list(plot_color[:num_celltype])
    ax=sc.pl.scatter(adata,alpha=1,x="y_pixel",y="x_pixel",color=domains,title=domains + " ARI:{}".format(ari),color_map=plot_color,show=False,size=50000/adata.shape[0])
    ax.set_aspect('equal', 'box')
    ax.axes.invert_yaxis()
    plt.savefig(result_dir+'pred.png', dpi=600)
    plt.close()

    domains="refined_pred"
    label = adata.obs[domains].to_list()
    truth = adata.obs['ground_truth'].to_list()
    ari = adjusted_rand_score(label,truth)
    num_celltype=len(adata.obs[domains].unique())
    adata.uns[domains+"_colors"]=list(plot_color[:num_celltype])
    ax=sc.pl.scatter(adata,alpha=1,x="y_pixel",y="x_pixel",color=domains,title=domains + " ARI:{}".format(ari),color_map=plot_color,show=False,size=50000/adata.shape[0])
    ax.set_aspect('equal', 'box')
    ax.axes.invert_yaxis()
    plt.savefig(result_dir+'refined_pred.png', dpi=600)
    plt.close()
    print("ARI:",ari)



def multiple_experiments(index,adata,adj,l,result_dir,embed,res=0.7):
    random.seed(100)
    random_list = random.sample(range(1, 100000), 1000)
    adj = torch.from_numpy(adj).double().cuda()

    print("start")
    for i in range(0,1000):

        time_begin = time.time()

        seed = random_list[i]
        #Set the random seed
        r_seed=t_seed=n_seed=seed
        res = res

        #Training

        clf=SpaGCN()
        clf.set_l(l)
        #Set seed
        random.seed(r_seed)
        torch.manual_seed(t_seed)
        np.random.seed(n_seed)

        #Run
        clf.train(embed,adj,init_spa=True,init="louvain",res=res, tol=5e-3, lr=0.05, max_epochs=200)
        y_pred, prob=clf.predict()
        mu = clf.model.mu.detach().numpy()

        np.savetxt(result_dir + str(i+1) + '_y_pred.txt', y_pred, fmt='%f', delimiter=',')
        np.savetxt(result_dir + str(i+1) + '_prob.txt', prob, fmt='%f', delimiter=',')
        np.savetxt(result_dir + str(i+1) + '_mu.txt', mu, fmt='%f', delimiter=',')

        time_end = time.time()
        print("Time cost:",time_end-time_begin,"s",'for ',i+1,'th seed',seed)

def main():
    for index in ["151507","151508","151509","151510","151669","151670","151671","151672","151673","151674","151675","151676"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        dataset_dir,result_dir,adata,x_pixel,y_pixel,adj,l,adj_2d,embed = initialization(index)
        multiple_experiments(index,adata,adj,l,result_dir,embed,res=0.7)

if __name__ == "__main__":
    main()




