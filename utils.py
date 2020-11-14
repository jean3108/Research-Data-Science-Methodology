import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD, NMF, FactorAnalysis, FastICA, LatentDirichletAllocation
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection, johnson_lindenstrauss_min_dim


class CorrelatedFeats():
    
    def __init__(self, n_components, coefMin):
        self.n_components = n_components
        self.coefMin = coefMin
    
    def fit(self, X):
        #TODO compute correlation matric (pd.corr(method='pearson'|'speardman'))
        #Sorted by coef DESC 
        #For each couple, remove one column until you get n_components OR the coef < coefMin
        self.deletedColumns = []#index of deleted columns
        pass
    
    def transform(self, X):
        return np.delete(a, self.deletedColumns, 1)

class DatasetReds(Dataset):
    def __init__(self, data, target):
        self.data = torch.tensor(data, dtype=torch.float)
        self.target = torch.tensor(target, dtype=torch.float)

    def __getitem__(self, index):
        return (self.data[index], self.target[index])

    def __len__(self):
        return len(self.data)

class Autoencoder(nn.Module):
    
    def __init__(self, activation, layers, epsi, num_epochs, batch_size):
        super().__init__()
        self.act = activation()
        self.layers = []
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1], bias=True))
        self.epsi = epsi
        self.num_epochs = num_epochs
        self.batch_size = batch_size   

            
    def fit(self, X):
        self.train()
        train_loader = DataLoader(DatasetReds(X, X), shuffle=True, batch_size=self.batch_size)
        
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.epsi)
        optimizer.zero_grad()

        error = torch.nn.MSELoss()


        for epoch in range(self.num_epochs):
            for data, labels in train_loader:

                optimizer.zero_grad()
                outputs = self.forward(data)
                train_loss = error(outputs, data)

                train_loss.backward()
                optimizer.step()            
        
    
    def transform(self, X):
        self.eval()
        with torch.no_grad():
            return self.encoder(X).numpy()

        
    def encoder(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.act(x)
        return x

    
    def decoder(self, x):
        for layer in reversed(self.layers):
            x = self.act(F.linear(x, layer.weight.t()))
        return x

    
    def forward(self, x):
        encoded_feats = self.encoder(x)
        reconstructed_output = self.decoder(encoded_feats)
        return  reconstructed_output
    
    
    def parameters(self):
        parameters = []
        for layer in self.layers:
            parameters.extend(layer.parameters())
        return parameters


class ChangeRepresentation():
    
    def __init__(self, algo=None, parameters=None):
        
        self.changeRep = None
        
        if(algo=="PCA"):
            self.changeRep = self.pca(parameters)
        elif(algo=="NMF"):
            self.changeRep = self.nmf(parameters)
        elif(algo=="FA"):
            self.changeRep = self.fa(parameters)
        elif(algo=="FastICA"):
            self.changeRep = self.fastica(parameters)
        elif(algo=="LDA"):
            self.changeRep = self.lda(parameters)
        elif(algo=="LSA"):
            self.changeRep = self.lsa(parameters)
        elif(algo=="ISO"):
            self.changeRep = self.iso(parameters)
        elif(algo=="LLE"):
            self.changeRep = self.lle(parameters)
        elif(algo=="GaussianRP"):
            self.changeRep = self.gaussianRP(parameters)
        elif(algo=="SparseRP"):
            self.changeRep = self.sparseRP(parameters)
        elif(algo=="autoencoder"):
            self.changeRep = self.autoencoder(parameters)
        elif(algo=="removeCorrelatedFeatures"):
            self.changeRep = self.correlatedFeats(parameters)
            
        #virer feaztures non corélé à la sortie
        #classif puis tf-idf

    
    
    def fit(self, X):
        
        if(self.changeRep!=None):
            self.changeRep.fit(X)
    
    def transform(self, X):
        X = torch.tensor(X, dtype=torch.float)
        if(self.changeRep!=None):
            return self.changeRep.transform(X)
        
        return X
    
    #Algos
    
    def pca(self, parameters):
        #defaut parameters
        n_components = parameters["n_components"] if "n_components" in parameters else None
        kernel = parameters["kernel"] if "kernel" in parameters else "linear"
        gamma = parameters["gamma"] if "gamma" in parameters else None
        degree = parameters["degree"] if "degree" in parameters else 3
        coef0 = parameters["coef0"] if "coef0" in parameters else 1

        #algo Object
        return KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)
    
    
    def nmf(self, parameters):
        #defaut parameters
        n_components = parameters["n_components"] if "n_components" in parameters else None
        init = parameters["init"] if "init" in parameters else None
        solver = parameters["solver"] if "solver" in parameters else 'cd'
        beta_loss = parameters["beta_loss"] if "beta_loss" in parameters else "frobenius"
        tol = parameters["tol"] if "tol" in parameters else 1e-4
        max_iter = parameters["max_iter"] if "max_iter" in parameters else 200
        alpha = parameters["alpha"] if "alpha" in parameters else 0
        shuffle = parameters["shuffle"] if "shuffle" in parameters else False

        #algo Object
        return NMF(n_components=n_components, init=init, solver=solver, beta_loss=beta_loss, tol=tol, \
                   max_iter=max_iter, alpha=alpha, shuffle=shuffle)
    
    def fa(self, parameters):
        #defaut parameters     
        n_components = parameters["n_components"] if "n_components" in parameters else None
        noise_variance_init = parameters["noise_variance_init"] if "noise_variance_init" in parameters else None
        tol = parameters["tol"] if "tol" in parameters else 1e-4
        max_iter = parameters["max_iter"] if "max_iter" in parameters else 200
        svd_method = parameters["svd_method"] if "svd_method" in parameters else 'randomized'
        iterated_power = parameters["iterated_power"] if "iterated_power" in parameters else 3


        #algo Object
        return FactorAnalysis(n_components=n_components, noise_variance_init=noise_variance_init, tol=tol, \
                              svd_method=svd_method, max_iter=max_iter, iterated_power=iterated_power)
        
    def fastica(self, parameters):
        #defaut parameters
        n_components = parameters["n_components"] if "n_components" in parameters else None
        algorithm = parameters["algorithm"] if "algorithm" in parameters else "parallel"
        whiten = parameters["whiten"] if "whiten" in parameters else 'cd'
        fun = parameters["fun"] if "fun" in parameters else "logcosh"
        fun_args = parameters["fun_args"] if "fun_args" in parameters else None
        max_iter = parameters["max_iter"] if "max_iter" in parameters else 200
        tol = parameters["tol"] if "tol" in parameters else 1e-4
        
        #algo Object
        return FastICA(n_components=n_components, algorithm=algorithm, whiten=whiten, fun=fun, tol=tol, \
                   max_iter=max_iter, fun_args=fun_args)
        
    def lda(self, parameters):
        #defaut parameters
        n_components = parameters["n_components"] if "n_components" in parameters else None
        max_iter = parameters["max_iter"] if "max_iter" in parameters else 10

        #algo Object
        return LatentDirichletAllocation(n_components=n_components, max_iter=max_iter)
        
    def lsa(self, parameters):
        #defaut parameters
        n_components = parameters["n_components"] if "n_components" in parameters else None
        algorithm = parameters["algorithm"] if "algorithm" in parameters else "randomized"
        n_iter = parameters["n_iter"] if "n_iter" in parameters else 5
        tol = parameters["tol"] if "tol" in parameters else 0.0
        
        #algo Object
        return TruncatedSVD(n_components=n_components, tol=tol,  algorithm=algorithm, n_iter=n_iter)
  
    def iso(self, parameters):
        #defaut parameters
        n_components = parameters["n_components"] if "n_components" in parameters else None
        n_neighbors = parameters["n_neighbors"] if "n_neighbors" in parameters else 5
        metric = parameters["metric"] if "metric" in parameters else 'minkowski'
        tol = parameters["tol"] if "tol" in parameters else 0
        max_iter = parameters["max_iter"] if "max_iter" in parameters else None
        p = parameters["p"] if "p" in parameters else 2
        
        #algo Object
        return Isomap(n_components=n_components, n_neighbors=n_neighbors, metric=metric, tol=tol, \
                   max_iter=max_iter, p=p)
   
    def lle(self, parameters):
        #defaut parameters
        n_components = parameters["n_components"] if "n_components" in parameters else None
        n_neighbors = parameters["n_neighbors"] if "n_neighbors" in parameters else 5
        method = parameters["method"] if "method" in parameters else 'standard'
        tol = parameters["tol"] if "tol" in parameters else 1e-6
        max_iter = parameters["max_iter"] if "max_iter" in parameters else 100
    
        #algo Object
        return LocallyLinearEmbedding(n_components=n_components, n_neighbors=n_neighbors, tol=tol, \
                   max_iter=max_iter, method=method)
     
    def mds(self, parameters):
        #defaut parameters
        n_components = parameters["n_components"] if "n_components" in parameters else None
        metric = parameters["metric"] if "metric" in parameters else True
        n_init = parameters["n_init"] if "n_init" in parameters else 4
        eps = parameters["eps"] if "eps" in parameters else 1e-3
        max_iter = parameters["max_iter"] if "max_iter" in parameters else 300
        
        #algo Object
        return MDS(n_components=n_components, metric=metric, n_init=n_init, eps=eps, \
                   max_iter=max_iter)

    def tsne(self, parameters):
        #defaut parameters
        n_components = parameters["n_components"] if "n_components" in parameters else None
        perplexity = parameters["perplexity"] if "perplexity" in parameters else 30
        learning_rate = parameters["learning_rate"] if "learning_rate" in parameters else 200
        n_iter = parameters["n_iter"] if "n_iter" in parameters else 1000
        n_iter_without_progress = parameters["n_iter_without_progress"] if "n_iter_without_progress" in parameters else 300
        
        #algo Object
        return TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, \
                   learning_rate=learning_rate, n_iter_without_progress=n_iter_without_progress)
    
    def spectral(self, parameters):
        #defaut parameters
        n_components = parameters["n_components"] if "n_components" in parameters else None
        affinity = parameters["affinity"] if "affinity" in parameters else "nearest_neighbors"
        gamma = parameters["gamma"] if "gamma" in parameters else None
        eigen_solver = parameters["eigen_solver"] if "eigen_solver" in parameters else None
        n_neighbors = parameters["n_neighbors"] if "n_neighbors" in parameters else None
        
        #algo Object
        return SpectralEmbedding(n_components=n_components, affinity=affinity, \
                   gamma=gamma, eigen_solver=eigen_solver, n_neighbors=n_neighbors)
      
    def gaussianRP(self, parameters):
        #defaut parameters
        n_components = parameters["n_components"] if "n_components" in parameters else "auto"
        eps = parameters["eps"] if "eps" in parameters else 1e-1
        if('johnsonRP' in parameters):
            n_components = johnson_lindenstrauss_min_dim(parameters['johnsonRP']['n_samples'], eps=parameters['johnsonRP']['eps'])
        
        #algo Object
        return GaussianRandomProjection(n_components=n_components, eps=eps)
      
    def sparseRP(self, parameters):
        #defaut parameters
        n_components = parameters["n_components"] if "n_components" in parameters else "auto"
        density = parameters["density"] if "density" in parameters else 'auto'
        eps = parameters["eps"] if "eps" in parameters else 1e-1
        if('johnsonRP' in parameters):
            n_components = johnson_lindenstrauss_min_dim(parameters['johnsonRP']['n_samples'], eps=parameters['johnsonRP']['eps'])
        
        #algo Object
        return SparseRandomProjection(n_components=n_components, eps=eps, density=density)
    
    def autoencoder(self, parameters):
        #defaut parameters
        activation = parameters["activation"] if "activation" in parameters else nn.ReLU
        layers = parameters["layers"] if "layers" in parameters else [29, 14, 7]
        epsi = parameters["epsi"] if "epsi" in parameters else 1e-3
        num_epochs = parameters["num_epochs"] if "num_epochs" in parameters else 10
        batch_size = parameters["batch_size"] if "batch_size" in parameters else 100
       
        #algo Object
        return Autoencoder(activation=activation, layers=layers, epsi=epsi, num_epochs=num_epochs, batch_size=batch_size)
    
    def correlatedFeats(self, parameters):
        #defaut parameters
        n_components = parameters["n_components"] if "n_components" in parameters else None
        coefMin = parameters["coefMin"] if "coefMin" in parameters else 0.9
       
        #algo Object
        return CorrelatedFeats(n_components=n_components, coefMin=coefMin)
        
       