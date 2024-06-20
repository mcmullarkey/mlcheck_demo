from losses import total_loss
from metrics import  AC, F1
import torch
from model import MyModel
from constants import num_heads
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score


class Trainer():
    def __init__(self) -> None:
        pass
    def initialize_data(self, graph, adj1, eta, n_clusters, lr1, clustering_labels, epochs, n):
        self.epochs = epochs
        self.graph = graph
        self.eta = eta
        self.clustering_labels = clustering_labels
        self.k = n_clusters
        self.init_w = torch.randn(n, 64)
        self.adj = torch.randn(n, n_clusters)

        self.y_actual = np.int64(adj1 > 0)
        num_community = n_clusters
        n_in_feat = self.init_w.shape[1]
        self.model = MyModel(self.graph, num_heads, num_community, self.k, n_in_feat)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = lr1)

    def train_loop(self):
        self.model(self.init_w, self.adj)
        Z = self.model.getZ()
        F = self.model.get_emb()
        self.Z_np = np.array(Z)
        pred = self.Z_np.argmax(1)
        loss = total_loss(self.y_actual, Z, F)
        self.optimizer.zero_grad()
        loss.requires_grad_(True)
        loss.backward()
        self.optimizer.step()
        return pred


    def train(self):
        for i in range(self.epochs):
            train = self.train_loop()
            pred_labels_z = train
            ac = AC(self.clustering_labels, pred_labels_z)
            f1 = F1(self.clustering_labels, pred_labels_z)
            nmi = nmi_score(self.clustering_labels, pred_labels_z, average_method='arithmetic')
            ari = ari_score(self.clustering_labels, pred_labels_z)
            print(f"epoch:{i}")
            print("ac:", ac)
            print("f1:", f1)
            print("nmi:", nmi)
            print("ari:", ari)



