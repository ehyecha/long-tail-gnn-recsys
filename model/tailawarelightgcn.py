import torch
import torch.nn as nn
from torch_geometric.nn import LGConv
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.utils import dropout_edge
import numpy as np

class TailAwareLightGCN(pl.LightningModule):
    def __init__(self, num_nodes, embedding_dim, num_layers, learning_rate,
                 weight_decay, dropout, edge_index, tail_items=None, lambda_tail=0.3):
        super().__init__()

        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.convs = nn.ModuleList([LGConv() for _ in range(num_layers)])
        self.learning_rate = learning_rate
        self.register_buffer('edge_index', edge_index)
        self.weight_decay = weight_decay
        self.dropout_rate = dropout
        self.tail_items = tail_items  # tail 아이템 set
        self.lambda_tail = lambda_tail      # tai

        # validation epoch 동안 metrics 저장
        self.metrics_list = []

    def get_all_embeddings(self):
        x = self.embedding.weight
        all_embeddings = [x]
        edge_index_temp = self.edge_index.clone()
        if self.training and self.dropout_rate > 0:
            edge_index_temp, _ = dropout_edge(edge_index_temp, p=self.dropout_rate)
        for conv in self.convs:
            x = conv(x, edge_index_temp)
            all_embeddings.append(x)
        return torch.mean(torch.stack(all_embeddings), dim=0)

    def forward(self, user_ids, item_ids):
        all_embeddings = self.get_all_embeddings()
        return all_embeddings[user_ids], all_embeddings[item_ids]

    def training_step(self, batch, batch_idx):
        all_embeddings = self.get_all_embeddings()
        users, pos_items, neg_items = batch
        u_emb = all_embeddings[users.squeeze()]
        pos_emb = all_embeddings[pos_items.squeeze()]
        neg_emb = all_embeddings[neg_items.squeeze()]

        pos_score = (u_emb * pos_emb).sum(dim=-1)
        neg_score = (u_emb * neg_emb).sum(dim=-1)
        loss = -F.logsigmoid(pos_score - neg_score).mean()
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        users, pos_items, neg_items = batch
        users, pos_items, neg_items = users.squeeze(), pos_items.squeeze(), neg_items

        metrics = self.compute_metrics(users, pos_items, neg_items, K=20, tail_items=self.tail_items)
        self.metrics_list.append(metrics)

        # step 로그
        self.log("HR_step", metrics["HR@K"], prog_bar=False)
        self.log("NDCG_step", metrics["NDCG@K"], prog_bar=False)
        self.log("HR_TAIL_step", metrics["HR_TAIL@K"], prog_bar=False)
        self.log("NDCG_TAIL_step", metrics["NDCG_TAIL@K"], prog_bar=False)
        return metrics

    def on_validation_epoch_end(self):
        if not self.metrics_list:
            return

        epoch_hr = np.mean([m["HR@K"] for m in self.metrics_list])
        epoch_ndcg = np.mean([m["NDCG@K"] for m in self.metrics_list])
        epoch_hr_tail = np.mean([m["HR_TAIL@K"] for m in self.metrics_list])
        epoch_ndcg_tail = np.mean([m["NDCG_TAIL@K"] for m in self.metrics_list])

         # 조건부 custom_score
        custom_score = (1 - self.lambda_tail) * epoch_ndcg + self.lambda_tail * epoch_ndcg_tail
        
        print(f"HR@20={epoch_hr:.6f}")
        print(f"NDCG@20={epoch_ndcg:.6f}")
        print(f"HR_TAIL@20={epoch_hr_tail:.6f}")
        print(f"NDCG_TAIL@20={epoch_ndcg_tail:.6f}")
        print(f"custom_score={custom_score:.6f}")

        self.log("HR@20", epoch_hr, prog_bar=True)
        self.log("NDCG@20", epoch_ndcg, prog_bar=True)
        self.log("HR_TAIL@20", epoch_hr_tail, prog_bar=True)
        self.log("NDCG_TAIL@20", epoch_ndcg_tail, prog_bar=True)
        self.log("custom_score", custom_score, prog_bar=True)

        # 리스트 초기화
        self.metrics_list = []

    def compute_metrics(self, users, pos_items, neg_items, K=10, tail_items=None):
        hr_list, ndcg_list = [], []
        hr_tail, ndcg_tail = [], []
        all_embeddings = self.get_all_embeddings()

        for u, pos, neg in zip(users, pos_items, neg_items):
            items = torch.cat([pos.unsqueeze(0), neg], dim=0)
            u_emb = all_embeddings[u].unsqueeze(0)
            i_emb = all_embeddings[items]
            scores = (u_emb * i_emb).sum(dim=-1)

            k = min(K, scores.size(0))
            top_k = torch.topk(scores, k).indices

            # 전체 지표
            if 0 in top_k:
                rank = (top_k == 0).nonzero(as_tuple=True)[0].item() + 1
                hr_list.append(1)
                ndcg_list.append(1 / np.log2(rank + 1))
            else:
                hr_list.append(0)
                ndcg_list.append(0)

            # Tail 지표
            if tail_items is not None and pos.item() in tail_items:
                if 0 in top_k:
                    rank = (top_k == 0).nonzero(as_tuple=True)[0].item() + 1
                    hr_tail.append(1)
                    ndcg_tail.append(1 / np.log2(rank + 1))
                else:
                    hr_tail.append(0)
                    ndcg_tail.append(0)

        return {
            "HR@K": np.mean(hr_list),
            "NDCG@K": np.mean(ndcg_list),
            "HR_TAIL@K": np.mean(hr_tail) if hr_tail else 0.0,
            "NDCG_TAIL@K": np.mean(ndcg_tail) if ndcg_tail else 0.0
        }

    def predict_step(self, batch, batch_idx):
        users, items = batch
        user_emb, item_emb = self(users.squeeze(), items.squeeze())
        scores = (user_emb * item_emb).sum(dim=-1)
        return scores

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
