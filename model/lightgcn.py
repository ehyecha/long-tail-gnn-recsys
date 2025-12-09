import torch
import torch.nn as nn
from torch_geometric.nn import LGConv
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.utils import dropout_edge # dropout_adj 함수를 import 합니다.
import numpy as np

class LightGCN(pl.LightningModule):
    def __init__(self, num_nodes, embedding_dim, num_layers, learning_rate, weight_decay, dropout, edge_index):
        super(LightGCN, self).__init__()

        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.convs = torch.nn.ModuleList([LGConv() for _ in range(num_layers)])
        self.learning_rate = learning_rate
        self.register_buffer('edge_index', edge_index)
        self.weight_decay = weight_decay
        self.dropout_rate = dropout

        self.hr_list = []    # epoch 동안 validation step 결과 저장
        self.ndcg_list = []
       

    
    def get_all_embeddings(self):
        """전체 forward 패스를 한 번 실행하는 헬퍼 함수."""
        x = self.embedding.weight
        all_embeddings = [x]
        edge_index_temp = self.edge_index.clone()
        
        if self.training and self.dropout_rate > 0:
            edge_index_temp, _ = dropout_edge(edge_index_temp, p=self.dropout_rate)
        
        for conv in self.convs:
            x = conv(x, edge_index_temp)
            all_embeddings.append(x)
            
        final_embeddings = torch.mean(torch.stack(all_embeddings), dim=0)
        return final_embeddings
    
    def forward(self, user_ids, item_ids):
        all_embeddings = self.get_all_embeddings()
        user_emb = all_embeddings[user_ids]
        item_emb = all_embeddings[item_ids]
        return user_emb, item_emb

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

    def validation_step(self, batch):

        # users, pos_items, neg_items = batch  # neg_items: [B, num_neg]
        # users, pos_items, neg_items = users.squeeze(), pos_items.squeeze(), neg_items
        users, pos_items, neg_items = batch  # neg_items: [B, num_neg]

        # shape 안전하게 변환
        users = users.view(-1)       # [B]
        pos_items = pos_items.view(-1)  # [B]
        if neg_items.dim() == 1:     # batch가 1일 때
            neg_items = neg_items.unsqueeze(0)  # [1, num_neg]
            
        hr, ndcg = self.compute_metrics(users, pos_items, neg_items, K=10)
    
        # step 로그
        self.log("HR_step", hr, prog_bar=False)
        self.log("NDCG_step", ndcg, prog_bar=False)
        
        # epoch-end에서 평균 계산용 리스트에 저장
        self.hr_list.append(hr)
        self.ndcg_list.append(ndcg)
        return {"HR@20": hr, "NDCG@20": ndcg}


    
    def on_validation_epoch_end(self):
        epoch_hr = np.mean(self.hr_list)
        epoch_ndcg = np.mean(self.ndcg_list)

        # SageMaker HPO용 stdout 출력
        print(f"HR@20={epoch_hr:.6f}")
        print(f"NDCG@20={epoch_ndcg:.6f}")

        # Lightning logging
        self.log("HR@20", epoch_hr, prog_bar=True)
        self.log("NDCG@20", epoch_ndcg, prog_bar=True)

        # 리스트 초기화
        self.hr_list = []
        self.ndcg_list = []

    def compute_metrics(self, users, pos_items, neg_items, K=20):
        """
        users: [B]
        pos_items: [B]
        neg_items: [B, num_neg]
        """
        all_embeddings = self.get_all_embeddings()  # [num_nodes, dim]

        # 임베딩 가져오기
        u_emb = all_embeddings[users]          # [B, dim]
        pos_emb = all_embeddings[pos_items]    # [B, dim]
        neg_emb = all_embeddings[neg_items]    # [B, num_neg, dim]

        # 브로드캐스트를 위해 u_emb 차원 확장
        u_emb_exp = u_emb.unsqueeze(1)         # [B, 1, dim]

        # 점수 계산
        pos_score = (u_emb * pos_emb).sum(dim=-1, keepdim=True)  # [B, 1]
        neg_score = (u_emb_exp * neg_emb).sum(dim=-1)            # [B, num_neg]

        # pos + neg 결합
        all_scores = torch.cat([pos_score, neg_score], dim=1)    # [B, 1 + num_neg]

        # top-K 추출
        topk_indices = torch.topk(all_scores, k=min(K, all_scores.size(1)), dim=1).indices

        # HR 계산
        hits = (topk_indices == 0).any(dim=1).float()  # positive item이 top-K 안에 있는지
        hr = hits.mean().item()

        # NDCG 계산
        ndcg = 0.0
        for i in range(topk_indices.size(0)):
            pos_ranks = (topk_indices[i] == 0).nonzero(as_tuple=False)
            if pos_ranks.numel() > 0:
                rank = pos_ranks[0].item() + 1
                ndcg += 1 / np.log2(rank + 1)
        ndcg = ndcg / users.size(0)

        return hr, ndcg 


    def predict_step(self, batch, batch_idx):
        """Lightning에서 `trainer.predict()`를 호출할 때 사용"""
        users, items = batch  # (user, candidate_items)
        user_emb, item_emb = self(users.squeeze(), items.squeeze())
        scores = (user_emb * item_emb).sum(dim=-1)
        return scores

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate,  weight_decay=self.weight_decay)