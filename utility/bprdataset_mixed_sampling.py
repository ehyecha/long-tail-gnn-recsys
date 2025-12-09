import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import random

class BPRDatasetMixedSampling(Dataset):
    def __init__(
        self,
        user_item_interactions_df,
        all_unique_item_global_ids,
        past_positive_items,
        item_degree_dict,  # tail-aware 확률 계산용 아이템 차수 딕셔너리
        omega=0.5,
        alpha=0.8,
        mode='train'
    ):
        """
        BPR 학습 또는 평가용 데이터셋 (Uniform + Tail-aware 혼합 샘플링)

        Args:
            user_item_interactions_df (pd.DataFrame): 'user_idx', 'location_idx' 컬럼 포함
            all_unique_item_global_ids (list): 모든 아이템 전역 ID
            past_positive_items (dict): {user_id: [past_positive_item_ids]}
            item_degree_dict (dict): {item_id: degree}
            omega (float): tail-aware 계산 시 degree 지수
            alpha (float): uniform 샘플링 비율
            mode (str): 'train' 또는 'val'
        """
        self.mode = mode
        self.alpha = alpha
        self.omega = omega

        # 사용자별 긍정 아이템 딕셔너리
        positive_user_item_interactions_df = user_item_interactions_df
        self.user_positive_items = positive_user_item_interactions_df.groupby('user_idx')['item_idx'].apply(list).to_dict()

        self.past_positive_items = past_positive_items
        self.all_unique_item_global_ids = list(all_unique_item_global_ids)
        self.all_users_in_dataset = list(self.user_positive_items.keys())

        # tail-aware 확률 계산
        degrees = np.array([item_degree_dict.get(i, 0) for i in self.all_unique_item_global_ids], dtype=np.float32)
       
        #degree centrality
        self.tail_probs = 1 / (np.power(degrees, self.omega) + 1e-8)
        
        #between centrality
#       #self.tail_probs = 1 / (np.power((degrees + 1e-8), self.omega))
        self.tail_probs = self.tail_probs / self.tail_probs.sum()

        # uniform 확률
        self.uniform_probs = np.ones_like(self.tail_probs) / len(self.tail_probs)

        print(f"BPRDatasetMixedSampling initialized for {self.mode} mode.")
        print(f"Total unique users: {len(self.all_users_in_dataset)}, Total unique items: {len(self.all_unique_item_global_ids)}")

    def __len__(self):
        return len(self.all_users_in_dataset)

    def __getitem__(self, idx):
        user_global_id = self.all_users_in_dataset[idx]
        positive_items_list = self.user_positive_items.get(user_global_id, [])
        if not positive_items_list:
            return torch.tensor(-1), torch.tensor(-1), torch.tensor(-1)

        pos_item_global_id = np.random.choice(positive_items_list)
        positive_items_set = set(positive_items_list)
        past_positives = self.past_positive_items.get(user_global_id, [])
        all_positive_items_set = positive_items_set | set(past_positives)

        neg_item_global_id = None
        if self.mode == 'train':
            # mixed sampling
            for _ in range(50):  # 최대 50번 시도
                if np.random.rand() < self.alpha:
                    candidate = np.random.choice(self.all_unique_item_global_ids, p=self.uniform_probs)
                else:
                    candidate = np.random.choice(self.all_unique_item_global_ids, p=self.tail_probs)

                if candidate not in all_positive_items_set:
                    neg_item_global_id = candidate
                    break

            # 안전장치
            if neg_item_global_id is None:
                while True:
                    candidate = np.random.choice(self.all_unique_item_global_ids)
                    if candidate not in all_positive_items_set:
                        neg_item_global_id = candidate
                        break
        else:
            # validation/test: uniform
            while True:
                candidate_neg = np.random.choice(self.all_unique_item_global_ids)
                if candidate_neg not in all_positive_items_set:
                    neg_item_global_id = candidate_neg
                    break

        return (
            torch.tensor(user_global_id, dtype=torch.long),
            torch.tensor(pos_item_global_id, dtype=torch.long),
            torch.tensor(neg_item_global_id, dtype=torch.long)
        )

    def get_user_samples(self, user_global_id):
        pos_items_set = self.user_positive_items.get(user_global_id, set())
        return list(pos_items_set)

    def get_all_unique_users_in_dataset(self):
        return torch.tensor(self.all_users_in_dataset, dtype=torch.long)
