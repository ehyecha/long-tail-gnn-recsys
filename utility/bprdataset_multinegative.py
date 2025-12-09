import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import random


class BPRDatasetMultiNegative(Dataset):
    def __init__(self, user_item_interactions_df, all_unique_item_global_ids, past_positive_items, mode='train', num_negatives=5):
        self.mode = mode
        self.num_negatives = num_negatives

        # positive pool 생성
        positive_user_item_interactions_df = user_item_interactions_df
        self.user_positive_items = positive_user_item_interactions_df.groupby('user_idx')['item_idx'].apply(list).to_dict()
        self.past_positive_items = past_positive_items

        self.all_unique_item_global_ids = list(all_unique_item_global_ids)
        self.all_users_in_dataset = list(self.user_positive_items.keys())

        print(f"BPRDatasetMultiNegative initialized for {self.mode} mode with {self.num_negatives} negatives per positive.")
        print(f"  Total users: {len(self.all_users_in_dataset)}, Total items: {len(self.all_unique_item_global_ids)}")

    def __len__(self):
        return len(self.all_users_in_dataset)

    def __getitem__(self, idx):
        user_global_id = self.all_users_in_dataset[idx]
        positive_items_list = self.user_positive_items.get(user_global_id, [])
        if not positive_items_list:
            return torch.tensor(-1), torch.tensor(-1), torch.full((self.num_negatives,), -1)

        # positive 하나 랜덤 선택
        pos_item_global_id = np.random.choice(positive_items_list)

        # positive set 구성 (현재 + 과거)
        positive_items_set = set(positive_items_list)
        past_positives = self.past_positive_items.get(user_global_id, [])
        all_positive_items_set = positive_items_set | set(past_positives)

        # negative 여러 개 샘플링
        neg_items = []
        while len(neg_items) < self.num_negatives:
            candidate_neg = np.random.choice(self.all_unique_item_global_ids)
            if candidate_neg not in all_positive_items_set:
                neg_items.append(candidate_neg)

        return (
            torch.tensor(user_global_id, dtype=torch.long),
            torch.tensor(pos_item_global_id, dtype=torch.long),
            torch.tensor(neg_items, dtype=torch.long)  # (num_negatives,)
        )
    
    def get_user_samples(self, user_global_id):
        """
        주어진 사용자 ID에 대한 긍정 아이템 ID 리스트를 반환합니다.
        부정 샘플링은 validation_step 내에서 서브그래프를 기반으로 수행됩니다.
        """
        pos_items_set = self.user_positive_items.get(user_global_id, set())
        return list(pos_items_set) # set을 리스트로 변환하여 반환

    def get_all_unique_users_in_dataset(self):
        """
        이 데이터셋에 포함된 모든 고유 사용자 전역 ID를 반환합니다.
        NeighborLoader의 input_nodes 인자로 사용됩니다.
        """
        return torch.tensor(self.all_users_in_dataset, dtype=torch.long)  
