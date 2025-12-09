import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import random

import torch
import numpy as np
import pandas as pd # 예시 데이터 생성을 위해 사용

class BPRDataset(Dataset):
    def __init__(self, user_item_interactions_df, all_unique_item_global_ids, past_positive_items, mode='train'):
        """
        BPR 학습 또는 평가를 위한 데이터셋.

        Args:
            user_item_interactions_df (pd.DataFrame):
                사용자-아이템 상호작용 데이터프레임. 반드시 'user_idx'와 'parent_asin_idx' 컬럼을 포함해야 하며,
                이들은 전역 ID여야 합니다. (예: sampled_train 또는 sampled_val)

            all_unique_item_global_ids (list or torch.Tensor):
                현재 그래프에 존재하는 모든 '고유한 아이템 전역 ID' 목록.
                부정 샘플링 시 이 목록 내에서 무작위로 선택합니다.
                (예: hetero_data['item'].n_id.tolist() 또는 val_all_unique_item_ids_in_edges.tolist())

            mode (str): 'train' 또는 'val'. 로깅 등에 사용.
        """
        self.mode = mode

        # user_item_interactions_df를 사용하여 {user_global_id: [pos_item_global_id, ...]} 딕셔너리 생성
        # 이 딕셔너리의 키와 값은 모두 '전역 ID'여야 합니다.
        positive_user_item_interactions_df = user_item_interactions_df
        self.user_positive_items = positive_user_item_interactions_df.groupby('user_idx')['item_idx'].apply(list).to_dict()

        self.past_positive_items =  past_positive_items
        # 부정 샘플링을 위한 모든 아이템 전역 ID 목록 (리스트로 변환하여 np.random.choice 사용)
        self.all_unique_item_global_ids = list(all_unique_item_global_ids)

        # 이 데이터셋에 포함된 모든 고유 사용자 ID 목록 (NeighborLoader input_nodes 구성에 사용)
        self.all_users_in_dataset = list(self.user_positive_items.keys())

        print(f"BPRDatasetV2 initialized for {self.mode} mode.")
        print(f"  Total unique users in this dataset: {len(self.all_users_in_dataset)}")
        print(f"  Total unique items for negative sampling: {len(self.all_unique_item_global_ids)}")
        if len(self.all_users_in_dataset) > 0:
            sample_user = self.all_users_in_dataset[0]
            print(f"  Sample user {sample_user} has {len(self.user_positive_items[sample_user])} positive items.")

    def __len__(self):
        # 이 데이터셋의 길이는 BPR 샘플링을 할 수 있는 고유 사용자의 수입니다.
        return len(self.all_users_in_dataset)

    def __getitem__(self, idx):
        # Retrieve the global user ID
        user_global_id = self.all_users_in_dataset[idx]
        
        # Get the list of positive items for this user
        positive_items_list = self.user_positive_items.get(user_global_id, [])
        if not positive_items_list:
            # Handle users with no positive items by skipping or re-sampling
            # For simplicity, we'll just return an empty tensor
            return torch.tensor(-1), torch.tensor(-1), torch.tensor(-1)
        
        # Choose a random positive item
        pos_item_global_id = np.random.choice(positive_items_list)
        
        # Convert the positive items list to a set for fast lookup
        positive_items_set = set(positive_items_list)
        past_positives = self.past_positive_items.get(user_global_id, [])
        all_positive_items_set = positive_items_set | set(past_positives)
        
        # Sample a negative item
        neg_item_global_id = None
        # Loop until a negative item is found
        while True:
            # Sample a random item from the entire item pool
            candidate_neg = np.random.choice(self.all_unique_item_global_ids)
            # Check if it's a negative item using the efficient set lookup
            if candidate_neg not in all_positive_items_set:
                neg_item_global_id = candidate_neg
                break # Exit the loop once a valid negative item is found
        
        return (
            torch.tensor(user_global_id, dtype=torch.long),
            torch.tensor(pos_item_global_id, dtype=torch.long),
            torch.tensor(neg_item_global_id, dtype=torch.long)
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
