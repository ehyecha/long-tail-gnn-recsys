import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class BPRDatasetHybridSampling(Dataset):
    def __init__(
        self,
        user_item_interactions_df,
        all_unique_item_global_ids,
        past_positive_items,
        item_degree_dict,
        item_betweenness_dict,
        omega=0.5,
        alpha=0.8,
        beta=0.5,
        mode='train'
    ):
        self.mode = mode
        self.alpha = alpha
        self.omega = omega
        self.beta = beta

        self.user_positive_items = user_item_interactions_df.groupby('user_idx')['item_idx'].apply(list).to_dict()
        self.past_positive_items = past_positive_items
        self.all_unique_item_global_ids = np.array(list(all_unique_item_global_ids))
        self.all_users_in_dataset = list(self.user_positive_items.keys())
        self.all_items_set = set(self.all_unique_item_global_ids)
        self.item_idx_map = {item: i for i, item in enumerate(self.all_unique_item_global_ids)}

        # 1. Uniform sampling probabilities (precomputed globally)
        self.global_uniform_probs = np.ones(len(self.all_unique_item_global_ids), dtype=np.float32) / len(self.all_unique_item_global_ids)

        # 2. Degree-based sampling probabilities (tail-aware)
        degrees = np.array([item_degree_dict.get(i, 0) for i in self.all_unique_item_global_ids], dtype=np.float32)
        deg_probs_unnormalized = 1 / (np.power(degrees, self.omega) + 1e-8)
        self.global_deg_probs = deg_probs_unnormalized / deg_probs_unnormalized.sum()

        # 3. Betweenness-based sampling probabilities (tail-aware)
        betweenness = np.array([item_betweenness_dict.get(i, 0) for i in self.all_unique_item_global_ids], dtype=np.float32)
        bet_log = np.log1p(betweenness)
        bet_min, bet_max = bet_log.min(), bet_log.max()
        # 대부분의 값이 0에 가까우므로, 0을 0에 매핑하는 Min-Max 스케일링은 그대로 사용.
        # 그 다음, 값을 반전시켜 테일(낮은 값)이 높은 확률을 갖도록 함.
        bet_norm = (bet_log - bet_min) / (bet_max - bet_min + 1e-8)
        bet_probs_unnormalized = 1 / (np.power(bet_norm, self.omega) + 1e-8)
        self.global_bet_probs = bet_probs_unnormalized / bet_probs_unnormalized.sum()
        
        self.global_combined_probs = (self.beta * self.global_deg_probs) + ((1 - self.beta) * self.global_bet_probs)
        # 결합된 분포를 다시 정규화
        self.global_combined_probs /= self.global_combined_probs.sum()
        print(f"BPRDatasetMixedSampling initialized for {self.mode} mode.")

    def __len__(self):
        return len(self.all_users_in_dataset)

    def __getitem__(self, idx):
        user_id = self.all_users_in_dataset[idx]
        pos_items = self.user_positive_items.get(user_id, [])
        if not pos_items:
            return torch.tensor(-1), torch.tensor(-1), torch.tensor(-1)

        pos_item = np.random.choice(pos_items)
        all_pos_set = set(pos_items) | set(self.past_positive_items.get(user_id, []))
        
        neg_item = None
        if self.mode == 'train':
            # Create a pool of candidate negative items
            candidate_pool = list(self.all_items_set - all_pos_set)
            if not candidate_pool:
                return torch.tensor(-1), torch.tensor(-1), torch.tensor(-1)

            # Get indices for efficient probability retrieval
            candidate_indices = np.array([self.item_idx_map[item] for item in candidate_pool])
            
            # Decide on the sampling method
            r_type = np.random.rand()
            if r_type < self.alpha:
                # Use precomputed uniform probabilities
                probs = self.global_uniform_probs[candidate_indices]
            else:
                probs = self.global_combined_probs[candidate_indices]
            
            # Normalize the selected probabilities to sum to 1
            probs /= probs.sum()
            # Sample from the candidate pool using the tailored probabilities
            neg_item = np.random.choice(candidate_pool, p=probs)
        else:
            # For validation/test, a simple uniform sample is sufficient
            remaining_pool = list(self.all_items_set - all_pos_set)
            if not remaining_pool:
                return torch.tensor(-1), torch.tensor(-1), torch.tensor(-1)
            neg_item = np.random.choice(remaining_pool)

        return (
            torch.tensor(user_id, dtype=torch.long),
            torch.tensor(pos_item, dtype=torch.long),
            torch.tensor(neg_item, dtype=torch.long)
        )

