from tqdm import tqdm
import torch
import numpy as np
import random


def evaluate_model_full_ranking_revised(
    model,
    test_df,
    train_df,
    all_items,
    long_tail_items,
    K=20,
    K_tail = 50,
    batch_size=256,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    model.eval()
    model.to(device)

    all_item_ids = sorted(all_items)
    all_item_tensor = torch.LongTensor(all_item_ids).to(device)
    all_users = test_df['user_idx'].unique().tolist()

    train_user_pos_items = train_df.groupby('user_idx')['item_idx'].apply(set).to_dict()
    test_user_pos_items = test_df.groupby('user_idx')['item_idx'].apply(set).to_dict()

    HR, NDCG, recall_list = [], [], []
    long_tail_ratios = []
    all_predicted_items = set()
    all_predicted_tail_items = set()
    tail_HR_binary, tail_recall_list, tail_NDCG_list = [], [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(all_users), batch_size), desc="Evaluating Batches (Full Ranking)", leave=False):
            user_batch = all_users[i:i + batch_size]

            batch_user_tensor_expanded = torch.cat([torch.full((len(all_item_ids),), user, dtype=torch.long) for user in user_batch]).to(device)
            batch_item_tensor_expanded = all_item_tensor.repeat(len(user_batch))

            all_scores_batch = model.predict_step((batch_user_tensor_expanded, batch_item_tensor_expanded), batch_idx=None).cpu().numpy()

            start_idx = 0

            # 🚀 Fix: Initialize batch-wise lists BEFORE the inner loop
            batch_HR, batch_NDCG = [], []
            tail_hit_users = 0  # 롱테일 아이템 추천받은 유저 수
            for user in user_batch:
                end_idx = start_idx + len(all_item_ids)
                user_scores = all_scores_batch[start_idx:end_idx]

                all_item_scores_dict = {item_id: score for item_id, score in zip(all_item_ids, user_scores)}

                train_pos_items = train_user_pos_items.get(user, set())
                train_pos_item_idxs = train_pos_items

                for train_pos_idx in train_pos_item_idxs:
                    if train_pos_idx in all_item_scores_dict:
                        del all_item_scores_dict[train_pos_idx]
                
                all_scores = sorted(all_item_scores_dict, key=all_item_scores_dict.get, reverse=True)
                top_k_items = all_scores[:K]
                top_k_tail_items = all_scores[:K_tail]  # tail metrics용
                predicted_items_orig = top_k_items
                all_predicted_items.update(predicted_items_orig)

                true_items_orig = list(test_user_pos_items.get(user, set()))
                if not true_items_orig:
                    start_idx = end_idx
                    continue

                hits = len(set(predicted_items_orig) & set(true_items_orig))

                hit_ratio = 1.0 if hits > 0 else 0.0
                HR.append(hit_ratio)
                batch_HR.append(hit_ratio)

                dcg = 0.0
                idcg = sum([1.0 / np.log2(i + 2) for i in range(min(K, len(true_items_orig)))])
                for rank, item_id in enumerate(top_k_items):
                    if item_id in true_items_orig:
                        dcg += 1.0 / np.log2(rank + 2)
                ndcg_val = dcg / idcg if idcg > 0 else 0.0
                NDCG.append(ndcg_val)
                batch_NDCG.append(ndcg_val)

                num_hit = len(set(predicted_items_orig) & set(true_items_orig))
                recall = num_hit / len(true_items_orig) if len(true_items_orig) > 0 else 0
                recall_list.append(recall)

                long_tail_item_set = set(item for item in predicted_items_orig if item in long_tail_items)
                all_predicted_tail_items.update(long_tail_item_set)
                
           
                long_tail_count = sum(1 for item in top_k_tail_items if item in long_tail_items)
                long_tail_ratio = long_tail_count / K_tail
                long_tail_ratios.append(long_tail_ratio)

                relevant_tail_items = {i for i in true_items_orig if i in long_tail_items}
                if relevant_tail_items:
                    tail_hit = int(any(item in relevant_tail_items for item in top_k_tail_items))
                    tail_HR_binary.append(tail_hit)
                    tail_hit_users += tail_hit
                    tail_hits = len(set(top_k_tail_items) & relevant_tail_items)
                    tail_recall = tail_hits / len(relevant_tail_items)
                    tail_recall_list.append(tail_recall)

                    tail_dcg = 0.0
                    tail_idcg = sum([1.0 / np.log2(i + 2) for i in range(min(K_tail, len(relevant_tail_items)))])
                    for rank, item in enumerate(top_k_tail_items):
                        if item in relevant_tail_items:
                            tail_dcg += 1.0 / np.log2(rank + 2)
                    tail_ndcg = tail_dcg / tail_idcg if tail_idcg > 0 else 0.0
                    tail_NDCG_list.append(tail_ndcg)
                start_idx = end_idx

            avg_batch_HR = np.mean(batch_HR) if batch_HR else 0
            avg_batch_NDCG = np.mean(batch_NDCG) if batch_NDCG else 0
            tqdm.write(f"Batch {i//batch_size + 1}/{len(all_users)//batch_size + 1}: HR={avg_batch_HR:.4f}, NDCG={avg_batch_NDCG:.4f}")

    total_items = len(all_items)
    coverage = len(all_predicted_items) / total_items if total_items > 0 else 0.0
    total_tail_items = len(long_tail_items)
    tail_coverage = len(all_predicted_tail_items) / total_tail_items if total_tail_items > 0 else 0.0
    print(f"전체 유저 수: {len(all_users)}")
    print(f"테스트 데이터에서 테일 아이템을 가진 유저 수: {sum(1 for u, items in test_user_pos_items.items() if any(i in long_tail_items for i in items))}")
    print(f"Tail_HR_binary 길이: {len(tail_HR_binary)}")
    print(f"Tail hit 사용자 수: {sum(tail_HR_binary)}")


    return {
        "HR@K": np.mean(HR) if HR else 0.0,
        "NDCG@K": np.mean(NDCG) if NDCG else 0.0,
        "Recall@K": np.mean(recall_list) if recall_list else 0.0,
        "LongTailRatio": np.mean(long_tail_ratios) if long_tail_ratios else 0.0,
        "Coverage@K": coverage,
        "Tail_Hit@K": np.mean(tail_HR_binary) if tail_HR_binary else 0.0,
        "Tail_Recall@K": np.mean(tail_recall_list) if tail_recall_list else 0.0,
        "Tail_NDCG@K": np.mean(tail_NDCG_list) if tail_NDCG_list else 0.0,
        "TailCoverage@K": tail_coverage,
        "Tail_Hit_Users": sum(tail_HR_binary),
    }