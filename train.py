import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader

from utility.bprdataset import BPRDataset
from utility.bprdataset_mixed_sampling import BPRDatasetMixedSampling
from utility.bprdataset_hybrid_sampling import BPRDatasetHybridSampling
from utility.bprdataset_multinegative import BPRDatasetMultiNegative
from model.lightgcn import LightGCN
from model.tailawarelightgcn import TailAwareLightGCN
from utility.dataloader import get_item_popularity, split_head_tail_items_by_cumulative_popularity

import numpy as np
import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import gc  # 가비지 컬렉션
import logging
from parse import parse_args

#parsing
args = parse_args()

print("args", args)

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)

# 콘솔로 로그를 출력하는 핸들러 생성
console_handler = logging.StreamHandler()
# 로그 포맷터 생성 (시간, 레벨, 메시지 포함)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# 로거에 핸들러 추가 (중복 추가 방지)
if not logger.handlers:
    logger.addHandler(console_handler)

# 로컬 체크포인트 디렉토리 생성 (매번 스크립트 시작 시 안전하게)
ckpt_dir = "checkpoints/"
os.makedirs(ckpt_dir, exist_ok=True)
S3_CHECKPOINT_DIR = ckpt_dir


local_last_ckpt_path = os.path.join(ckpt_dir, "last.ckpt")
ckpt_path_for_trainer = local_last_ckpt_path

if (args.dataset == 'gowalla'):
    total_data = np.load('./data/gowalla_all.npz', allow_pickle=True)
    train_df = pd.DataFrame(total_data['train'], columns=['user_id', 'item_id','user_idx','item_idx'])
    val_df = pd.DataFrame(total_data['val'], columns=['user_id', 'item_id','user_idx','item_idx'])
    if (args.sampling == 'dc'):
        degrees = np.load('./data/all_items_degree_gowalla.npz', allow_pickle= True)
        item_degree_dict = degrees['item_degree'].item()
    elif (args.sampling == 'bc'):
        degrees = np.load('./data/all_items_bc_gowalla.npz', allow_pickle= True)
        item_betweeness_dict  = degrees['item_bc'].item()
elif(args.dataset == 'animation'):
    total_data = np.load('./data/anime_dataset.npz', allow_pickle=True)
    train_df = pd.DataFrame(total_data['train'], columns=['user_id', 'item_id', 'rating', 'user_idx', 'item_idx'])
    val_df = pd.DataFrame(total_data['val'], columns=['user_id', 'item_id', 'rating', 'user_idx', 'item_idx'])
    test_df = pd.DataFrame(total_data['test'], columns=['user_id', 'item_id', 'rating', 'user_idx', 'item_idx'])
    degrees = np.load('./data/all_items_degree_anime.npz', allow_pickle= True)
    item_degree_dict = degrees['item_degree'].item()


print(train_df.shape, val_df.shape)
#훈련데이터에 있는 사용자, 위치만 사용함

train_users = train_df['user_id'].unique()
train_locations = train_df['item_id'].unique()

val_df = val_df[val_df['user_id'].isin(train_users)]
val_df = val_df[val_df['item_id'].isin(train_locations)].reset_index(drop=True)


train_df['user_idx'] = train_df['user_idx'].astype(int)
train_df['item_idx'] = train_df['item_idx'].astype(int)


val_df['user_idx'] = val_df['user_idx'].astype(int)
val_df['item_idx'] = val_df['item_idx'].astype(int)


# hyper parameter
embedding_dim = args.embedding
learning_rate = args.lr
epochs = args.epochs
num_layers = args.layer
batch_size = args.bpr_batch
weight_decay = args.decay
dropout = args.dropout
num_workers = args.num_workers

# 데이터셋 및 로더 구성
train_user_idx = train_df['user_idx'].values
train_location_idx = train_df['item_idx'].values

combined_train = np.array([train_user_idx, train_location_idx])
train_user_item_edge_index = torch.tensor(combined_train, dtype=torch.long)

combined_train2 = np.array([train_location_idx, train_user_idx])
train_item_user_edge_index = torch.tensor(combined_train2, dtype=torch.long) # 인덱스는 보통 long 타입으로 지정

edge_index = torch.cat([train_user_item_edge_index, train_item_user_edge_index], dim =1)

all_items = train_df['item_idx'].unique()

past_positive_interaction_map_train = (
                train_df.groupby('user_idx')['item_idx']
                .apply(set).to_dict()
)

positive_df = pd.concat([train_df, val_df])

past_positive_interaction_map_val = (
                positive_df.groupby('user_idx')['item_idx']
                .apply(set).to_dict())

if (args.sampling == 'base'):
    train_dataset = BPRDataset(train_df, all_items, past_positive_interaction_map_train, mode='train')
else:
    if (args.sampling == 'bc'):
        betweenness = np.array(list(item_betweeness_dict.values()))
        bet_log = np.log1p(betweenness)
        bet_norm_log = (bet_log - bet_log.min()) / (bet_log.max() - bet_log.min() + 1e-8)
        scaled_betweenness = dict(zip(item_betweeness_dict.keys(), bet_norm_log))
        train_dataset =  BPRDatasetMixedSampling(
                train_df,
                all_items,
                past_positive_interaction_map_train,
                scaled_betweenness,  # tail-aware 확률 계산용 아이템 차수 딕셔너리
                omega=args.omega,
                alpha=args.alpha,
                mode='train')
    elif (args.sampling == 'dc'):
        train_dataset =  BPRDatasetMixedSampling(
                train_df,
                all_items,
                past_positive_interaction_map_train,
                item_degree_dict,  # tail-aware 확률 계산용 아이템 차수 딕셔너리
                omega=args.omega,
                alpha=args.alpha,
                mode='train')
    else:
        train_dataset =  BPRDatasetHybridSampling(
            train_df,
            all_items,
            past_positive_interaction_map_train,
            item_degree_dict,  # tail-aware 확률 계산용 아이템 차수 딕셔너리
            item_betweeness_dict ,
            omega=args.omega,
            alpha=args.alpha,
            beta = args.beta,
            mode='train')

    
val_dataset = BPRDatasetMultiNegative(val_df, all_items, past_positive_interaction_map_val, mode='val', num_negatives=20)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# # 모델 정의

num_users = train_df['user_idx'].max() + 1
num_items = train_df['item_idx'].max() + 1
num_nodes = num_users + num_items

if (args.sampling =='base'):
    model = LightGCN(num_nodes, embedding_dim, num_layers, learning_rate, weight_decay, dropout, edge_index)
else:
    items = train_df['item_idx']
    item_counts = get_item_popularity(items.values)
    head_items, tail_items = split_head_tail_items_by_cumulative_popularity(item_counts)
    model = TailAwareLightGCN(num_nodes, embedding_dim, num_layers, 
                          learning_rate, weight_decay, dropout, edge_index, tail_items)
# for testing
ckpt_dir = "checkpoints/"

last_ckpt_callback = pl.callbacks.ModelCheckpoint(
    dirpath=S3_CHECKPOINT_DIR,
    filename="last",  # 저장될 파일 이름
    save_last=True, 
    save_top_k=0,     # 가장 마지막 것만 유지
    verbose=True,
)

# 2. best checkpoint
if (args.sampling == 'base'):
    best_ckpt_callback = pl.callbacks.ModelCheckpoint(
        dirpath=S3_CHECKPOINT_DIR,
        filename="best-checkpoint",
        monitor="NDCG@20",   # 또는 "HR@20"
        mode="max",          # NDCG나 HR는 클수록 좋으므로 "max"
        save_top_k=1,        # best만 유지
        verbose=True, 
    )
    early_stop = EarlyStopping(monitor="NDCG@20", mode="max", patience= 10, verbose=True)
else:

    best_ckpt_callback = pl.callbacks.ModelCheckpoint(
        dirpath=S3_CHECKPOINT_DIR,
        filename="best-checkpoint",
        monitor="custom_score",   # 또는 "HR@20"
        mode="max",          # NDCG나 HR는 클수록 좋으므로 "max"
        save_top_k=1,        # best만 유지
        verbose=True, 
    )
    early_stop = EarlyStopping(monitor="custom_score", mode="max", patience= 15, verbose=True)

# # Trainer 정의
trainer = pl.Trainer(
    enable_checkpointing=True,
    # logger=wandb_logger,
    max_epochs=epochs,
    precision="16-mixed",
    num_sanity_val_steps=0,  # ← 이걸로 sanity check 비활성화
    callbacks=[last_ckpt_callback,best_ckpt_callback, early_stop],
)

# local_last_ckpt_path
ckpt_path = local_last_ckpt_path if os.path.exists(local_last_ckpt_path) else None

# 학습 시작
try:
    if os.path.exists("checkpoints/last.ckpt"):
        trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path_for_trainer)
    else:
        trainer.fit(model, train_loader, val_loader)

    # 이 부분이 중요합니다! 훈련이 성공적으로 끝난 후에만 실행됩니다.
    logger.info("--- 훈련 성공. 최종 모델 저장 및 업로드 시도 ---")
    
    final_model_path = os.path.join(ckpt_dir, "final_model.ckpt")
    trainer.save_checkpoint(final_model_path)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "final_model.pth"))
    
except Exception as e:
    logger.error(f"❌ 훈련 중 오류 발생: {e}")
    # 오류가 발생한 경우에 대한 추가적인 처리 (선택 사항)

finally:
    # ⭐⭐ 이 부분은 훈련 성공/실패와 관계없이 항상 실행되어야 합니다. ⭐⭐
    logger.info("--- 훈련 종료 ---")
    # 메모리 정리 및 로컬 디렉토리 삭제
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("모델 학습 스크립트 최종 종료.")