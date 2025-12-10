# train.py 파일 내용
import sys
import os



import torch # torch 모듈 추가 (혹시 pytorch가 아직 설치 안됐다면 여기서 에러날수 있음)
import pandas as pd
import numpy as np

from utility.bprdataset import BPRDataset
from utility.bprdataset_multi_negative import BPRDatasetMultiNegative
from torch_geometric.loader import DataLoader

# print("--- train.py 스크립트 시작 ---", flush=True)

# try:
#     import torch_geometric
#     print("torch_geometric is already installed.")
# except ImportError:
#     print("torch_geometric not found. Installing...")
#     os.system('pip install torch-geometric')
#     print("torch_geometric installation complete.")

# os.system('pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html')


import pytorch_lightning as pl
import torch.nn.functional as F
from torch_geometric.data import Data
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.optim as optim

from model.lightgcn import LightGCN
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
# from pytorch_lightning.loggers import WandbLogger
# import boto3
import os




# # BPRDataModule 수정: 이제 CustomBPRDataset을 사용합니다.
class BPRDataModule(pl.LightningDataModule):
    def __init__(self, train_data_path, batch_size,  num_layers):
        super().__init__()
        self.train_data_path = train_data_path
        self.batch_size = batch_size
        self.num_layers = num_layers


    def setup(self, stage=None):
        # 실제 데이터 파일 경로를 CustomBPRDataset에 전달합니다.
        # SageMaker 학습 시, 이 경로들은 `/opt/ml/input/data/training/` 같은 로컬 경로가 됩니다.

        #movielense
        sample_file_path = os.path.join(self.train_data_path, 'anime_sampled_data1028.npz')
        total_data = np.load(sample_file_path, allow_pickle=True)


        train_df = pd.DataFrame(total_data['train'], columns=['user_id', 'anime_id', 'rating', 'user_idx', 'anime_idx'])
        val_df = pd.DataFrame(total_data['val'], columns=['user_id', 'anime_id', 'rating', 'user_idx', 'anime_idx'])

        
        num_users = int(total_data['num_users'])
        num_items = int(total_data['num_items'])

        self.num_nodes = num_users + num_items
        self.all_items = total_data['all_items']
        self.past_positive_interaction_map_train= total_data['past_positive_interaction_map_train'].item()
        self.past_positive_interaction_map_val = total_data['past_positive_interaction_map_val'].item()
        

        train_users = train_df['user_id'].unique()
        train_locations = train_df['anime_id'].unique()
        
        val_df = val_df[val_df['user_id'].isin(train_users)]
        val_df = val_df[val_df['anime_id'].isin(train_locations)].reset_index(drop=True)


        train_df['user_idx'] = train_df['user_idx'].astype(int)
        train_df['anime_idx'] = train_df['anime_idx'].astype(int)


        val_df['user_idx'] = val_df['user_idx'].astype(int)
        val_df['anime_idx'] = val_df['anime_idx'].astype(int)
          
        print("train_data, val_data", train_df.shape, val_df.shape)

        train_user_idx = train_df['user_idx'].values
        train_location_idx = train_df['anime_idx'].values

        combined_train = np.array([train_user_idx, train_location_idx])
        train_user_item_edge_index = torch.tensor(combined_train, dtype=torch.long)

        combined_train2 = np.array([train_location_idx, train_user_idx])
        train_item_user_edge_index = torch.tensor(combined_train2, dtype=torch.long) # 인덱스는 보통 long 타입으로 지정

        edge_index = torch.cat([train_user_item_edge_index, train_item_user_edge_index], dim =1)

        if torch.cuda.is_available():
            self.edge_index = edge_index.to('cuda:0')
        else:
            self.edge_index = edge_index
        
   
        self.train_dataset = BPRDataset(train_df, self.all_items, self.past_positive_interaction_map_train, mode='train')
        self.val_dataset = BPRDatasetMultiNegative(val_df, self.all_items, self.past_positive_interaction_map_val, mode='val', num_negatives=20)

        #yelp sampled
        # sample_file_path = os.path.join(self.train_data_path, 'yelp_sampled_data1016.npz')
        # total_data = np.load(sample_file_path, allow_pickle=True)
        # train_df = pd.DataFrame(total_data['train'], columns=['user_id', 'item_id','user_idx','item_idx'])
        # val_df = pd.DataFrame(total_data['val'], columns=['user_id', 'item_id','user_idx','item_idx'])
        
        # num_users = int(total_data['num_users'])
        # num_itmes = int(total_data['num_items'])

        # self.num_nodes = num_users + num_itmes
        # self.all_items = total_data['all_items']
        # self.past_positive_interaction_map_train= total_data['past_positive_interaction_map'].item()
        # self.past_positive_interaction_map_val = total_data['past_positive_interaction_map_val'].item()
        

        # train_users = train_df['user_id'].unique()
        # train_locations = train_df['item_id'].unique()
        
        # val_df = val_df[val_df['user_id'].isin(train_users)]
        # val_df = val_df[val_df['item_id'].isin(train_locations)].reset_index(drop=True)


        # train_df['user_idx'] = train_df['user_idx'].astype(int)
        # train_df['item_idx'] = train_df['item_idx'].astype(int)


        # val_df['user_idx'] = val_df['user_idx'].astype(int)
        # val_df['item_idx'] = val_df['item_idx'].astype(int)
          
        # print("train_data, val_data", train_df.shape, val_df.shape)

        # train_user_idx = train_df['user_idx'].values
        # train_location_idx = train_df['item_idx'].values

        # combined_train = np.array([train_user_idx, train_location_idx])
        # train_user_item_edge_index = torch.tensor(combined_train, dtype=torch.long)

        # combined_train2 = np.array([train_location_idx, train_user_idx])
        # train_item_user_edge_index = torch.tensor(combined_train2, dtype=torch.long) # 인덱스는 보통 long 타입으로 지정

        # edge_index = torch.cat([train_user_item_edge_index, train_item_user_edge_index], dim =1)

        # if torch.cuda.is_available():
        #     self.edge_index = edge_index.to('cuda:0')
        # else:
        #     self.edge_index = edge_index
        
        # self.train_dataset = BPRDataset(train_df, self.all_items, self.past_positive_interaction_map_train, mode='train')
        # self.val_dataset = BPRDatasetMultiNegative(val_df, self.all_items, self.past_positive_interaction_map_val, mode='val', num_negatives=20)
        #yelp sampled

        #def __init__(self, user_item_interactions_df, all_unique_item_global_ids, past_positive_items, mode='train', num_negatives=5):
        #self.val_dataset = BPRDataset(val_df, self.all_items, self.past_positive_interaction_map_val, mode='val')
        
        
        # for train
        # sample_file_path = os.path.join(self.train_data_path, 'gowalla_sampled_1026.npz')
        # total_data = np.load(sample_file_path, allow_pickle=True)
        # train_df = pd.DataFrame(total_data['train'], columns=['user_id', 'item_id','user_idx','item_idx'])
        # val_df = pd.DataFrame(total_data['val'], columns=['user_id', 'item_id','user_idx','item_idx'])
                
        
        # train_users = train_df['user_id'].unique()
        # train_locations = train_df['item_id'].unique()

        # val_df = val_df[val_df['user_id'].isin(train_users)]
        # val_df = val_df[val_df['item_id'].isin(train_locations)].reset_index(drop=True)


        # train_df['user_idx'] = train_df['user_idx'].astype(int)
        # train_df['item_idx'] = train_df['item_idx'].astype(int)


        # val_df['user_idx'] = val_df['user_idx'].astype(int)
        # val_df['item_idx'] = val_df['item_idx'].astype(int)
          
        # print("train_data, val_data", train_df.shape, val_df.shape)

        # train_user_idx = train_df['user_idx'].values
        # train_location_idx = train_df['item_idx'].values

        # combined_train = np.array([train_user_idx, train_location_idx])
        # train_user_item_edge_index = torch.tensor(combined_train, dtype=torch.long)

        # combined_train2 = np.array([train_location_idx, train_user_idx])
        # train_item_user_edge_index = torch.tensor(combined_train2, dtype=torch.long) # 인덱스는 보통 long 타입으로 지정

        # edge_index = torch.cat([train_user_item_edge_index, train_item_user_edge_index], dim =1)

        # if torch.cuda.is_available():
        #     self.edge_index = edge_index.to('cuda:0')
        # else:
        #     self.edge_index = edge_index
        
        # num_users = int(total_data['num_users'])
        # num_items = int(total_data['num_items'])

        # self.num_nodes = num_users + num_items
        # self.all_items = total_data['all_items']
        # self.past_positive_interaction_map_train= total_data['past_positive_interaction_map'].item()
        # self.past_positive_interaction_map_val = total_data['past_positive_interaction_map_val'].item()
        
        # self.train_dataset = BPRDataset(train_df, self.all_items, self.past_positive_interaction_map_train, mode='train')
        # #self.val_dataset = BPRDataset(val_df, self.all_items, self.past_positive_interaction_map_val, mode='val')
        # self.val_dataset = BPRDatasetMultiNegative(val_df, self.all_items, self.past_positive_interaction_map_val, mode='val', num_negatives=20)



    def train_dataloader(self):
        #  return neighbor_loader_with_bpr(self.data, self.train_dataset.all_users_in_dataset, 
        #                                  self.train_dataset, batch_size=self.batch_size, num_neighbors=self.num_neighbors, shuffle=True)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        # return neighbor_loader_with_bpr(self.data, self.val_dataset.all_users_in_dataset, 
        #                                 self.val_dataset, batch_size=self.batch_size, num_neighbors=self.num_neighbors, shuffle=False)
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)


# # # --- 3. 학습 실행 로직 (main 함수) ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    # 실제 데이터셋의 사용자/아이템 수
    parser.add_argument("--num_users", type=int, default=10000) 
    parser.add_argument("--num_items", type=int, default=5000)

    # 데이터 파일 경로 (SageMaker 환경 변수로 전달될 경로)
    # SageMaker는 S3에 있는 데이터를 `/opt/ml/input/data/<channel_name>/` 경로에 마운트합니다.
    # 예: training 채널은 /opt/ml/input/data/training/ 에 매핑됩니다.
    # parser.add_argument('train', type=str, nargs='?', default=None, help='Dummy argument for SageMaker compatibility')
    parser.add_argument("--train_data_path", type=str, default=os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
    parser.add_argument("--val_data_path",type=str, default=os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation"))
    parser.add_argument("--embedding_dim", type=int, default=64)
    #parser.add_argument("--hidden_channel", type=int, default=64)
    #parser.add_argument("--out_channel", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./model"))
    parser.add_argument("--output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "./output"))
    parser.add_argument("--current_host", type=str, default=os.environ.get("SM_CURRENT_HOST", "algo-1"))
    parser.add_argument("--num_hosts", type=int, default=os.environ.get("SM_NUM_HOSTS", 1))
    
    args = parser.parse_args()

    data_module = BPRDataModule(
        train_data_path=args.train_data_path,
        batch_size=args.batch_size,
        num_layers=args.num_layers
    )
    data_module.setup() # setup()을 명시적으로 호출하여 데이터셋 생성

    print(f"--- DataModule 상태 ---", flush=True)
    print(f"num_users: {data_module.num_nodes}", flush=True)
    print(f"train_dataset 길이: {len(data_module.train_dataset)}", flush=True)
    print(f"val_dataset 길이: {len(data_module.val_dataset)}", flush=True)

    model = LightGCN(data_module.num_nodes, args.embedding_dim, args.num_layers, 
            args.lr, args.weight_decay, args.dropout)
    
    
    sagemaker_checkpoint_dir = os.environ.get('SM_CHECKPOINT_DIR', './checkpoints')

    checkpoint_callback = ModelCheckpoint(
        dirpath=sagemaker_checkpoint_dir,
        #filename="best_model-{epoch:02d}-{validation_loss:.4f}",
        filename="best_model-{epoch:02d}-{NDCG@20:.4f}",
        monitor="NDCG@20",   # 또는 "HR@20"
        mode="max",
        save_top_k=1,
        save_last=True
    )
    early_stop = EarlyStopping(monitor="NDCG@20", mode="max",       # HR/NDCG는 클수록 좋음
                               patience= 5, verbose=True)
     # ❗ Trainer 객체는 한 번만 생성합니다. 
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=[early_stop, checkpoint_callback],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed",
    )

    # 훈련 재개 로직
    # SageMaker가 파일을 다운로드했다면, 'last.ckpt' 파일은 항상 SM_CHECKPOINT_DIR에 존재합니다.
    checkpoint_path_for_trainer = os.path.join(sagemaker_checkpoint_dir, 'last.ckpt')

    # 체크포인트 파일이 존재하면 경로를 설정하고, 없으면 None으로 둡니다.
    resume_from_checkpoint = checkpoint_path_for_trainer if os.path.exists(checkpoint_path_for_trainer) else None

    if resume_from_checkpoint:
        print(f"체크포인트 파일 발견: {resume_from_checkpoint}. 훈련을 재개합니다.")
    else:
        print("체크포인트 파일을 찾을 수 없습니다. 처음부터 훈련을 시작합니다.")

    # ❗ Trainer.fit()을 한 번만 호출하고, ckpt_path 인자를 전달합니다.
    trainer.fit(model, datamodule=data_module, ckpt_path=resume_from_checkpoint)

    best_checkpoint_path = checkpoint_callback.best_model_path
    
    # 1. 베스트 체크포인트 파일이 존재하는지 확인합니다.
    if best_checkpoint_path and os.path.exists(best_checkpoint_path):
        print(f"가장 좋은 모델을 찾았습니다: {best_checkpoint_path}")
        final_model_state_dict = trainer.model.state_dict()
        
    else:
        # 베스트 체크포인트가 없을 경우 마지막 모델을 사용합니다.
        print("가장 좋은 모델을 찾을 수 없습니다. 마지막 모델을 저장합니다.")
        final_model_state_dict = model.state_dict() # 훈련이 끝난 시점의 model.state_dict()
    
    # 최종 모델을 SM_MODEL_DIR에 model.pth로 저장합니다.
    model_path = os.path.join(os.environ['SM_MODEL_DIR'], 'model.pth')
    torch.save(final_model_state_dict, model_path)