import pandas as pd
import numpy as np
import torch
from batch_evaluate import  evaluate_model_full_ranking_revised
from utility.dataloader import get_item_popularity,  split_head_tail_items_by_cumulative_popularity
from model.lightgcn import LightGCN
import os
import tarfile
import json


test_data_path = "/opt/ml/processing/input/data/gowalla_all.npz"
model_input_path = "/opt/ml/processing/input/model/"
model_tar_path = os.path.join(model_input_path, "model.tar.gz")

# 압축 해제할 디렉터리 생성 (선택 사항)
extracted_path = os.path.join(model_input_path, "extracted_model")
os.makedirs(extracted_path, exist_ok=True)

# 압축 해제
with tarfile.open(model_tar_path, "r:gz") as tar:
    tar.extractall(path=extracted_path)

print("extracted",os.listdir(extracted_path))

# best_checkpoint 파일의 전체 경로
checkpoint_file_path = os.path.join(extracted_path, 'model', 'best-checkpoint.ckpt')

# 파일이 존재하는지 확인 (선택 사항)
if not os.path.exists(test_data_path):
    raise FileNotFoundError(f"Test data not found at {test_data_path}")

total_data = np.load(test_data_path, allow_pickle=True)

#yelp

# train_df = pd.DataFrame(total_data['train'], columns=['user_id','item_id','user_idx', 'item_idx'])
# val_df = pd.DataFrame(total_data['val'], columns=['user_id','item_id','user_idx', 'item_idx'])
# test_df =  pd.DataFrame(total_data['test'], columns=['user_id','item_id','user_idx', 'item_idx'])

#gowalla
train_df = pd.DataFrame(total_data['train'], columns=['user_id', 'item_id','user_idx','item_idx'])
val_df = pd.DataFrame(total_data['val'], columns=['user_id', 'item_id','user_idx','item_idx'])
test_df = pd.DataFrame(total_data['test'], columns=['user_id','item_id','user_idx', 'item_idx'])

# train_df = pd.DataFrame(total_data['train'], columns=['user_id','item_id','user_idx', 'item_idx'])
# val_df = pd.DataFrame(total_data['val'], columns=['user_id','item_id','user_idx', 'item_idx'])
# test_df = pd.DataFrame(total_data['test'], columns=['user_id','item_id','user_idx', 'item_idx'])


#anime dataset
# train_df = pd.DataFrame(total_data['train'], columns=['user_id', 'item_id', 'rating', 'user_idx', 'item_idx'])
# val_df = pd.DataFrame(total_data['val'], columns=['user_id', 'item_id', 'rating', 'user_idx', 'item_idx'])
# test_df = pd.DataFrame(total_data['test'], columns=['user_id', 'item_id', 'rating', 'user_idx', 'item_idx'])

#movielense
# total_data = np.load(test_data_path, allow_pickle=True)
# train_df = pd.DataFrame(total_data['train'], columns=['item_id', 'title', 'genres', 'user_id', 'rating', 'timestamp',
#        'user_idx', 'item_idx'])
# val_df = pd.DataFrame(total_data['val'], columns=['item_id', 'title', 'genres', 'user_id', 'rating', 'timestamp',
#        'user_idx', 'item_idx'])
# test_df = pd.DataFrame(total_data['test'], columns=['item_id', 'title', 'genres', 'user_id', 'rating', 'timestamp',
#        'user_idx', 'item_idx'])
print(train_df.shape, val_df.shape)
# #훈련데이터에 있는 사용자, 위치만 사용함

train_users = train_df['user_id'].unique()
train_locations = train_df['item_id'].unique()

val_df = val_df[val_df['user_id'].isin(train_users)]
val_df = val_df[val_df['item_id'].isin(train_locations)].reset_index(drop=True)


test_df = test_df[test_df['user_id'].isin(train_users)]
test_df = test_df[test_df['item_id'].isin(train_locations)].reset_index(drop=True)




train_df['user_idx'] = train_df['user_idx'].astype(int)
train_df['item_idx'] = train_df['item_idx'].astype(int)


val_df['user_idx'] = val_df['user_idx'].astype(int)
val_df['item_idx'] = val_df['item_idx'].astype(int)


test_df['user_idx'] = test_df['user_idx'].astype(int)
test_df['item_idx'] = test_df['item_idx'].astype(int)

train_user_idx = train_df['user_idx'].values
train_location_idx = train_df['item_idx'].values

combined_train = np.array([train_user_idx, train_location_idx])
train_user_item_edge_index = torch.tensor(combined_train, dtype=torch.long)

combined_train2 = np.array([train_location_idx, train_user_idx])
train_item_user_edge_index = torch.tensor(combined_train2, dtype=torch.long) # 인덱스는 보통 long 타입으로 지정

edge_index = torch.cat([train_user_item_edge_index, train_item_user_edge_index], dim =1)







items = train_df['item_idx'].values
item_counts = get_item_popularity(items)
head_items, tail_items = split_head_tail_items_by_cumulative_popularity(item_counts)


# # 모델 정의
num_users = train_df['user_idx'].max() + 1
num_items = train_df['item_idx'].max() + 1
num_nodes = num_users + num_items



best_params = {
    'embedding_dim': 128, 
    'batch_size': 128, 
    'learning_rate': 0.003316329630716579, 
    'weight_decay': 4.347174837825076e-06, 
    'num_layers': 3,
    'dropout':	0.016266682301975397,
    'alpha': 0.7,
    'omega': 1

}

embedding_dim = best_params['embedding_dim']
lr = best_params['learning_rate']
epochs = 100
num_layers = best_params['num_layers']
dropout = best_params['dropout']
batch_size = best_params['batch_size']
weight_decay = best_params['weight_decay']
num_workers = 4

past_positive_interaction_map_train = (
                train_df.groupby('user_idx')['item_idx']
                .apply(set).to_dict()
)

positive_df = pd.concat([train_df, val_df])

past_positive_interaction_map_val = (
                positive_df.groupby('user_idx')['item_idx']
                .apply(set).to_dict())

all_items = train_df['item_idx'].unique()


loaded_model = LightGCN.load_from_checkpoint(checkpoint_file_path,
                          num_nodes=num_nodes,
                          embedding_dim=best_params['embedding_dim'],
                          num_layers=best_params['num_layers'],
                          learning_rate= best_params['learning_rate'],
                          weight_decay = best_params['weight_decay'],
                          dropout_rate = best_params['dropout'], 
                          edge_index = edge_index)


all_items = train_df['item_idx'].unique()


k_tail = 50
evaluate_results = evaluate_model_full_ranking_revised(
    loaded_model,
    test_df,
    train_df,
    all_items,
    tail_items,
    K=20,
    K_tail=k_tail,
    batch_size=64,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

output_data_path = "/opt/ml/processing/output/"
output_path = os.path.join(output_data_path, f'evaluation_full_tail{k_tail}.json')
print("results", evaluate_results)
with open(output_path, 'w') as f:
    json.dump(evaluate_results, f)