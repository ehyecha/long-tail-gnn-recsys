import sagemaker
from sagemaker.pytorch import PyTorch
import os
from sagemaker.tuner import HyperparameterTuner, IntegerParameter, ContinuousParameter,  CategoricalParameter
from sagemaker.tuner import WarmStartConfig, WarmStartTypes

role = sagemaker.get_execution_role()
sess = sagemaker.Session()
bucket = sess.default_bucket()
prefix = 'sagemaker/bpr-hp-tuning' # S3 버킷 내의 경로



training_data_path = f's3://{bucket}/{prefix}/train_data' # S3 경로만 정의
validation_data_path = f's3://{bucket}/{prefix}/val_data'

current_script_dir = os.path.dirname(os.path.abspath(__file__))
source_dir_path = os.path.join(current_script_dir, 'src')

checkpoint_s3_uri = f's3://{bucket}/{prefix}/checkpoints/ligntgcn-model-tuning/'
checkpoint_local_path="/opt/ml/checkpoints"

parent_job_name = 'anime-lightgcn-251028-1429'

# #WarmStartConfig 객체 생성
# # WarmStartTypes.CONTINUE_TRAINING는 이전 작업에서 완료하지 못한 작업을 이어서 탐색하도록 지시합니다.
warm_start_config = WarmStartConfig(
    parents=[parent_job_name],
    warm_start_type=WarmStartTypes.IDENTICAL_DATA_AND_ALGORITHM
)

estimator = PyTorch(
    # entry_point='train_lightgcn.py',
    entry_point = 'train_lightgcn_anime.py',
    source_dir=source_dir_path,
    role=role,
    framework_version='2.3.0',
    py_version='py311',
    instance_count=1,
    #instance_type='ml.g5.xlarge',
    instance_type='ml.g4dn.xlarge',
    use_spot_instances=True,
    max_run=3600 * 12,  # 초 단위로 최대 4시간 실행
    max_wait=3600 * 13, # 초 단위로 최대 5시간 대기 (max_run보다 길거나 같아야 함)
    checkpoint_s3_uri=checkpoint_s3_uri,
    environment={'SM_CHECKPOINT_DIR': checkpoint_local_path, "WANDB_API_KEY": "307711fe800b379c13bbf3de8a153fb991d6e4af"},
    dependencies=['./src/requirements.txt'],
    hyperparameters={
        # "batch_size": 1024,  # This is the fixed parameter
        # "embedding_dim": 64,    # Another fixed parameter
        #"lr": 0.001,
        # "weight_decay": 1e-4,
        # "num_layers":3,
    }
)


# # 4. 하이퍼파라미터 튜닝 정의 (Optuna의 Study.optimize() 역할)
tuner = HyperparameterTuner(
    estimator, # 위에서 정의한 Estimator
    objective_metric_name='NDCG@20', # 학습 스크립트에서 Trainer.log로 기록하는 지표 이름
    objective_type='Maximize', # 이 지표를
    # objective_metric_name='validation_ndcg20',
    # objective_type='Maximize',
    base_tuning_job_name="anime-lightgcn",
    hyperparameter_ranges={ # 탐색할 하이퍼파라미터 범위
        # 'embedding_dim': CategoricalParameter([64, 96, 128]),
        # 'lr': ContinuousParameter(1e-4, 5e-3, scaling_type='Logarithmic'),
        # 'batch_size': CategoricalParameter([32, 64, 128]),
        # 'epochs': IntegerParameter(15, 20), # 충분히 큰 에포크 범위
        # 'dropout': ContinuousParameter(0, 0.3),
        # 'weight_decay': ContinuousParameter(1e-6, 1e-3, scaling_type='Logarithmic'),
        # 'num_layers': IntegerParameter(2,4),
        
        #movielense
        # 'embedding_dim': CategoricalParameter([32, 48, 64, 80]),         # 범위 줄임
        # 'lr': ContinuousParameter(0.0002, 0.002, scaling_type='Logarithmic'),
        # 'batch_size': CategoricalParameter([128, 256]),
        # 'epochs': IntegerParameter(15,20), # 충분히 큰 에포크 범위
        # 'dropout': ContinuousParameter(0, 0.2), 
        # 'weight_decay': ContinuousParameter(1e-6, 1e-4, scaling_type='Logarithmic'),
        # 'num_layers': IntegerParameter(3, 4),

        #anime
        'embedding_dim': CategoricalParameter([32, 48, 64, 80]),         # 범위 줄임
        'lr': ContinuousParameter(0.0001, 0.01, scaling_type='Logarithmic'),
        'batch_size': CategoricalParameter([64, 128, 256]),
        'epochs': IntegerParameter(15,20), # 충분히 큰 에포크 범위
        'dropout': ContinuousParameter(0, 0.3), 
        'weight_decay': ContinuousParameter(1e-6, 1e-4, scaling_type='Logarithmic'),
        'num_layers': IntegerParameter(1,3),
     

        #yelp2018
        # 'embedding_dim': CategoricalParameter([64, 96, 128]),
        # 'lr': ContinuousParameter(5e-4, 1.5e-3, scaling_type='Logarithmic'),
        # 'batch_size': CategoricalParameter([256, 512, 1024]),
        # 'epochs': IntegerParameter(10, 11), # 충분히 큰 에포크 범위
        # 'dropout': ContinuousParameter(0.08, 0.18),
        # 'weight_decay': ContinuousParameter(1e-7, 1e-5, scaling_type='Logarithmic'),
        # 'num_layers': IntegerParameter(2,4),
    },
    metric_definitions=[ # CloudWatch에서 지표를 파싱하기 위한 정규식 (필수!)
        # {'Name': 'validation_loss', 'Regex': 'validation_loss: ([0-9\\.eE\\-]+)'},
        {'Name': 'train_loss', 'Regex': 'train_loss: ([0-9\\.eE\\-]+)'},
        {"Name": "HR@20", "Regex": "HR@20=([0-9\\.]+)"},
        {"Name": "NDCG@20", "Regex": "NDCG@20=([0-9\\.]+)"}
    ],
    max_jobs=10, # 총 하이퍼파라미터 튜닝 작업 수 (트라이얼 수)
    max_parallel_jobs=1, # 동시에 실행될 최대 병렬 작업 수
    # strategy='Bayesian', # 기본값은 Bayesian, Random도 가능
    early_stopping_type='Auto',
    warm_start_config=warm_start_config # 여기서 WarmStartConfi
)

print("SageMaker single training job을 시작합니다.", flush=True)

tuner.fit({'training': training_data_path}, wait=True) # wait=False로 설정하면 비동기로 실행

print("SageMaker single training job이 완료되었습니다.")