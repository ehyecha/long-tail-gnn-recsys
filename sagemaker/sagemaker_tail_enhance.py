import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.estimator import Estimator
import os
from sagemaker.tuner import HyperparameterTuner, IntegerParameter, ContinuousParameter,  CategoricalParameter
from sagemaker.tuner import WarmStartConfig, WarmStartTypes

role = sagemaker.get_execution_role()
sess = sagemaker.Session()
bucket = sess.default_bucket()
prefix = 'sagemaker/bpr-hp-tuning' # S3 버킷 내의 경로


training_data_path = f's3://{bucket}/{prefix}/train_data' # S3 경로만 정의

current_script_dir = os.path.dirname(os.path.abspath(__file__))
source_dir_path = os.path.join(current_script_dir, 'src')

checkpoint_s3_uri = f's3://{bucket}/{prefix}/checkpoints/gowalla-ligtgcn-tail-enhance-model-tuning/'
checkpoint_local_path="/opt/ml/checkpoints"

parent_job_id ="anime-tail-lcc-251108-0615"
warm_start_config = WarmStartConfig(
    parents=[parent_job_id],
    warm_start_type=WarmStartTypes.IDENTICAL_DATA_AND_ALGORITHM
)

estimator = PyTorch(
    role=role,
    framework_version='2.3.0', # PyTorch 버전
    py_version='py311', # Python 버전
    instance_count=1,
    instance_type='ml.g5.xlarge',
    #instance_type='ml.g4dn.xlarge',
    source_dir=source_dir_path,  # train.py가 있는 로컬 디렉토리
    entry_point='train_tail_enhance.py', # 실행할 스크립트 파일명
    use_spot_instances=True,
    max_run=3600 * 12,  # 초 단위로 최대 4시간 실행
    max_wait=3600 * 13, # 초 단위로 최대 5시간 대기 (max_run보다 길거나 같아야 함)
    checkpoint_s3_uri=checkpoint_s3_uri,
    environment={'SM_CHECKPOINT_DIR': checkpoint_local_path, "WANDB_API_KEY": "307711fe800b379c13bbf3de8a153fb991d6e4af"},
    dependencies=['./src/requirements.txt'],
    hyperparameters={
        # "alpha": 0.9,
        # "omega": 0.5,
        # "dropout": 0.13340041478950637,
        # "lr": 0.00032520998331792185,
        # "weight_decay": 5.054741164066878e-06,
        # "embedding_dim": 80,
        # "batch_size": 32,
        # "num_layers": 4,
        # "embedding_dim": 64,    # Another fixed parameter
        # "lr": 0.003316329630716579,
        # "weight_decay": 4.347174837825076e-06,
        # "num_layers":3,
        # "dropout":	0.016266682301975397,
        # 'embedding_dim': 32, 
        # 'batch_size': 128, 
        # 'lr': 0.009047629368429547, 
        # 'weight_decay': 1.944682686673428e-06, 
        # 'num_layers': 1,
        # 'dropout': 0.23037612381604228,
        # "beta": 1,
        # "alpha": 0.7
    }
)




#hypterparameter tuning
# # # 4. 하이퍼파라미터 튜닝 정의 (Optuna의 Study.optimize() 역할)
tuner = HyperparameterTuner(
    estimator, # 위에서 정의한 Estimator
    base_tuning_job_name='anime-tail-mixed',
    # objective_metric_name='validation_loss', # 학습 스크립트에서 Trainer.log로 기록하는 지표 이름
    # objective_type='Minimize', # 이 지표를 최대화할지 최소화할지
    objective_metric_name='custom_score', # 학습 스크립트에서 Trainer.log로 기록하는 지표 이름
    # objective_metric_name="NDCG_TAIL@20", # Tail 지표 최적화
    objective_type='Maximize', # 이 지표를
    hyperparameter_ranges={ # 탐색할 하이퍼파라미터 범위
        #gowalla 
        # 'embedding_dim': CategoricalParameter([48, 64, 80]),         # 범위 줄임
        # 'lr': ContinuousParameter(0.0003, 0.0007, scaling_type='Logarithmic'),
        # 'batch_size': CategoricalParameter([32, 64]),
        # 'epochs': IntegerParameter(15, 20), # 충분히 큰 에포크 범위
        # 'dropout': ContinuousParameter(0.1, 0.3), 
        # 'weight_decay': ContinuousParameter(1e-6, 1e-4, scaling_type='Logarithmic'),
        # 'num_layers': IntegerParameter(3, 4),
        # 'omega':  CategoricalParameter([0.5, 1, 1.5, 2]),
        # 'alpha': CategoricalParameter([ 0.7, 0.8, 0.9]),
        # 'beta': CategoricalParameter([0.1, 0.3, 0.5, 0.7, 0.9])
        
        #movielense
        #   'batch_size': CategoricalParameter([64, 128, 256]),
        #   'dropout': ContinuousParameter(0.15, 0.35),
        #   'embedding_dim': CategoricalParameter([64, 128, 256]),
        #   'epochs': IntegerParameter(50, 51), # 충분히 큰 에포크 범위
        #   'lr': ContinuousParameter(0.0008, 0.0032, scaling_type='Logarithmic'),
        #   'num_layers': IntegerParameter(1, 3),
        #   'weight_decay': ContinuousParameter(1e-6, 5e-6, scaling_type='Logarithmic'),
        #   'omega':  CategoricalParameter([0.5, 1, 1.5, 2]),
        #   'alpha': CategoricalParameter([ 0.7, 0.8, 0.9]),
        #   'beta': CategoricalParameter([0.1, 0.3, 0.5, 0.7, 0.9])

        # 'embedding_dim': CategoricalParameter([96, 128]),
        # 'lr': ContinuousParameter(7e-4, 1.5e-3, scaling_type='Logarithmic'),
        # 'batch_size': CategoricalParameter([256, 512, 1024]),
        # 'epochs': IntegerParameter(15, 20), # 충분히 큰 에포크 범위
        # # 'dropout': ContinuousParameter(0.15, 0.35),
        # # 'weight_decay': ContinuousParameter(1e-6, 5e-5, scaling_type='Logarithmic'),
        # # 'num_layers': IntegerParameter(2,4),
        # # 'omega':  CategoricalParameter([0.5, 1, 1.5, 2]),
        # # 'alpha': CategoricalParameter([0.4, 0.5, 0.6, 0.7]),
        # # 'beta': CategoricalParameter([0.1, 0.3, 0.5, 0.7, 0.9])
        #   'alpha': CategoricalParameter([0.3, 0.4, 0.5, 0.6]),
        # #   'beta': CategoricalParameter([0.1, 0.3, 0.5, 0.7, 0.9]),
        #   'omega':  CategoricalParameter([0.5, 1, 1.2,1.5]),
        # #    "dropout": ContinuousParameter(0.0, 0.3),
        # #    "lr": ContinuousParameter(1e-4, 5e-3),
        #   "batch_size": CategoricalParameter([64, 128, 256]),
        #    "weight_decay" : ContinuousParameter(1e-6, 1e-4, scaling_type='Logarithmic'),
        
        # 'embedding_dim': CategoricalParameter([64, 192]),
        # 'lr': ContinuousParameter(3e-4, 1.2e-3, scaling_type='Logarithmic'),
        # 'batch_size': CategoricalParameter([128, 512]),
        # 'epochs': IntegerParameter(10, 11), # 충분히 큰 에포크 범위
        # 'dropout': ContinuousParameter(0.08, 0.13),
        # 'weight_decay': ContinuousParameter(1e-7, 1e-5, scaling_type='Logarithmic'),
        # 'num_layers': IntegerParameter(3,5),
        # 'omega':  CategoricalParameter([1.5, 3]),
        # 'alpha': CategoricalParameter([0.6,1]),
        # 'beta': CategoricalParameter([0.1, 0.3, 0.5, 0.7, 0.9])

        #anime
        'embedding_dim': CategoricalParameter([32, 48, 64, 80, 96]),         # 범위 줄임
        'lr': ContinuousParameter(0.001, 0.01, scaling_type='Logarithmic'),
        'batch_size': CategoricalParameter([128, 256]),
        'epochs': IntegerParameter(50, 60), # 충분히 큰 에포크 범위
        'dropout': ContinuousParameter(0.18, 0.4), 
        'weight_decay': ContinuousParameter(1e-6, 5e-5, scaling_type='Logarithmic'),
        'num_layers': IntegerParameter(1,3),
        'omega':  CategoricalParameter([0.5, 1, 1.2, 1.5]),
        'alpha': CategoricalParameter([0.7, 0.8, 0.9]),
        'beta': CategoricalParameter([0.8, 0.9])
        
        # gowalla
        # 'embedding_dim': CategoricalParameter([64, 128, 256]),
        # 'lr': ContinuousParameter(1e-3, 1e-2, scaling_type='Logarithmic'),
        # 'batch_size': CategoricalParameter([32, 64, 128]),
        # 'epochs': IntegerParameter(29, 30), # 충분히 큰 에포크 범위
        # 'dropout': CategoricalParameter([0, 0.05, 0.1, 0.2, 0.3]),
        # 'weight_decay': ContinuousParameter(1e-6, 1e-4, scaling_type='Logarithmic'),
        # 'num_layers': IntegerParameter(2,4),
        # 'omega':  CategoricalParameter([0.5, 1.2, 1, 1.5]),
        # 'alpha': CategoricalParameter([0.7,0.8, 0.9]),
        # 'beta': CategoricalParameter([0.1, 0.3, 0.5, 0.7, 0.9])

        #yelp2018
        # 'embedding_dim': CategoricalParameter([64, 96, 128]),
        # 'lr': ContinuousParameter(5e-4, 1.5e-3, scaling_type='Logarithmic'),
        # 'batch_size': CategoricalParameter([256, 512, 1024]),
        # 'epochs': IntegerParameter(10, 11), # 충분히 큰 에포크 범위
        # 'dropout': ContinuousParameter(0.08, 0.18),
        # 'weight_decay': ContinuousParameter(1e-7, 1e-5, scaling_type='Logarithmic'),
        # 'num_layers': IntegerParameter(2,4),
        # 'alpha': CategoricalParameter([0.7,0.8, 0.9]),
        # 'beta': CategoricalParameter([0.1, 0.3, 0.5, 0.7, 0.9])


    },
    metric_definitions=[ 
        # CloudWatch에서 지표를 파싱하기 위한 정규식 (필수!)
        # {'Name': 'validation_loss', 'Regex': 'validation_loss: ([0-9\\.eE\\-]+)'},
        {'Name': 'train_loss', 'Regex': 'train_loss: ([0-9\\.eE\\-]+)'},
        {"Name": "HR@20", "Regex": "HR@20=([0-9\\.]+)"},
        {"Name": "NDCG@20", "Regex": "NDCG@20=([0-9\\.]+)"},
        {"Name": "NDCG_TAIL@20", "Regex": "NDCG_TAIL@20=([0-9\\.]+)"},
        {"Name": "HR_TAIL@20", "Regex": "HR_TAIL@20=([0-9\\.]+)"},
        {"Name": "custom_score", "Regex": "custom_score=([0-9\\.]+)"}
        # 다른 로깅된 지표도 추가 가능
    ],
    max_jobs=10, # 총 하이퍼파라미터 튜닝 작업 수 (트라이얼 수)
    max_parallel_jobs=1, # 동시에 실행될 최대 병렬 작업 수
    # strategy='Bayesian', # 기본값은 Bayesian, Random도 가능
    early_stopping_type='Auto',
    # warm_start_config=warm_start_config # 여기서 WarmStartConfi
)

print("SageMaker single training job을 시작합니다.", flush=True)

tuner.fit({'training': training_data_path}, wait=True) # wait=False로 설정하면 비동기로 실행

print("SageMaker single training job이 완료되었습니다.")