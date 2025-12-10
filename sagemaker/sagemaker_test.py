# from sagemaker.sklearn.processing import SKLearnProcessor
# from sagemaker.processing import ProcessingInput, ProcessingOutput
import sagemaker
from sagemaker.pytorch.processing import PyTorchProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput


role = sagemaker.get_execution_role() # SageMaker 실행 역할 (IAM 역할)
sess = sagemaker.Session() # SageMaker 세션
pytorch_processor = PyTorchProcessor(
    framework_version="2.3.0", # Specify a PyTorch version that supports your code
    py_version='py311',     # Specify the Python version
    role=role,
    instance_type='ml.g5.xlarge',
    #instance_type="ml.g4dn.xlarge", # This GPU instance type is supported
    instance_count=1,
    base_job_name="gowall-omega-2-50-full-ranking-test"
)



# 베스트 체크포인트 모델 파일
input_model = ProcessingInput(
    source="s3://ml-checkpoints-hy/gowalla-lightgcn-tail-omega2/model.tar.gz",
    destination="/opt/ml/processing/input/model"
)

# 테스트 데이터 파일
input_test_data = ProcessingInput(
    source="s3://sagemaker-ap-northeast-2-718159740639/sagemaker/bpr-hp-tuning/train_data/gowalla_all.npz",
    destination="/opt/ml/processing/input/data"
)

output_data = ProcessingOutput(source="/opt/ml/processing/output/", 
                               destination="s3://ml-checkpoints-hy/gowalla-lightgcn-tail-omega2/")

pytorch_processor.run(
        source_dir="./src/",
        code="evaluate.py",
        inputs=[input_model, input_test_data],
        outputs=[output_data]
)