FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y git

# 작업 디렉토리
WORKDIR /app

# requirements 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# PyTorch Geometric 설치 (CUDA 11.8 / torch 2.1.0 대응 버전)
RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv pyg-lib torch-geometric \
    --find-links https://data.pyg.org/whl/torch-2.1.0+cu118.html
# 소스코드 복사
COPY . .

# 기본 실행명령
ENTRYPOINT ["python", "train.py"]

CMD []
