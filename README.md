long-tail-gnn-recsys

## 1. Overview
본 프로젝트는 GNN 기반 추천 시스템에서 발생하는 롱테일 아이템 학습 부족 문제를 해결하기 위해
아이템의 네트워크 중심성(DC/BC)에 기반한 Tail-Aware Negative Sampling 전략을 적용한 모델 구현 프로젝트입니다.

기존의 균일 샘플링 방식은 상위 인기 아이템에 학습이 집중되어
테일 아이템의 표현 공간이 매우 빈약해지는 문제가 있습니다.
본 연구에서는 중심성 기반 테일 샘플링을 활용하여 테일 아이템의 학습 기회를 증가시키는 것을 목표로 한다.

## 2. Features

LightGCN 기반 임베딩 모델 구현

DC(Degree Centrality) / BC(Betweenness Centrality) 기반 아이템 중심성 계산

Tail-aware Negative Sampling 전략 적용

Gowalla Dataset, Animation 기반 실험

PyTorch Lightning + PyTorch Geometric로 모델 구조 구현

Optuna 기반 하이퍼파라미터 튜닝 적용

EC2에서 clean environment 재현성 테스트 완료

