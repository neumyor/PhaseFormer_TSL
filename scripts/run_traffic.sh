#!/bin/bash

# 基础配置
TASK_NAME="long_term_forecast"
MODEL="PhaseFormer"
MODEL_ID="phaseformer_traffic"
DATA="custom"
ROOT_PATH="../TimeSeriesSim/resources/all_datasets/traffic/"
DATA_PATH="traffic.csv"
SEQ_LEN=720
TRAIN_EPOCHS=30
BATCH_SIZE=16
USE_GPU=1
GPU=0
USE_HUBER=1
HUBER_DELTA=1.0
USE_REVIN=1
REVIN_AFFINE=0
REVIN_EPS=1e-5

# 四种预测长度
for PRED_LEN in 96 192 336 720
do
  if [ "$PRED_LEN" -eq 96 ]; then
    LR=0.001
    LAYERS=2
    LATENT_DIM=32
    ENCODER_HIDDEN=64
    PREDICTOR_HIDDEN=128
    NUM_ROUTERS=1
    ATT_HEADS=8
  else
    LR=0.001
    LAYERS=1
    LATENT_DIM=128
    ENCODER_HIDDEN=16
    PREDICTOR_HIDDEN=32
    NUM_ROUTERS=4
    ATT_HEADS=4
  fi

  echo "🚀 开始训练: PRED_LEN=$PRED_LEN, LR=$LR, Layers=$LAYERS, Latent=$LATENT_DIM"

  python run.py \
    --task_name $TASK_NAME \
    --is_training 1 \
    --model_id "${MODEL_ID}_pl${PRED_LEN}" \
    --model $MODEL \
    --data $DATA \
    --root_path $ROOT_PATH \
    --data_path $DATA_PATH \
    --seq_len $SEQ_LEN \
    --pred_len $PRED_LEN \
    --train_epochs $TRAIN_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --use_revin $USE_REVIN \
    --revin_affine $REVIN_AFFINE \
    --revin_eps $REVIN_EPS \
    --use_huber $USE_HUBER \
    --huber_delta $HUBER_DELTA \
    --extra_tag "layers${LAYERS}-latent${LATENT_DIM}-enc${ENCODER_HIDDEN}-pred${PREDICTOR_HIDDEN}-routers${NUM_ROUTERS}-heads${ATT_HEADS}"
done