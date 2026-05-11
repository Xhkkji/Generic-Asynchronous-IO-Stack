#!/usr/bin/env bash

set -euo pipefail

PYTHON_BIN=/home/xhk/miniconda3/envs/pytorch/bin/python
REPO_ROOT=/home/xhk/hyperion/GIDS

INPUT_ROOT="${INPUT_ROOT:-${REPO_ROOT}/dataset/cluecorpus2020/raw}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/dataset/cluecorpus2020/bert_base_seq256}"
MODEL_NAME="${MODEL_NAME:-bert-base-chinese}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_TRAIN_ITERS="${MAX_TRAIN_ITERS:-200}"
EPOCHS="${EPOCHS:-1}"
ENABLE_PROFILE="${ENABLE_PROFILE:-0}"

mkdir -p "${REPO_ROOT}/llm_evaluation/logs"

echo "[CLUE_BERT_COMPARE] Step 1/4: prepare CLUECorpus2020 fixed-length token chunks"
"${PYTHON_BIN}" "${REPO_ROOT}/llm_evaluation/prepare_cluecorpus2020_bert_mlm_dataset.py" \
  --input-root "${INPUT_ROOT}" \
  --output-root "${DATA_ROOT}" \
  --model-name-or-path "${MODEL_NAME}" \
  --seq-len 256 \
  --validation-ratio 0.01

echo "[CLUE_BERT_COMPARE] Step 2/4: torch mmap baseline"
"${PYTHON_BIN}" "${REPO_ROOT}/llm_evaluation/llmids_train_bert_mlm.py" \
  --train-root "${DATA_ROOT}/train" \
  --val-root "${DATA_ROOT}/val" \
  --model-name-or-path "${MODEL_NAME}" \
  --io-mode torch \
  --torch-read-mode mmap \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --max-train-iters "${MAX_TRAIN_ITERS}" \
  --run-val 1 \
  --enable-profile "${ENABLE_PROFILE}" \
  --profile-dir "${REPO_ROOT}/llm_evaluation/profile_clue_torch_mmap" \
  | tee "${REPO_ROOT}/llm_evaluation/logs/clue_bert_torch_mmap.log"

echo "[CLUE_BERT_COMPARE] Step 3/4: load prepared tokens to BaM once for sync/registered"
"${PYTHON_BIN}" "${REPO_ROOT}/llm_evaluation/llmids_load_prepared_tokens_to_bam.py" \
  --train-root "${DATA_ROOT}/train" \
  --val-root "${DATA_ROOT}/val"

echo "[CLUE_BERT_COMPARE] Step 4/4: LLMIDS sync baseline"
"${PYTHON_BIN}" "${REPO_ROOT}/llm_evaluation/llmids_train_bert_mlm.py" \
  --train-root "${DATA_ROOT}/train" \
  --val-root "${DATA_ROOT}/val" \
  --model-name-or-path "${MODEL_NAME}" \
  --io-mode sync \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --max-train-iters "${MAX_TRAIN_ITERS}" \
  --run-val 1 \
  --enable-profile "${ENABLE_PROFILE}" \
  --profile-dir "${REPO_ROOT}/llm_evaluation/profile_clue_sync" \
  | tee "${REPO_ROOT}/llm_evaluation/logs/clue_bert_llmids_sync.log"

echo "[CLUE_BERT_COMPARE] Step 5/5: LLMIDS registered baseline"
"${PYTHON_BIN}" "${REPO_ROOT}/llm_evaluation/llmids_train_bert_mlm.py" \
  --train-root "${DATA_ROOT}/train" \
  --val-root "${DATA_ROOT}/val" \
  --model-name-or-path "${MODEL_NAME}" \
  --io-mode registered \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --max-train-iters "${MAX_TRAIN_ITERS}" \
  --run-val 1 \
  --enable-profile "${ENABLE_PROFILE}" \
  --profile-dir "${REPO_ROOT}/llm_evaluation/profile_clue_registered" \
  | tee "${REPO_ROOT}/llm_evaluation/logs/clue_bert_llmids_registered.log"
