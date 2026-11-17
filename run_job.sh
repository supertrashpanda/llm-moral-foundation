#!/bin/bash
#SBATCH --account=open
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --job-name=moral_qwen32b_test
#SBATCH --output=job_%j.log

module load miniforge/25.3.0-3
source activate vllm_env
cd /storage/work/ykl5847/moral_coding

vllm serve /storage/work/ykl5847/Qwen2.5-32B-Instruct-AWQ \
  --served-model-name qwen2.5-32b-awq --quantization awq \
  --max-model-len 8192 --port 8000 --seed 502 > vllm.log 2>&1 &
SERVER_PID=$!

for i in $(seq 1 90); do
  curl -s localhost:8000/v1/models > /dev/null && break
  sleep 10
done

python qwen_starter.py test
kill $SERVER_PID
