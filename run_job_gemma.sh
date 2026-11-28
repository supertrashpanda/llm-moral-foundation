#!/bin/bash
#SBATCH --account=open
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:40:00
#SBATCH --job-name=moral_gemma
#SBATCH --mem=64G
#SBATCH --output=job_%j.log

module load miniforge/25.3.0-3
source activate vllm_gemma
cd /storage/work/ykl5847/moral_coding

vllm serve /storage/work/ykl5847/gemma-4-31B-it-qat-w4a16-ct \
  --served-model-name gemma-4-31b-it-w4a16 \
  --max-model-len 8192 --port 8000 --seed 502 > vllm_gemma.log 2>&1 &
SERVER_PID=$!

# wait up to 20 min for the server; stop early (and show why) if vLLM has crashed
READY=0
for i in $(seq 1 120); do
  if curl -s localhost:8000/v1/models > /dev/null; then READY=1; break; fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "vLLM exited early. Last lines of vllm_gemma.log:"; tail -n 40 vllm_gemma.log; exit 1
  fi
  sleep 10
done
if [ $READY -ne 1 ]; then
  echo "vLLM not ready after 20 min. Last lines of vllm_gemma.log:"; tail -n 40 vllm_gemma.log
  kill $SERVER_PID; exit 1
fi

python gemma_starter.py test
kill $SERVER_PID
