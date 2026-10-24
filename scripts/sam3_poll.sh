#!/bin/bash
# Polls Hugging Face until facebook/sam3 access is granted, then downloads the transformers weights.
HF=/home/nail/stuff/ideas/action-index/.venv/bin/hf
T=$(cat ~/.cache/huggingface/token)
until [ "$(curl -s -L -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $T" https://huggingface.co/facebook/sam3/resolve/main/config.json)" = "200" ]; do sleep 60; done
echo "access granted $(date +%H:%M)"
$HF download facebook/sam3 --exclude 'sam3.pt' && echo SAM3_DONE
