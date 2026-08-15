#!/bin/bash
# boot.sh — own-engine pod provisioning, fetched from the site by the pod's
# dockerArgs at start (gpu-desk op:'engine' start). Builds the MuseTalk 1.5
# environment (~18 min cold), downloads presence.mp4 + worker.py from this
# site, then starts the resident render worker. Progress: /workspace/state.json
# (phases: apt, clone, pip, weights, assets, worker-loading-models,
# worker-preparing-avatar, ready). Log: /workspace/boot.log.
set -x
cd /workspace
SITE=https://factory-simple-avatar-site.netlify.app

state() {
  python3 - "$1" <<'PY'
import json, sys, time
json.dump({'phase': sys.argv[1], 'ready': False, 't': time.time()}, open('/workspace/state.json', 'w'))
PY
}

state apt
apt-get update -qq && apt-get install -y -qq ffmpeg libgl1 git wget > /dev/null

state clone
git clone --depth 1 https://github.com/TMElyralab/MuseTalk.git
cd /workspace/MuseTalk

state pip
grep -v -i '^torch' requirements.txt > req_notorch.txt
pip install -q -r req_notorch.txt 2>&1 | tail -3
pip install -q -U openmim 2>&1 | tail -1
mim install mmengine "mmcv==2.1.0" "mmdet>=3.1.0" "mmpose>=1.1.0" 2>&1 | tail -3

state weights
pip install -q -U "huggingface_hub[hf_transfer]" 2>&1 | tail -1
export HF_HUB_ENABLE_HF_TRANSFER=1
python3 - <<'PY'
from huggingface_hub import snapshot_download, hf_hub_download
import os
os.makedirs('models', exist_ok=True)
snapshot_download('TMElyralab/MuseTalk', local_dir='models', allow_patterns=['musetalkV15/*', 'musetalk/*'])
print('musetalk weights ok')
snapshot_download('stabilityai/sd-vae-ft-mse', local_dir='models/sd-vae', allow_patterns=['config.json', 'diffusion_pytorch_model.bin'])
print('vae ok')
snapshot_download('openai/whisper-tiny', local_dir='models/whisper')
print('whisper ok')
hf_hub_download('yzd-v/DWPose', 'dw-ll_ucoco_384.pth', local_dir='models/dwpose')
print('dwpose ok')
hf_hub_download('ManyOtherFunctions/face-parse-bisent', '79999_iter.pth', local_dir='models/face-parse-bisent')
print('bisent ok')
PY
wget -q https://download.pytorch.org/models/resnet18-5c106cde.pth -O models/face-parse-bisent/resnet18-5c106cde.pth
# huggingface_hub 1.x breaks transformers 4.39 — the proven pin:
pip install -q huggingface_hub==0.25.2 2>&1 | tail -1

state assets
wget -q "$SITE/engine/presence.mp4" -O /workspace/presence.mp4
wget -q "$SITE/engine/worker.py" -O /workspace/worker.py
mkdir -p /workspace/jobs /workspace/out

# The worker sets state.json to ready:true itself once models + avatar are loaded.
cd /workspace/MuseTalk
nohup python3 /workspace/worker.py > /workspace/worker.log 2>&1 &
echo "BOOT DONE rc=$?"
