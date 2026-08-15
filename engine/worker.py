# worker.py — resident MuseTalk 1.5 render worker for the Factory own-engine pod.
# Started by engine/boot.sh after provisioning. Loads all models ONCE, prepares
# (or loads) the avatar ONCE, then loops: each /workspace/jobs/<id>.mp3 becomes
# /workspace/out/<id>.mp4 (720x720, 25 fps, single H.264 encode, audio muxed).
# Progress is reported in /workspace/state.json; per-job failures land in
# /workspace/out/<id>.err. This file is fetched by the pod from the site at boot.
# Working code (verified end to end on a live pod before commit).
import os
import sys
import json
import time
import glob
import shutil
import subprocess
import traceback

MT = '/workspace/MuseTalk'
os.chdir(MT)
sys.path.insert(0, MT)


def state(phase, ready, extra=None):
    d = {'phase': phase, 'ready': ready, 't': time.time()}
    if extra:
        d.update(extra)
    tmp = '/workspace/state.json.tmp'
    with open(tmp, 'w') as f:
        json.dump(d, f)
    os.replace(tmp, '/workspace/state.json')


state('worker-loading-models', False)

from types import SimpleNamespace  # noqa: E402
import copy  # noqa: E402
import queue  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import scripts.realtime_inference as ri  # noqa: E402

# The globals scripts/realtime_inference.py normally builds under __main__ —
# built here exactly once, then injected into the module so its Avatar class
# (prepare_material / inference) finds them.
args = SimpleNamespace(
    version='v15', gpu_id=0, vae_type='sd-vae',
    unet_config='./models/musetalkV15/musetalk.json',
    unet_model_path='./models/musetalkV15/unet.pth',
    whisper_dir='./models/whisper',
    bbox_shift=0, extra_margin=10, fps=25,
    audio_padding_length_left=2, audio_padding_length_right=2,
    batch_size=20, parsing_mode='jaw',
    left_cheek_width=90, right_cheek_width=90,
    skip_save_images=False,
)
ri.args = args

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ri.device = device
vae, unet, pe = ri.load_all_model(
    unet_model_path=args.unet_model_path,
    vae_type=args.vae_type,
    unet_config=args.unet_config,
    device=device,
)
ri.timesteps = torch.tensor([0], device=device)
ri.pe = pe.half().to(device)
vae.vae = vae.vae.half().to(device)
ri.vae = vae
unet.model = unet.model.half().to(device)
ri.unet = unet
ri.audio_processor = ri.AudioProcessor(feature_extractor_path=args.whisper_dir)
ri.weight_dtype = unet.model.dtype
whisper = ri.WhisperModel.from_pretrained(args.whisper_dir)
ri.whisper = whisper.to(device=device, dtype=ri.weight_dtype).eval()
ri.whisper.requires_grad_(False)
ri.fp = ri.FaceParsing(
    left_cheek_width=args.left_cheek_width,
    right_cheek_width=args.right_cheek_width,
)

# Same compositing as upstream Avatar.process_frames, but frames go to disk
# as JPEG (quality 95) instead of PNG — several seconds faster per sentence
# on the 4090; the H.264 encode at CRF 18 dominates quality anyway. Note:
# ri.Avatar is wrapped by a class-level @torch.no_grad() decorator, so it is
# a function wrapper, not a class — we patch the method on the instance's
# real class after construction (see below).
def fast_process_frames(self, res_frame_queue, video_len, skip_save_images):
    while True:
        if self.idx >= video_len - 1:
            break
        try:
            res_frame = res_frame_queue.get(block=True, timeout=1)
        except queue.Empty:
            continue
        bbox = self.coord_list_cycle[self.idx % (len(self.coord_list_cycle))]
        ori_frame = copy.deepcopy(self.frame_list_cycle[self.idx % (len(self.frame_list_cycle))])
        x1, y1, x2, y2 = bbox
        try:
            res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
        except Exception:
            continue
        mask = self.mask_list_cycle[self.idx % (len(self.mask_list_cycle))]
        mask_crop_box = self.mask_coords_list_cycle[self.idx % (len(self.mask_coords_list_cycle))]
        combine_frame = ri.get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)
        if skip_save_images is False:
            cv2.imwrite(self.avatar_path + '/tmp/' + str(self.idx).zfill(8) + '.jpg',
                        combine_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        self.idx = self.idx + 1


AVATAR_ID = 'factory_her'
avatar_dir = './results/v15/avatars/' + AVATAR_ID
prep = not os.path.exists(avatar_dir)
state('worker-preparing-avatar' if prep else 'worker-loading-avatar', False)
avatar = ri.Avatar(
    avatar_id=AVATAR_ID,
    video_path='/workspace/presence.mp4',
    bbox_shift=0,
    batch_size=args.batch_size,
    preparation=prep,
)
type(avatar).process_frames = fast_process_frames   # JPEG frame writes

JOBS = '/workspace/jobs'
OUT = '/workspace/out'
os.makedirs(JOBS, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
state('ready', True)
print('worker ready', flush=True)


def render(job_id, mp3_path):
    wav = OUT + '/' + job_id + '.wav'
    subprocess.run(
        ['ffmpeg', '-y', '-v', 'error', '-i', mp3_path, '-ar', '16000', '-ac', '1', wav],
        check=True,
    )
    tmp_dir = avatar.avatar_path + '/tmp'
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
    # out_vid_name=None: frames are written to tmp/ and the script's own
    # two-pass mux is skipped — we encode ONCE below (avoids double compression).
    avatar.inference(wav, None, args.fps, False)
    out_tmp = OUT + '/' + job_id + '.tmp.mp4'
    subprocess.run(
        ['ffmpeg', '-y', '-v', 'error', '-r', str(args.fps),
         '-i', tmp_dir + '/%08d.jpg', '-i', wav,
         '-c:v', 'libx264', '-crf', '18', '-preset', 'veryfast',
         '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '160k', '-shortest', out_tmp],
        check=True,
    )
    os.replace(out_tmp, OUT + '/' + job_id + '.mp4')
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.remove(wav)


while True:
    jobs = sorted(glob.glob(JOBS + '/*.mp3'), key=os.path.getmtime)
    if not jobs:
        time.sleep(0.4)
        continue
    p = jobs[0]
    jid = os.path.splitext(os.path.basename(p))[0]
    t0 = time.time()
    try:
        state('rendering:' + jid, True)
        render(jid, p)
        state('ready', True, {'last_job': jid, 'last_secs': round(time.time() - t0, 2)})
        print('job', jid, 'done in', round(time.time() - t0, 2), 's', flush=True)
    except Exception as e:
        with open(OUT + '/' + jid + '.err', 'w') as f:
            f.write(str(e)[:500])
        traceback.print_exc()
        state('ready', True, {'last_err': jid})
    finally:
        try:
            os.remove(p)
        except OSError:
            pass
