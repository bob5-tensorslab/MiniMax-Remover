import os
import gradio as gr
import cv2
import numpy as np
from PIL import Image

os.makedirs("./SAM2-Video-Predictor/checkpoints/", exist_ok=True)
os.makedirs("./model/", exist_ok=True)

from huggingface_hub import snapshot_download

def download_sam2():
    snapshot_download(repo_id="facebook/sam2-hiera-large", local_dir="./SAM2-Video-Predictor/checkpoints/")
    print("Download sam2 completed")

def download_remover():
    snapshot_download(repo_id="zibojia/minimax-remover", local_dir="./model/")
    print("Download minimax remover completed")

download_sam2()
download_remover()

import torch
import argparse
import random

import torch.nn.functional as F
import time
import random
from omegaconf import OmegaConf
from einops import rearrange
from diffusers.models import AutoencoderKLWan
import scipy
from transformer_minimax_remover import Transformer3DModel
from einops import rearrange
from diffusers.schedulers import UniPCMultistepScheduler
from pipeline_minimax_remover import Minimax_Remover_Pipeline

from diffusers.utils import export_to_video
from decord import VideoReader, cpu
from moviepy.editor import ImageSequenceClip

from sam2 import load_model

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

COLOR_PALETTE = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (255, 128, 0), (128, 0, 255), (0, 128, 255), (128, 255, 0)
]

random_seed = 42
video_length = 201
W = 1024
H = W
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_pipe_image_and_video_predictor():
    vae = AutoencoderKLWan.from_pretrained("./model/vae", torch_dtype=torch.float16)
    transformer = Transformer3DModel.from_pretrained("./model/transformer", torch_dtype=torch.float16)
    scheduler = UniPCMultistepScheduler.from_pretrained("./model/scheduler")

    pipe = Minimax_Remover_Pipeline(transformer=transformer, vae=vae, scheduler=scheduler)
    pipe.to(device)

    sam2_checkpoint = "./SAM2-Video-Predictor/checkpoints/sam2_hiera_large.pt"
    config = "sam2_hiera_l.yaml"

    video_predictor = build_sam2_video_predictor(config, sam2_checkpoint, device=device)
    model = build_sam2(config, sam2_checkpoint, device=device)
    model.image_size = 1024
    image_predictor = SAM2ImagePredictor(sam_model=model)

    return pipe, image_predictor, video_predictor

def get_video_fps(video_reader):
    try:
        fps = float(video_reader.get_avg_fps())
    except (AttributeError, TypeError, ValueError):
        fps = 15.0
    return fps if np.isfinite(fps) and fps > 0 else 15.0


def read_uploaded_mask_video(mask_video_path, frame_count, source_fps, width, height):
    mask_reader = VideoReader(mask_video_path, ctx=cpu(0))
    mask_fps = get_video_fps(mask_reader)
    mask_indices = np.rint(
        np.arange(frame_count, dtype=np.float64) * mask_fps / source_fps
    ).astype(np.int64)
    if len(mask_indices) and mask_indices[-1] >= len(mask_reader):
        gr.Warning(
            "掩码视频时长不足，无法覆盖当前源视频和选择的帧数。"
            "请上传更长的掩码视频，或减少跟踪帧数 N。"
        )
        return None

    masks = []
    for mask_idx in mask_indices:
        mask_frame = mask_reader[int(mask_idx)].asnumpy()
        # Treat bright pixels in any channel as foreground so both white masks
        # and colored annotation masks can be used.
        binary_mask = np.any(mask_frame > 127, axis=2).astype(np.uint8)
        binary_mask = cv2.resize(
            binary_mask, (width, height), interpolation=cv2.INTER_NEAREST
        )
        masks.append(np.repeat(binary_mask[:, :, None], 3, axis=2).astype(np.float32))
    del mask_reader
    return masks


def build_fixed_mask_video(painted_data, video_path, n_frames, video_state):
    if not video_path:
        return None, "请先上传源视频。"
    if video_state.get("video_path") not in (None, video_path):
        return None, "源视频正在刷新，请稍后重试。"
    if not isinstance(painted_data, dict) or painted_data.get("mask") is None:
        return None, "Paint the target area on the first-frame editor first."

    painted_mask = np.asarray(painted_data["mask"])
    if painted_mask.ndim == 3:
        painted_mask = np.max(painted_mask[..., :3], axis=2)
    elif painted_mask.ndim != 2:
        return None, "Could not read the painted mask."
    mask_max = float(np.max(painted_mask))
    threshold = 0.5 if np.issubdtype(painted_mask.dtype, np.floating) and mask_max <= 1 else 127
    foreground = painted_mask > threshold
    if not np.any(foreground):
        return None, "The mask is empty. Paint the object to remove."

    source_reader = VideoReader(video_path, ctx=cpu(0))
    frame_count = min(len(source_reader), max(1, int(n_frames)))
    if frame_count < 1:
        del source_reader
        return None, "源视频不包含可用帧。"
    source_fps = get_video_fps(source_reader)
    first_frame = source_reader[0].asnumpy()
    del source_reader
    height, width = first_frame.shape[:2]

    if foreground.shape != (height, width):
        foreground = cv2.resize(
            foreground.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ) > 0

    mask_frame = foreground.astype(np.uint8) * 255
    # libx264 requires even dimensions; the reader resizes back to source size.
    mask_frame = np.pad(
        mask_frame,
        ((0, mask_frame.shape[0] % 2), (0, mask_frame.shape[1] % 2)),
        mode="constant",
    )
    mask_rgb = np.repeat(mask_frame[:, :, None], 3, axis=2)
    mask_video_file = f"/tmp/{time.time()}-{random.random()}-fixed_mask.mp4"
    mask_clip = ImageSequenceClip([mask_rgb] * frame_count, fps=source_fps)
    try:
        mask_clip.write_videofile(
            mask_video_file,
            codec="libx264",
            audio=False,
            verbose=False,
            logger=None,
            ffmpeg_params=["-frames:v", str(frame_count)],
        )
    finally:
        mask_clip.close()

    return mask_video_file, (
        f"已生成固定掩码视频：{frame_count} 帧，{source_fps:.3f} FPS。"
        "点击“移除目标”开始处理。"
    )


def get_video_info(video_path, video_state):
    video_state["input_points"] = []
    video_state["scaled_points"] = []
    video_state["input_labels"] = []
    video_state["frame_idx"] = 0
    video_state["origin_images"] = None
    video_state["inference_state"] = None
    video_state["video_path"] = None
    video_state["video_fps"] = 15.0
    video_state["masks"] = None
    video_state["painted_images"] = None
    if not video_path:
        return None
    vr = VideoReader(video_path, ctx=cpu(0))
    video_state["video_fps"] = get_video_fps(vr)
    first_frame = vr[0].asnumpy()
    del vr

    if first_frame.shape[0] > first_frame.shape[1]:
        W_ = W
        H_ = int(W_ * first_frame.shape[0] / first_frame.shape[1])
    else:
        H_ = H
        W_ = int(H_ * first_frame.shape[1] / first_frame.shape[0])

    first_frame = cv2.resize(first_frame, (W_, H_))
    video_state["origin_images"] = np.expand_dims(first_frame, axis=0)
    video_state["inference_state"] = None
    video_state["video_path"] = video_path
    video_state["masks"] = None
    video_state["painted_images"] = None
    image = Image.fromarray(first_frame)
    return image


def handle_video_change(video_path, video_state):
    image = get_video_info(video_path, video_state)
    return image, image, video_state, None, None, None

def segment_frame(evt: gr.SelectData, label, video_state):
    if video_state["origin_images"] is None:
        gr.Warning("请先点击“提取首帧”，再在图像上点选目标。")
        return None
    x, y = evt.index
    new_point = [x, y]
    label_value = 1 if label == "正向点" else 0

    video_state["input_points"].append(new_point)
    video_state["input_labels"].append(label_value)
    height, width = video_state["origin_images"][0].shape[0:2]
    scaled_points = []
    for pt in video_state["input_points"]:
        sx = pt[0] / width
        sy = pt[1] / height
        scaled_points.append([sx, sy])

    video_state["scaled_points"] = scaled_points

    image_predictor.set_image(video_state["origin_images"][0])
    mask, _, _ = image_predictor.predict(
        point_coords=video_state["scaled_points"],
        point_labels=video_state["input_labels"],
        multimask_output=False,
        normalize_coords=False,
    )

    mask = np.squeeze(mask)
    mask = cv2.resize(mask, (width, height))
    mask = mask[:,:,None]

    color = np.array(COLOR_PALETTE[int(time.time()) % len(COLOR_PALETTE)], dtype=np.float32) / 255.0
    color = color[None, None, :]
    org_image = video_state["origin_images"][0].astype(np.float32) / 255.0
    painted_image = (1 - mask * 0.5) * org_image + mask * 0.5 * color
    painted_image = np.uint8(np.clip(painted_image * 255, 0, 255))
    video_state["painted_images"] = np.expand_dims(painted_image, axis=0)
    video_state["masks"] = np.expand_dims(mask[:,:,0], axis=0)

    for i in range(len(video_state["input_points"])):
        point = video_state["input_points"][i]
        if video_state["input_labels"][i] == 0:
            cv2.circle(painted_image, point, radius=3, color=(0, 0, 255), thickness=-1)  # 红色点，半径为3
        else:
            cv2.circle(painted_image, point, radius=3, color=(255, 0, 0), thickness=-1)

    return Image.fromarray(painted_image)

def clear_clicks(video_state):
    video_state["input_points"] = []
    video_state["input_labels"] = []
    video_state["scaled_points"] = []
    video_state["inference_state"] = None
    video_state["masks"] = None
    video_state["painted_images"] = None
    return Image.fromarray(video_state["origin_images"][0]) if video_state["origin_images"] is not None else None


def preprocess_for_removal(images, masks):
    out_images = []
    out_masks = []
    for img, msk in zip(images, masks):
        if img.shape[0] > img.shape[1]:
            img_resized = cv2.resize(img, (480, 832), interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = cv2.resize(img, (832, 480), interpolation=cv2.INTER_LINEAR)
        img_resized = img_resized.astype(np.float32) / 127.5 - 1.0  # [-1, 1]
        out_images.append(img_resized)
        if msk.shape[0] > msk.shape[1]:
            msk_resized = cv2.resize(msk, (480, 832), interpolation=cv2.INTER_NEAREST)
        else:
            msk_resized = cv2.resize(msk, (832, 480), interpolation=cv2.INTER_NEAREST)
        msk_resized = msk_resized.astype(np.float32)
        msk_resized = (msk_resized > 0.5).astype(np.float32)
        out_masks.append(msk_resized)
    arr_images = np.stack(out_images)
    arr_masks = np.stack(out_masks)
    if arr_masks.ndim == 3:
        arr_masks = arr_masks[:, :, :, None]
    return torch.from_numpy(arr_images).half().to(device), torch.from_numpy(arr_masks).half().to(device)


def inference_and_return_video(
    dilation_iterations,
    num_inference_steps,
    video_path,
    mask_video_path,
    n_frames,
    video_state=None,
):
    if video_state is None:
        video_state = {}

    source_video_path = video_path or video_state.get("video_path")
    if video_path and video_state.get("video_path") not in (None, video_path):
        gr.Warning("上传的源视频已变更，请等待首帧刷新完成。")
        return None
    if source_video_path and mask_video_path:
        source_reader = VideoReader(source_video_path, ctx=cpu(0))
        source_fps = get_video_fps(source_reader)
        frame_count = min(len(source_reader), int(n_frames))
        images = [source_reader[i].asnumpy() for i in range(frame_count)]
        del source_reader
        if not images:
            gr.Warning("源视频不包含可用帧")
            return None

        height, width = images[0].shape[:2]
        masks = read_uploaded_mask_video(
            mask_video_path, frame_count, source_fps, width, height
        )
        if masks is None:
            return None
        video_state["origin_images"] = images
        video_state["masks"] = masks
        video_state["video_path"] = source_video_path
        video_state["video_fps"] = source_fps
    elif video_state.get("origin_images") is None or video_state.get("masks") is None:
        gr.Warning(
            "Upload both the source and mask videos for direct removal, or run Tracking first."
        )
        return None
    elif video_path and video_state.get("video_path") != video_path:
        gr.Warning("The uploaded source video changed; upload its mask video or run Tracking again.")
        return None

    images = video_state["origin_images"]
    masks = video_state["masks"]

    images = np.array(images)
    masks = np.array(masks)
    img_tensor, mask_tensor = preprocess_for_removal(images, masks)
    mask_tensor = mask_tensor[:,:,:,:1]

    if mask_tensor.shape[1] < mask_tensor.shape[2]:
        height = 480
        width = 832
    else:
        height = 832
        width = 480

    with torch.no_grad():
        out = pipe(
                images=img_tensor,
                masks=mask_tensor,
                num_frames=mask_tensor.shape[0],
                height=height,
                width=width,
                num_inference_steps=int(num_inference_steps),
                generator=torch.Generator(device=device).manual_seed(random_seed),
                iterations=int(dilation_iterations)
        ).frames[0]

        out = np.uint8(out * 255)
        output_frames = [img for img in out]

    video_file = f"/tmp/{time.time()}-{random.random()}-removed_output.mp4"
    clip = ImageSequenceClip(output_frames, fps=video_state.get("video_fps", 15.0))
    clip.write_videofile(
        video_file,
        codec='libx264',
        audio=False,
        verbose=False,
        logger=None,
        ffmpeg_params=["-frames:v", str(len(output_frames))],
    )
    return video_file


def track_video(n_frames, video_path, mask_video_path, video_state):
    if not video_path:
        gr.Warning("请先上传源视频，再执行跟踪。")
        return None, None
    if video_state.get("video_path") != video_path:
        gr.Warning("源视频已变更，请等待首帧刷新完成后重新跟踪。")
        return None, None
    if video_state["origin_images"] is None:
        gr.Warning("请上传源视频并先点击“提取首帧”。")
        return None, None
    if not mask_video_path and video_state["masks"] is None:
        gr.Warning("请先在首帧完成目标分割，再点击“跟踪”。")
        return None, None

    obj_id = video_state["obj_id"]

    vr = VideoReader(video_path, ctx=cpu(0))
    source_fps = get_video_fps(vr)
    frame_count = min(len(vr), int(n_frames))
    images = [vr[i].asnumpy() for i in range(frame_count)]
    del vr
    video_state["video_fps"] = source_fps

    if images[0].shape[0] > images[0].shape[1]:
        W_ = W
        H_ = int(W_ * images[0].shape[0] / images[0].shape[1])
    else:
        H_ = H
        W_ = int(H_ * images[0].shape[1] / images[0].shape[0])

    images = [cv2.resize(img, (W_, H_)) for img in images]
    video_state["origin_images"] = images
    images = np.array(images)
    if mask_video_path:
        mask_frames = read_uploaded_mask_video(
            mask_video_path, frame_count, source_fps, W_, H_
        )
        if mask_frames is None:
            return None, None
    else:
        inference_state = video_predictor.init_state(images=images/255, device=device)
        video_state["inference_state"] = inference_state

        if len(torch.from_numpy(video_state["masks"][0]).shape) == 3:
            mask = torch.from_numpy(video_state["masks"][0])[:,:,0]
        else:
            mask = torch.from_numpy(video_state["masks"][0])

        video_predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=obj_id,
            mask=mask
        )

    output_frames = []
    color = np.array(COLOR_PALETTE[int(time.time()) % len(COLOR_PALETTE)], dtype=np.float32) / 255.0
    color = color[None, None, :]
    if mask_video_path:
        for frame, mask in zip(images, mask_frames):
            frame = frame.astype(np.float32) / 255.0
            painted = (1 - mask * 0.5) * frame + mask * 0.5 * color
            output_frames.append(np.uint8(np.clip(painted * 255, 0, 255)))
    else:
        tracked_masks = []
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
            frame = images[out_frame_idx].astype(np.float32) / 255.0
            mask = np.zeros((H, W, 3), dtype=np.float32)
            for i, logit in enumerate(out_mask_logits):
                out_mask = logit.cpu().squeeze().detach().numpy()
                out_mask = (out_mask[:,:,None] > 0).astype(np.float32)
                mask += out_mask
            mask = np.clip(mask, 0, 1)
            mask = cv2.resize(mask, (W_, H_))
            tracked_masks.append(mask)
            painted = (1 - mask * 0.5) * frame + mask * 0.5 * color
            output_frames.append(np.uint8(np.clip(painted * 255, 0, 255)))
        mask_frames = tracked_masks

    video_state["masks"] = mask_frames
    mask_output_frames = [
        np.uint8((np.asarray(mask) > 0.5).astype(np.uint8) * 255)
        for mask in mask_frames
    ]
    mask_video_file = f"/tmp/{time.time()}-{random.random()}-tracked_mask.mp4"
    mask_clip = ImageSequenceClip(mask_output_frames, fps=source_fps)
    mask_clip.write_videofile(
        mask_video_file,
        codec='libx264',
        audio=False,
        verbose=False,
        logger=None,
        ffmpeg_params=["-frames:v", str(len(mask_output_frames))],
    )

    video_file = f"/tmp/{time.time()}-{random.random()}-tracked_output.mp4"
    clip = ImageSequenceClip(output_frames, fps=source_fps)
    clip.write_videofile(
        video_file,
        codec='libx264',
        audio=False,
        verbose=False,
        logger=None,
        ffmpeg_params=["-frames:v", str(len(output_frames))],
    )
    return video_file, mask_video_file

text = """
<div style='text-align:center; font-size:32px; font-family: Arial, Helvetica, sans-serif;'>
  Minimax-Remover: Taming Bad Noise Helps Video Object Removal
</div>
<div style="display: flex; justify-content: center; align-items: center; gap: 10px; flex-wrap: nowrap;">
  <a href="https://huggingface.co/zibojia/minimax-remover"><img alt="Huggingface Model" src="https://img.shields.io/badge/%F0%9F%A4%97%20Huggingface-Model-brightgreen"></a>
  <a href="https://github.com/zibojia/MiniMax-Remover"><img alt="Github" src="https://img.shields.io/badge/MiniMaxRemover-github-black"></a>
  <a href="https://huggingface.co/spaces/zibojia/MiniMaxRemover"><img alt="Huggingface Space" src="https://img.shields.io/badge/%F0%9F%A4%97%20Huggingface-Space-1e90ff"></a>
  <a href="https://arxiv.org/abs/2505.24873"><img alt="arXiv" src="https://img.shields.io/badge/MiniMaxRemover-arXiv-b31b1b"></a>
  <a href="https://www.youtube.com/watch?v=KaU5yNl6CTc"><img alt="YouTube" src="https://img.shields.io/badge/Youtube-video-ff0000"></a>
  <a href="https://minimax-remover.github.io"><img alt="Demo Page" src="https://img.shields.io/badge/Website-Demo%20Page-yellow"></a>
</div>
<div style='text-align:center; font-size:20px; margin-top: 10px; font-family: Arial, Helvetica, sans-serif;'>
  Bojia Zi<sup>*</sup>, Weixuan Peng<sup>*</sup>, Xianbiao Qi<sup>†</sup>, Jianan Wang, Shihao Zhao, Rong Xiao, Kam-Fai Wong
</div>
<div style='text-align:center; font-size:14px; color: #888; margin-top: 5px; font-family: Arial, Helvetica, sans-serif;'>
  <sup>*</sup> Equal contribution &nbsp; &nbsp; <sup>†</sup> Corresponding author
</div>
"""

pipe, image_predictor, video_predictor = get_pipe_image_and_video_predictor()

with gr.Blocks() as demo:
    video_state = gr.State({
        "origin_images": None,
        "inference_state": None,
        "masks": None,  # Store user-generated masks
        "painted_images": None,
        "video_path": None,
        "video_fps": 15.0,
        "input_points": [],
        "scaled_points": [],
        "input_labels": [],
        "frame_idx": 0,
        "obj_id": 1
    })
    gr.Markdown(f"<div style='text-align:center;'>{text}</div>")

    with gr.Column():
        video_input = gr.Video(label="上传视频", elem_id="my-video1")
        get_info_btn = gr.Button("提取首帧", elem_id="my-btn")

        gr.Examples(
            examples=[
                ["./cartoon/0.mp4"],
                ["./cartoon/1.mp4"],
                ["./cartoon/2.mp4"],
                ["./cartoon/3.mp4"],
                ["./cartoon/4.mp4"],
                ["./normal_videos/0.mp4"],
                ["./normal_videos/1.mp4"],
                ["./normal_videos/3.mp4"],
                ["./normal_videos/4.mp4"],
                ["./normal_videos/5.mp4"],
            ],
            inputs=[video_input],
            label="选择要移除目标的视频。",
            elem_id="my-btn2"
        )

        image_output = gr.Image(label="首帧目标分割", interactive=True, elem_id="my-video")#, height="35%", width="60%")
        demo.css = """
        #my-btn {
           width: 60% !important;
           margin: 0 auto;
        }

        #my-video1 {
           width: 60% !important;
           height: 35% !important;
           margin: 0 auto;
        }
        #my-video {
           width: 60% !important;
           height: 35% !important;
           margin: 0 auto;
        }
        #my-mask-video {
           width: 60% !important;
           height: 35% !important;
           margin: 0 auto;
        }
        #fixed-mask-editor {
            width: 60% !important;
            margin: 0 auto !important;
        }
        #fixed-mask-actions {
            width: 60% !important;
            margin: 0 auto !important;
        }
        #fixed-mask-status {
            width: 60% !important;
            margin: 0 auto !important;
        }
        #my-md {
           margin: 0 auto;
        }
        #my-btn2 {
            width: 60% !important;
            margin: 0 auto;
        }
        #my-btn2 button {
            width: 120px !important;
            max-width: 120px !important;
            min-width: 120px !important;
            height: 70px !important;
            max-height: 70px !important;
            min-height: 70px !important;
            margin: 8px !important;
            border-radius: 8px !important;
            overflow: hidden !important;
            white-space: normal !important;
        }
        body.mask-editor-zoom-open {
            overflow: hidden !important;
        }
        body.mask-editor-zoom-open::before {
            display: none !important;
            content: "";
            position: fixed;
            inset: 0;
            z-index: 9999;
            background: rgba(0, 0, 0, 0.68);
        }
        #fixed-mask-editor.mask-editor-expanded {
            position: fixed !important;
            inset: 2vh 2vw !important;
            width: 96vw !important;
            height: 96vh !important;
            max-width: 96vw !important;
            max-height: 96vh !important;
            z-index: 10000 !important;
            display: flex !important;
            flex-direction: column !important;
            box-sizing: border-box !important;
            margin: 0 !important;
            padding: 14px !important;
            overflow: hidden !important;
            background: var(--background-fill-primary, white) !important;
            border-radius: 12px !important;
            box-shadow: 0 0 0 100vmax rgba(0, 0, 0, 0.68) !important;
        }
        #fixed-mask-editor.mask-editor-expanded .image-container {
            position: relative !important;
            flex: 1 1 auto !important;
            width: 100% !important;
            height: auto !important;
            min-height: 0 !important;
            overflow: hidden !important;
        }
        /*
         * The canvas wrapper is the direct child. Do not target every .wrap:
         * Gradio also uses that class for the brush controls.
         */
        #fixed-mask-editor.mask-editor-expanded .image-container > .wrap {
            position: relative !important;
            width: 100% !important;
            height: 100% !important;
            min-height: 0 !important;
        }
        /*
         * Set one dimension only. CSS derives the other from the canvas buffer,
         * so wide and tall frames both retain their exact source aspect ratio.
         */
        #fixed-mask-editor.mask-editor-expanded .image-container canvas {
            position: absolute !important;
            inset: auto !important;
            top: 50% !important;
            left: 50% !important;
            width: auto !important;
            height: 100% !important;
            max-width: 100% !important;
            max-height: 100% !important;
            margin: 0 !important;
            transform: translate(-50%, -50%) !important;
        }

        #mask-video-help { width: 60% !important; box-sizing: border-box !important; margin: 0 auto !important; text-align: center !important; overflow-wrap: anywhere !important; }
        .mask-editor-close { display: none !important; }
        #fixed-mask-editor.mask-editor-expanded .mask-editor-close.mask-editor-close-visible {
            position: absolute !important; top: 12px !important; right: 12px !important; z-index: 10020 !important;
            display: flex !important; align-items: center !important; justify-content: center !important; width: 38px !important; height: 38px !important;
            min-width: 38px !important; padding: 0 !important; border: 0 !important; border-radius: 50% !important;
            background: rgba(30, 30, 30, .78) !important; color: white !important; font-size: 30px !important; line-height: 1 !important; cursor: pointer !important;
        }
        """
        with gr.Row(elem_id="my-btn"):
            point_prompt = gr.Radio(["正向点", "反向点"], label="点选类型", value="正向点")
            clear_btn = gr.Button("清空点选")

        with gr.Row(elem_id="my-btn"):
            n_frames_slider = gr.Slider(minimum=1, maximum=361, value=81, step=1, label="处理帧数 N")
            track_btn = gr.Button("跟踪并生成掩码")
        fixed_mask_editor = gr.Image(
            label="在首帧上涂抹固定掩码（白色区域将被移除）",
            source="upload",
            tool="sketch",
            type="numpy",
            brush_color="#ffffff",
            brush_radius=20,
            mask_opacity=0.65,
            interactive=True,
            elem_id="fixed-mask-editor",
        )
        with gr.Row(elem_id="fixed-mask-actions"):
            fixed_mask_zoom_btn = gr.Button("放大涂抹区（按 Esc 返回）", elem_id="mask-editor-zoom-button", scale=1)
            fixed_mask_btn = gr.Button("生成固定掩码视频（所选帧数）", scale=1)
        fixed_mask_status = gr.Markdown(
            "在首帧涂抹目标区域，生成固定掩码视频后，点击“移除目标”开始处理。",
            elem_id="fixed-mask-status",
        )
        mask_video_input = gr.Video(
            label="掩码视频（黑底、亮色前景）",
            elem_id="my-mask-video",
        )
        gr.Markdown(
            "直接移除：上传源视频与掩码视频后，点击“移除目标”即可处理；"
            "点选掩码仅在需要预览或传播跟踪时使用。",
            elem_id="mask-video-help",
        )
        video_output = gr.Video(label="跟踪结果", elem_id="my-video")

        with gr.Column(elem_id="my-btn"):
            dilation_slider = gr.Slider(minimum=1, maximum=20, value=6, step=1, label="掩码膨胀")
            inference_steps_slider = gr.Slider(minimum=1, maximum=100, value=6, step=1, label="推理步数")

        remove_btn = gr.Button("移除目标", elem_id="my-btn")
        remove_video = gr.Video(label="移除结果", elem_id="my-video")
        fixed_mask_btn.click(
            build_fixed_mask_video,
            inputs=[fixed_mask_editor, video_input, n_frames_slider, video_state],
            outputs=[mask_video_input, fixed_mask_status],
        )
        remove_btn.click(
            inference_and_return_video,
            inputs=[
                dilation_slider,
                inference_steps_slider,
                video_input,
                mask_video_input,
                n_frames_slider,
                video_state,
            ],
            outputs=remove_video,
        )
        video_input.change(
            fn=handle_video_change,
            inputs=[video_input, video_state],
            outputs=[image_output, fixed_mask_editor, video_state, mask_video_input, video_output, remove_video],
        )
        get_info_btn.click(
            fn=handle_video_change,
            inputs=[video_input, video_state],
            outputs=[image_output, fixed_mask_editor, video_state, mask_video_input, video_output, remove_video],
        )
        image_output.select(fn=segment_frame, inputs=[point_prompt, video_state], outputs=image_output)
        clear_btn.click(clear_clicks, inputs=video_state, outputs=image_output)
        track_btn.click(
            track_video,
            inputs=[n_frames_slider, video_input, mask_video_input, video_state],
            outputs=[video_output, mask_video_input],
        )

    demo.load(
        fn=None, inputs=[], outputs=[],
        _js="""() => {
            const editorSelector = "#fixed-mask-editor", buttonSelector = "#mask-editor-zoom-button";
            const openLabel = "\u7f29\u5c0f\u5e76\u8fd4\u56de\uff08Esc\uff09", closedLabel = "\u653e\u5927\u6d82\u62b9\u533a\uff08\u6309 Esc \u8fd4\u56de\uff09";
            const getEditor = () => document.querySelector(editorSelector);
            const getButton = () => document.querySelector(buttonSelector + " button") || document.querySelector(buttonSelector);
            const isOpen = () => { const e = getEditor(); return !!e && e.classList.contains("mask-editor-expanded"); };
            const setOpen = (open) => {
                const editor = getEditor(); if (!editor) return;
                editor.classList.toggle("mask-editor-expanded", open);
                document.body.classList.toggle("mask-editor-zoom-open", open);
                const button = getButton(); if (button) button.textContent = open ? openLabel : closedLabel;
                let close = editor.querySelector(".mask-editor-close");
                if (!close) {
                    close = document.createElement("button"); close.type = "button"; close.className = "mask-editor-close";
                    close.title = "\u9000\u51fa\u653e\u5927\u6a21\u5f0f"; close.setAttribute("aria-label", "\u9000\u51fa\u653e\u5927\u6a21\u5f0f"); close.textContent = "\u00d7";
                    close.addEventListener("click", (event) => { event.preventDefault(); event.stopPropagation(); setOpen(false); });
                    editor.appendChild(close);
                }
                close.classList.toggle("mask-editor-close-visible", open);
                if (open) editor.scrollIntoView({block: "center"});
            };
            document.addEventListener("click", (event) => {
                const target = event.target instanceof Element ? event.target : null;
                if (!target || !target.closest(buttonSelector)) return;
                event.preventDefault(); event.stopImmediatePropagation(); setOpen(!isOpen());
            }, true);
            document.addEventListener("keydown", (event) => {
                if (event.key === "Escape" && isOpen()) { setOpen(false); event.preventDefault(); event.stopImmediatePropagation(); }
            }, true);
            return [];
        }""",
    )
demo.launch(server_name="0.0.0.0", server_port=8000)
