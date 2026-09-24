"""Frame-by-frame compositing of a removal result back onto the source video."""

import os
import tempfile

import cv2
import numpy as np

MASK_GROW_PIXELS = 40
MASK_BLUR_SIGMA = 10


def _fps(capture, fallback=15.0):
    value = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    return value if np.isfinite(value) and value > 0 else fallback


def _bicubic_resize(frame, width, height, *, mask=False):
    """Match ImageCompositeMaskedOneByOne's torch bicubic interpolation."""
    import torch
    import torch.nn.functional as functional

    tensor = torch.from_numpy(np.ascontiguousarray(frame)).to(torch.float32)
    if mask:
        tensor = tensor[None, None] / 255.0
    else:
        tensor = tensor.permute(2, 0, 1)[None] / 255.0
    with torch.inference_mode():
        resized = functional.interpolate(
            tensor,
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        )
    if mask:
        return resized[0, 0].clamp(0, 1).numpy()
    return resized[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255.0


def composite_video(
    source_path,
    result_path,
    mask_path,
    output_path=None,
    mask_grow_pixels=MASK_GROW_PIXELS,
):
    """Replace bright-mask pixels in source frames with the resized result frames.

    All streams are decoded and composited one frame at a time on CPU. Result
    and mask frames are scaled to the full source frame; no crop or offset is
    introduced. The source audio track is copied onto the final video when it
    exists.
    """
    if not all(os.path.isfile(path) for path in (source_path, result_path, mask_path)):
        raise ValueError("源视频、结果视频和Mask视频都必须存在。")

    source = cv2.VideoCapture(source_path)
    result = cv2.VideoCapture(result_path)
    mask = cv2.VideoCapture(mask_path)
    captures = (source, result, mask)
    if not all(capture.isOpened() for capture in captures):
        for capture in captures:
            capture.release()
        raise ValueError("无法读取源视频、结果视频或Mask视频。")

    width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = _fps(source)
    mask_fps = _fps(mask, source_fps)
    if width < 1 or height < 1:
        for capture in captures:
            capture.release()
        raise ValueError("无法读取源视频尺寸。")

    if output_path is None:
        output_fd, output_path = tempfile.mkstemp(
            prefix="minimax-composited-", suffix=".mp4"
        )
        os.close(output_fd)
        os.unlink(output_path)
    fd, silent_path = tempfile.mkstemp(prefix="minimax-composite-silent-", suffix=".mp4")
    os.close(fd)
    writer = cv2.VideoWriter(
        silent_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps,
        (width, height),
    )
    if not writer.isOpened():
        for capture in captures:
            capture.release()
        os.unlink(silent_path)
        raise RuntimeError("无法创建贴回视频。")

    frame_count = 0
    mask_frame_index = -1
    sampled_mask_frame = None
    try:
        try:
            while True:
                result_ok, result_frame = result.read()
                if not result_ok:
                    break

                source_ok, source_frame = source.read()
                if not source_ok:
                    raise ValueError(
                        f"贴回视频帧数不匹配：源视频缺少第 {frame_count + 1} 帧。"
                    )

                # Select the mask frame at the source timestamp. The mask
                # video's FPS can differ from the source/result FPS.
                target_mask_index = int(round(frame_count * mask_fps / source_fps))
                while mask_frame_index < target_mask_index:
                    mask_ok, sampled_mask_frame = mask.read()
                    if not mask_ok:
                        raise ValueError(
                            f"Mask视频时长不足：无法覆盖第 {frame_count + 1} 帧。"
                        )
                    mask_frame_index += 1

                # MaskFastGrow operates before ImageCompositeMaskedOneByOne scales
                # the mask to the destination dimensions.
                alpha = (sampled_mask_frame[:, :, 2] > 127).astype(np.uint8) * 255
                if mask_grow_pixels:
                    kernel_size = int(mask_grow_pixels) * 2 + 1
                    kernel = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
                    )
                    alpha = cv2.dilate(alpha, kernel, iterations=1)
                if MASK_BLUR_SIGMA:
                    alpha = cv2.GaussianBlur(
                        alpha, (0, 0), sigmaX=MASK_BLUR_SIGMA
                    )
                alpha = _bicubic_resize(alpha, width, height, mask=True)
                if result_frame.shape[1] != width or result_frame.shape[0] != height:
                    result_frame = _bicubic_resize(result_frame, width, height)
                alpha = alpha[:, :, None]
                composited = (
                    result_frame.astype(np.float32) * alpha
                    + source_frame.astype(np.float32) * (1.0 - alpha)
                ).clip(0, 255).astype(np.uint8)
                writer.write(composited)
                frame_count += 1
        finally:
            for capture in captures:
                capture.release()
            writer.release()
    except Exception:
        if os.path.exists(silent_path):
            os.unlink(silent_path)
        raise

    if frame_count == 0:
        os.unlink(silent_path)
        raise ValueError("源视频不包含可用帧。")

    # MoviePy muxes the original audio without loading all decoded frames into
    # memory. Its older API is used here to match the app's existing imports.
    source_clip = None
    silent_clip = None
    try:
        from moviepy.editor import VideoFileClip

        source_clip = VideoFileClip(source_path)
        if source_clip.audio is None:
            os.replace(silent_path, output_path)
            return output_path

        silent_clip = VideoFileClip(silent_path, audio=False)
        silent_clip.set_audio(source_clip.audio).set_duration(
            min(silent_clip.duration, source_clip.duration)
        ).write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            fps=source_fps,
            verbose=False,
            logger=None,
            ffmpeg_params=["-pix_fmt", "yuv420p", "-frames:v", str(frame_count)],
        )
        return output_path
    finally:
        if silent_clip is not None:
            silent_clip.close()
        if source_clip is not None:
            source_clip.close()
        if os.path.exists(silent_path):
            os.unlink(silent_path)
