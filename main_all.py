import argparse
import os
import re
from pathlib import Path

import torch
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image

from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video

from i2vdetach_cog import i2vdetach_cog
from i2vdetach_ltx import i2vdetach_ltx


ATTACK_REGISTRY = {
    ("v2", "cog"): i2vdetach_cog,
    ("v2", "ltx"): i2vdetach_ltx,
}

MODEL_ATTACK_CONFIG = {
    ("v2", "cog"): {
        "attack_num_frames": 49,
        "preprocess_h": 480,
        "preprocess_w": 720,
        "latent_repeat_frames": 4,
        "use_only_first_video_frame": True,
        "video_num_frames": 49,
        "video_fps": 8,
    },
    ("v2", "ltx"): {
        "attack_num_frames": 121,
        "preprocess_h": 480,
        "preprocess_w": 704,
        "latent_repeat_frames": 8,
        "use_only_first_video_frame": True,
        "video_num_frames": 121,
        "video_fps": 25,
    },
}


def safe_filename(s: str, max_len: int = 200) -> str:
    s = s.strip()
    s = re.sub(r"[\/\\\:\*\?\"\<\>\|]+", "_", s)
    s = re.sub(r"\s+", " ", s)
    return s[:max_len]


def load_pil(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def preprocess_with_video_processor(
    pil_img: Image.Image,
    height: int,
    width: int,
    device,
    dtype,
) -> torch.Tensor:
    img = pil_img.resize((width, height), resample=Image.BICUBIC)
    return TF.to_tensor(img).unsqueeze(0).to(device=device, dtype=dtype)


def save_x_f3hw_as_pngs(
    x_f3hw: torch.Tensor,
    out_dir: str,
    *,
    prefix: str = "",
    start_idx: int = 0,
):
    os.makedirs(out_dir, exist_ok=True)

    x = x_f3hw.detach()
    if x.is_cuda:
        x = x.cpu()
    x = x.clamp(0, 1)

    for i in range(x.shape[0]):
        img_u8 = (x[i] * 255.0).round().to(torch.uint8)
        img_hwc = img_u8.permute(1, 2, 0).numpy()
        Image.fromarray(img_hwc).save(
            os.path.join(out_dir, f"{prefix}{start_idx + i:04d}.png")
        )


def save_delta_vis(delta_13hw: torch.Tensor, path_vis: str):
    os.makedirs(os.path.dirname(path_vis), exist_ok=True)
    d = delta_13hw.detach().float().cpu()[0]
    dmin = d.min()
    dmax = d.max()
    if (dmax - dmin) < 1e-12:
        vis = torch.zeros_like(d)
    else:
        vis = (d - dmin) / (dmax - dmin)
    torchvision.utils.save_image(vis, path_vis)


def get_artifact_dirs(args):
    base = Path(f"Results_{args.method}")
    dirs = {
        "base": base,
        "adv": base / "adv",
        "perturbation": base / "perturbation",
        "video": base / "video",
        "txt": base / "txt",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def get_sample_paths(artifact_dirs, sample_name: str):
    name = safe_filename(sample_name)
    return {
        "name": name,
        "adv_png": artifact_dirs["adv"] / f"{name}.png",
        "delta_pt": artifact_dirs["perturbation"] / f"{name}.pt",
        "delta_vis_png": artifact_dirs["perturbation"] / f"{name}_vis.png",
        "es_dir": artifact_dirs["perturbation"] / f"{name}_ES",
        "video_mp4": artifact_dirs["video"] / f"{name}.mp4",
        "txt": artifact_dirs["txt"] / f"{name}.txt",
    }


def save_txt_metadata(txt_path: Path, args, prompt: str, src_img_path: str, model_cfg: dict):
    lines = [
        f"prompt: {prompt}",
        f"source_image: {os.path.abspath(src_img_path)}",
        f"method_version: {args.method_version}",
        f"model: {args.method}",
        f"model_id: {args.model_id}",
        f"apply_ES: {args.apply_ES}",
        f"apply_black: {args.apply_black}",
        f"steps: {args.steps}",
        f"eps: {args.eps}",
        f"step_size: {args.step_size}",
        f"timesteps_total: {args.timesteps_total}",
        f"strength: {args.strength}",
        f"noise_mode: {args.noise_mode}",
        f"fixed_t_index: {args.fixed_t_index}",
        f"attack_num_frames: {model_cfg['attack_num_frames']}",
        f"preprocess_h: {model_cfg['preprocess_h']}",
        f"preprocess_w: {model_cfg['preprocess_w']}",
        f"latent_repeat_frames: {model_cfg['latent_repeat_frames']}",
        f"use_only_first_video_frame: {model_cfg['use_only_first_video_frame']}",
        f"video_num_frames: {model_cfg['video_num_frames']}",
        f"video_fps: {model_cfg['video_fps']}",
        f"device: {args.device}",
        f"dtype: {args.dtype}",
    ]
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_attack_outputs(sample_paths, adv_image, delta, X_v):
    if X_v is not None:
        save_x_f3hw_as_pngs(
            X_v,
            out_dir=str(sample_paths["es_dir"]),
            prefix="ES-Xv_",
        )

    torch.save(delta.detach().cpu(), sample_paths["delta_pt"])
    save_delta_vis(delta, str(sample_paths["delta_vis_png"]))

    adv_vis = adv_image[0].float().clamp(0, 1).cpu()
    adv_pil = torchvision.transforms.ToPILImage()(adv_vis)
    adv_pil.save(sample_paths["adv_png"])
    return adv_pil


def build_cog_pipeline(args, device, dtype):
    print("Loading:", args.model_id)
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
    ).to(device)

    pipe.vae.requires_grad_(False)
    pipe.transformer.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    pipe.transformer.enable_gradient_checkpointing()
    return pipe


def build_ltx_pipeline(args, device, dtype):
    from diffusers import LTXImageToVideoPipeline
    from diffusers.models.transformers.transformer_ltx import LTXVideoTransformer3DModel

    transformer = LTXVideoTransformer3DModel.from_single_file(
        "../ltxv-2b-0.9.6-dev-04-25.safetensors",
        torch_dtype=dtype,
    )
    pipe = LTXImageToVideoPipeline.from_pretrained(
        "Lightricks/LTX-Video-0.9.7-dev",
        transformer=transformer,
        torch_dtype=dtype,
    ).to(device)

    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.transformer.requires_grad_(False)
    return pipe


def generate_video_cog(pipe, prompt: str, adv_pil: Image.Image, model_cfg: dict):
    with torch.no_grad():
        video_frames = pipe(
            prompt=prompt,
            image=adv_pil,
            num_videos_per_prompt=1,
            num_inference_steps=50,
            num_frames=model_cfg["video_num_frames"],
            guidance_scale=6,
            generator=torch.Generator(device="cuda").manual_seed(42),
        ).frames[0]
    return video_frames, model_cfg["video_fps"]


def generate_video_ltx(pipe, prompt: str, adv_pil: Image.Image, args, model_cfg: dict):
    with torch.no_grad():
        video_frames = pipe(
            prompt=prompt,
            image=adv_pil,
            num_frames=model_cfg["video_num_frames"],
            guidance_scale=3.0,
            generator=torch.Generator(device=args.device).manual_seed(42),
        ).frames[0]
    return video_frames, model_cfg["video_fps"]


def run_attack(pipe, img01, args, prompt, attack_fn, model_cfg):
    eps = args.eps / 255.0
    step = args.step_size / 255.0

    common_kwargs = dict(
        pipe=pipe,
        img01=img01,
        steps=args.steps,
        eps=eps,
        step_size=step,
        timesteps_total=args.timesteps_total,
        noise_mode=args.noise_mode,
        fixed_t_index=args.fixed_t_index,
        latent_repeat_frames=model_cfg["latent_repeat_frames"],
        apply_ES=args.apply_ES,
        apply_black=args.apply_black,
        use_only_first_video_frame=model_cfg["use_only_first_video_frame"],
        args=args,
        prompt=prompt,
    )

    if args.method == "cog":
        return attack_fn(
            num_frames=model_cfg["attack_num_frames"],
            strength=args.strength,
            **common_kwargs,
        )

    if args.method == "ltx":
        return attack_fn(
            num_frames=model_cfg["attack_num_frames"],
            height=model_cfg["preprocess_h"],
            width=model_cfg["preprocess_w"],
            **common_kwargs,
        )

    raise ValueError(f"Unsupported method: {args.method}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_path", type=str, required=True)
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--sample_name", type=str, default=None)

    ap.add_argument("--model_id", type=str, default="THUDM/CogVideoX-5b-I2V")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])

    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--eps", type=float, default=8)
    ap.add_argument("--step_size", type=float, default=1.0)
    ap.add_argument("--method", type=str, default="cog", choices=["cog", "ltx"])
    ap.add_argument("--method_version", type=str, default="v2", choices=["v2"])
    ap.add_argument("--timesteps_total", type=int, default=50)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--noise_mode", type=str, default="random_t", choices=["random_t", "fixed_t"])
    ap.add_argument("--fixed_t_index", type=int, default=5)
    ap.add_argument("--apply_ES", action="store_true")
    ap.add_argument("--apply_black", action="store_true")
    args = ap.parse_args()

    key = (args.method_version, args.method)
    if key not in ATTACK_REGISTRY or key not in MODEL_ATTACK_CONFIG:
        raise ValueError(
            f"Unsupported attack/model combination: method_version={args.method_version}, method={args.method}"
        )
    attack_fn = ATTACK_REGISTRY[key]
    model_cfg = MODEL_ATTACK_CONFIG[key]

    image_path = Path(args.image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    prompt = args.prompt if args.prompt is not None else image_path.stem
    sample_name = args.sample_name if args.sample_name is not None else prompt

    device = torch.device(args.device)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    artifact_dirs = get_artifact_dirs(args)
    sample_paths = get_sample_paths(artifact_dirs, sample_name)

    if args.method == "cog":
        pipe = build_cog_pipeline(args, device, dtype)
    else:
        pipe = build_ltx_pipeline(args, device, dtype)

    print(f"image: {image_path}")
    print(f"prompt: {prompt}")
    print(f"adv: {sample_paths['adv_png']}")
    print(f"vid: {sample_paths['video_mp4']}")
    print(f"txt: {sample_paths['txt']}")

    pil = load_pil(str(image_path))
    img01 = preprocess_with_video_processor(
        pil,
        model_cfg["preprocess_h"],
        model_cfg["preprocess_w"],
        device=device,
        dtype=dtype,
    )

    adv_image, delta, aux_samples, X_v = run_attack(
        pipe=pipe,
        img01=img01,
        args=args,
        prompt=prompt,
        attack_fn=attack_fn,
        model_cfg=model_cfg,
    )

    adv_pil = save_attack_outputs(
        sample_paths=sample_paths,
        adv_image=adv_image,
        delta=delta,
        X_v=X_v,
    )

    save_txt_metadata(
        txt_path=sample_paths["txt"],
        args=args,
        prompt=prompt,
        src_img_path=str(image_path),
        model_cfg=model_cfg,
    )

    if args.method == "cog":
        video_frames, fps = generate_video_cog(pipe, prompt, adv_pil, model_cfg)
    else:
        video_frames, fps = generate_video_ltx(pipe, prompt, adv_pil, args, model_cfg)

    export_to_video(video_frames, str(sample_paths["video_mp4"]), fps=fps)


if __name__ == "__main__":
    main()
