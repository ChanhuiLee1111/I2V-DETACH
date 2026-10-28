"""
Minimization
Query: Condition latent, Key: Condition latent

Maximization
Query: Video latents, Key: Video latents
"""
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode
import os, random
import re
import math
import argparse
import time
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from utils import seed_everything, _infer_sp_simple


def _start_cost_measurement(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    return time.perf_counter()


def _finish_cost_measurement(device: torch.device, start_time: float, steps: int, attack_name: str):
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    latency_sec = time.perf_counter() - start_time
    metrics = {
        "attack": attack_name,
        "latency_sec": float(latency_sec),
        "latency_per_step_sec": float(latency_sec / max(1, int(steps))),
    }

    if device.type == "cuda":
        metrics["peak_allocated_mb"] = float(torch.cuda.max_memory_allocated(device) / 1024**2)
        metrics["peak_reserved_mb"] = float(torch.cuda.max_memory_reserved(device) / 1024**2)

    return metrics


def _log_cost_metrics(pipe, metrics):
    setattr(pipe, "_last_attack_cost", metrics)
    peak_allocated = metrics.get("peak_allocated_mb")
    peak_reserved = metrics.get("peak_reserved_mb")
    mem_msg = "peak_allocated=n/a peak_reserved=n/a"
    if peak_allocated is not None and peak_reserved is not None:
        mem_msg = (
            f"peak_allocated={peak_allocated:.2f}MB "
            f"peak_reserved={peak_reserved:.2f}MB"
        )
    print(
        f"[Cost] attack={metrics['attack']} "
        f"latency={metrics['latency_sec']:.4f}s "
        f"per_step={metrics['latency_per_step_sec']:.4f}s "
        f"{mem_msg}"
    )

def compute_latent_frames(num_frames: int, pipe) -> int:
    tc = pipe.vae_scale_factor_temporal  # typically 4
    return (num_frames - 1) // tc + 1



def make_image_latents_F(pipe, img01: torch.Tensor, latent_frames: int) -> torch.Tensor:
    """
    img01: (1,3,H,W) in [0,1]
    return img_lat: (1, latent_frames, C, H', W')  (scaled)
    where first frame is encoded from image, remaining are zeros padding.
    """
    device = img01.device
    dtype = pipe.transformer.dtype

    latent_channels = pipe.transformer.config.in_channels // 2
    H = img01.shape[-2]
    W = img01.shape[-1]

    img_m11 = img01.to(dtype=dtype) * 2.0 - 1.0  # [-1,1]
    z = pipe.vae.encode(img_m11.unsqueeze(2)).latent_dist.mode()  # (1,C,1,H',W')
    z = z.permute(0, 2, 1, 3, 4)  # (1,1,C,H',W')

    pad_frames = latent_frames - 1
    assert pad_frames >= 0
    pad = torch.zeros(
        (1, pad_frames, latent_channels, H // pipe.vae_scale_factor_spatial, W // pipe.vae_scale_factor_spatial),
        device=device, dtype=dtype
    )
    z = torch.cat([z, pad], dim=1)  # (1,F,C,H',W')

    if not pipe.vae.config.invert_scale_latents:
        z = pipe.vae_scaling_factor_image * z
    else:
        z = (1.0 / pipe.vae_scaling_factor_image) * z
    return z

def make_noisy_video_latents_all_es(
    pipe,
    *,
    base_img01: torch.Tensor,
    latent_frames: int,
    latent_repeat_frames: int,
    t0: torch.Tensor,
    it: int,
    Xv=None,
    apply_ES: bool = True,
):
    device = base_img01.device
    dtype = pipe.transformer.dtype

    assert base_img01.shape[0] == 1 and base_img01.ndim == 4
    assert latent_frames >= 1
    assert latent_repeat_frames >= 1

    base01 = base_img01.to(device=device, dtype=dtype)  # (1,3,H,W)

    def encode_video_3d_from_f3hw(x_f3hw: torch.Tensor):
        """
        x_f3hw: (T,3,H,W) in [0,1]
        return : (1,T',C,Hp,Wp)
        """
        x_m11 = x_f3hw * 2.0 - 1.0
        x_m11 = x_m11.permute(1, 0, 2, 3).unsqueeze(0)      # (1,3,T,H,W)
        z = pipe.vae.encode(x_m11).latent_dist.mean         # (1,C,T',Hp,Wp)
        z = z.permute(0, 2, 1, 3, 4).contiguous()           # (1,T',C,Hp,Wp)
        if not pipe.vae.config.invert_scale_latents:
            z = pipe.vae_scaling_factor_image * z
        else:
            z = (1.0 / pipe.vae_scaling_factor_image) * z
        return z

    base_repeat = base01.repeat(latent_repeat_frames, 1, 1, 1)
    z_base = encode_video_3d_from_f3hw(base_repeat)[:, :1]        # (1,1,C,Hp,Wp)

    # --------------------------------------------------
    # (1) tail optimize
    # first latent is condition, remaining latents are all ES
    # --------------------------------------------------
    Fr = latent_frames - 1  # ES-only: copy Xv into every non-condition video latent slot

    step_v = 2.0 / 255.0
    steps_v = 1

    
    if not apply_ES:
        # Xv = torch.zeros((Fr, 3, base01.shape[-2], base01.shape[-1]), device=device, dtype=dtype)
        Xv = base01.repeat(Fr, 1, 1, 1)
    else: 
        if Fr > 0:
            if Xv is None:
                Xv_single = base01.clone()
            else:
                Xv = Xv.to(device=device, dtype=dtype).detach()
                assert Xv.shape[0] == Fr, f"Xv has {Xv.shape[0]} frames but expected non-condition slots={Fr}"
                Xv_single = Xv[:1].detach().clone()

            noise_std = 1.0 / 255.0
            Xv_single = (Xv_single + noise_std * torch.randn_like(Xv_single)).clamp(0, 1).detach()
            Xv_single = Xv_single.to(device=device, dtype=dtype).detach()

            delta_v = torch.zeros_like(Xv_single)
            loss_log = []

            for _ in range(steps_v):
                xk = Xv_single.to(dtype=dtype)
                dk = delta_v.detach().clone().to(dtype=dtype).requires_grad_(True)

                xk_adv = (xk + dk).clamp(0, 1)
                x4 = xk_adv.repeat(latent_repeat_frames, 1, 1, 1)
                z = encode_video_3d_from_f3hw(x4)[:, :1]

                loss_base = ((z - z_base) ** 2).mean()

                loss_nbr = 0.0
                cnt = 0
                loss_v = 1024 * (loss_base + loss_nbr)

                gk = torch.autograd.grad(loss_v, dk, retain_graph=False, create_graph=False)[0]

                with torch.no_grad():
                    dk = dk + step_v * gk.sign()
                    base_k = base01
                    delta_k = (xk + dk - base_k).clamp(-1, 1)
                    Xv_single = (base_k + delta_k).clamp(0, 1).detach()
                    delta_v = delta_k.detach()
                    loss_log.append(float(loss_v.detach().item()))

            Xv = Xv_single.repeat(Fr, 1, 1, 1)
        else:
            Xv = base01.new_zeros((0, 3, base01.shape[-2], base01.shape[-1]))

    
    # --------------------------------------------------
    # (3) encode tail latents
    # --------------------------------------------------
    if Fr > 0:
        # v1:
        # x_tail_f3hw = Xv.repeat_interleave(latent_repeat_frames, dim=0)
        # z_tail = encode_video_3d_from_f3hw(x_tail_f3hw)[:, :Fr]   # (1,Fr,C,Hp,Wp)
        x_tail_short = Xv[:1].repeat(latent_repeat_frames, 1, 1, 1)
        z_tail_unit = encode_video_3d_from_f3hw(x_tail_short)[:, :1]
        z_tail = z_tail_unit.repeat(1, Fr, 1, 1, 1)
    else:
        C = z_base.shape[2]
        Hp, Wp = z_base.shape[-2], z_base.shape[-1]
        z_tail = z_base.new_zeros((1, 0, C, Hp, Wp))

    z_init = torch.cat([z_base, z_tail], dim=1)  # (1,latent_frames,C,Hp,Wp)


    # --------------------------------------------------
    # (5) add noise at timestep t0
    # --------------------------------------------------
    eps_noise = torch.randn_like(z_init, device=device, dtype=dtype)
    noisy = pipe.scheduler.add_noise(z_init, eps_noise, t0)
    noisy_in = pipe.scheduler.scale_model_input(noisy, t0)

    aux_samples = base01.new_zeros((0, 3, base01.shape[-2], base01.shape[-1]))
    return noisy_in, z_base, aux_samples, Xv


def two_logits_simple(
    pipe,
    *,
    text_len: int,
    latent_frames: int,
    layer_start: int = 0,
    layer_end: int | None = None,       
    use_abs: bool = False,              
):
    blocks = pipe.transformer.transformer_blocks
    if layer_end is None:
        layer_end = len(blocks)

    s_text = None
    s_img = None
    s_tail = None
    n = 0

    for li in range(layer_start, layer_end):
        layer = blocks[li]
        q = layer.attn1.temp_query   # (B,H,seq,d)
        k = layer.attn1.temp_key

        if q is None or k is None:
            raise RuntimeError(f"temp_query/temp_key None at layer {li}. forward first!")

        B, Hh, seq, d = q.shape
        sp = _infer_sp_simple(q, text_len=text_len, latent_frames=latent_frames)
        scale = 1.0 / math.sqrt(d)


        q_img_s = text_len + sp
        q_img_e = text_len + latent_frames * sp
        q_vid_s = text_len + sp
        q_vid_e = text_len + latent_frames * sp
        k0_s = text_len
        k0_e = text_len + sp
        kt_s = text_len + sp
        kt_e = text_len + latent_frames * sp

        acc_text = torch.zeros((), device=q.device, dtype=q.dtype)
        acc_img = torch.zeros((), device=q.device, dtype=q.dtype)
        acc_tail = torch.zeros((), device=q.device, dtype=q.dtype)
        for h in range(Hh):
            qh_img = q[:, h, q_img_s:q_img_e, :]
            qh_vid = q[:, h, q_vid_s:q_vid_e, :]
            k0 = k[:, h, k0_s:k0_e, :]
            kt = k[:, h, kt_s:kt_e, :]
            
            #condition image          # (B,K0,d)
            log0 = (qh_img @ k0.transpose(-1, -2)) * scale  # (B,Q,K0)
            acc_img = acc_img + (log0.abs().mean() if use_abs else log0.mean()) / float(Hh)

            #video
            logt = (qh_vid @ kt.transpose(-1, -2))* scale
            acc_tail = acc_tail + (logt.abs().mean() if use_abs else logt.mean()) / float(Hh)

        s_img = acc_img if s_img is None else (s_img + acc_img)
        s_tail = acc_tail if s_tail is None else (s_tail + acc_tail)
        n += 1
    if n > 0:
        s_img = s_img / n
        s_tail = s_tail / n
    return s_text, s_img, s_tail

def i2vdetach_cog(
    pipe,
    img01: torch.Tensor,              # (1,3,H,W) in [0,1]
    *,
    steps: int,
    eps: float,
    step_size: float,
    num_frames: int,
    timesteps_total: int,
    strength: float,
    noise_mode: str,                  # "random_t" or "fixed_t"
    fixed_t_index: int,
    latent_repeat_frames: int = 4,
    # whether to use only first video frame queries (frame1) or all (frame1..)
    apply_ES : bool = True,
    apply_black: bool = False,
    use_only_first_video_frame: bool = False,
    args=None,
    prompt = ""
):
    device = img01.device
    dtype = pipe.transformer.dtype
    cost_start = _start_cost_measurement(device)

    # mid layer only
    n_layers = len(pipe.transformer.transformer_blocks)
    layer = pipe.transformer.transformer_blocks[n_layers // 2]

    # empty prompt
    with torch.no_grad():
        emb, _ = pipe.encode_prompt(prompt, device=device)
    text_len = emb.size(1)

    # scheduler timesteps
    pipe.scheduler.set_timesteps(timesteps_total, device=device)
    ts = pipe.scheduler.timesteps
    ts, _ = pipe.get_timesteps(timesteps_total, ts, strength, device)

    ofs_emb = torch.tensor([2.0], dtype=dtype, device=device)

    latent_frames = compute_latent_frames(num_frames, pipe)  # typically 13

    layer_start = len(pipe.transformer.transformer_blocks) // 4
    layer_end   = len(pipe.transformer.transformer_blocks) * 3 // 4

    H = img01.shape[-2]
    W = img01.shape[-1]
    img_rot_emb = (
        pipe._prepare_rotary_positional_embeddings(H, W, latent_frames, device)
        if pipe.transformer.config.use_rotary_positional_embeddings
        else None
    )

    # PGD variables
    img_f = img01.to(dtype=dtype)
    delta = torch.zeros_like(img_f, device=device, dtype=dtype, requires_grad=True)

    for it in range(steps):
        # conditioning image (adv)
        if it == 0:
            X_v = None
        img_adv01 = (img_f + delta).clamp(0, 1).to(dtype=dtype)

        # choose timestep
        seed_everything(42+it, deterministic=True)
        if noise_mode == "fixed_t":
            ti = max(0, min(fixed_t_index, len(ts) - 1))
            t0 = ts[ti:ti+1]
        else:
            ti = torch.randint(0, fixed_t_index, (1,), device=device).item()
            t0 = ts[ti:ti+1]

        base_for_video = img01.detach()
        noisy_in, z_ori, aux_samples, X_v = make_noisy_video_latents_all_es(
            pipe,
            base_img01=base_for_video,
            latent_frames=latent_frames,
            latent_repeat_frames=latent_repeat_frames,
            t0=t0,
            it=it,
            Xv=X_v,
            apply_ES=args.apply_ES,
        )

        # condition latents (grad) : first frame encoded, rest padded zeros (same as your make_image_latents_F)
        img_lat_adv = make_image_latents_F(pipe, img_adv01, latent_frames=latent_frames)  # (1,F,C,Hp,Wp)
        
        # transformer input
        hs_adv = torch.cat([noisy_in, img_lat_adv], dim=2)  # (1,F,2C,Hp,Wp)


        _ = pipe.transformer(
            hidden_states=hs_adv,
            encoder_hidden_states=emb,
            timestep=t0,
            ofs=ofs_emb,
            image_rotary_emb=img_rot_emb,
            return_dict=False,
        )[0]
        
        loss_text, loss_img, loss_video = two_logits_simple(
                pipe,
                text_len=emb.size(1),
                latent_frames=latent_frames,
                layer_start=layer_start,
                layer_end=layer_end,
                use_abs=False,   
            )
        loss = loss_img -loss_video 
        grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]
        #print(grad.min(), grad.max())
        with torch.no_grad():
            delta -= step_size * grad.sign()          
            delta.clamp_(-eps, eps)
            #delta.clamp_(-img_f, 1.0 - img_f)
        delta.grad = None
        torch.cuda.empty_cache()
        # del q_adv, k_adv, hs_adv, img_lat_adv, noisy_in
        # optional cleanup (avoid references lingering)
        if hasattr(layer.attn1, "temp_query"):
            layer.attn1.temp_query = None
        if hasattr(layer.attn1, "temp_key"):
            layer.attn1.temp_key = None
        if hasattr(layer.attn1, "temp_value"):
            layer.attn1.temp_value = None

        print(f"[{it+1:04d}/{steps}] total_loss={(loss).item():.6f} | img_loss={(loss_img).item():.6f}, tail_loss={(loss_video).item():.6f}, |delta|_inf={delta.abs().max().item():.5f} t_idx={ti}")
    img_out = (img_f + delta).clamp(0, 1).detach()
    cost_metrics = _finish_cost_measurement(device, cost_start, steps, "method_v2_cog_es_only_1_frame")
    _log_cost_metrics(pipe, cost_metrics)
    return img_out, delta.detach(), aux_samples, X_v 
