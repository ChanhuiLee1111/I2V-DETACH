import math
import numpy as np
import torch
import torch.nn.functional as F

from utils import seed_everything, _infer_sp_simple
"""
Minimization
Query: Condition + K latents + Video latents, Key: Condition

Maximization
Query: Video latents, Key: Video latents
"""
def apply_rotary_emb(x, freqs):
    cos, sin = freqs
    x_real, x_imag = x.unflatten(2, (-1, 2)).unbind(-1)  # [B, S, C // 2]
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(2)
    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
    return out
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

def _ltx_ensure_bhsd(x: torch.Tensor, heads: int) -> torch.Tensor:
    """
    Accept q/k in either (B,H,seq,d) or (B,seq,H,d), return (B,H,seq,d).
    """
    if x is None:
        raise RuntimeError("q/k is None")
    if x.ndim != 4:
        raise RuntimeError(f"Expected q/k ndim=4, got {tuple(x.shape)}")

    # already (B,H,seq,d)
    if x.shape[1] == heads:
        return x.contiguous()

    # likely (B,seq,H,d)
    if x.shape[2] == heads:
        return x.permute(0, 2, 1, 3).contiguous()

    raise RuntimeError(
        f"Cannot infer q/k layout from shape={tuple(x.shape)} with heads={heads}. "
        "Expected (B,H,seq,d) or (B,seq,H,d)."
    )


def _ltx_infer_sp_from_q(q_bhsd: torch.Tensor, latent_frames: int) -> int:
    seq = q_bhsd.shape[2]
    if seq % latent_frames != 0:
        raise ValueError(
            f"[LTX self-attn slicing] seq={seq} not divisible by latent_frames={latent_frames}. "
            "Check LTX token ordering / _pack_latents assumption."
        )
    return seq // latent_frames


def two_logits_ltx_self_split(
    pipe,
    *,
    latent_frames: int,
    layer_start: int = 0,
    layer_end: int | None = None,
    use_abs: bool = False,
):
    """
    LTX self-attn version of split logits:
      - cond loss: q_eot vs k_cond (minimize)
      - tail loss: q_tail vs k_tail (maximize)

    Returns:
      (loss_text=None, loss_cond, loss_tail)
    """
    blocks = pipe.transformer.transformer_blocks
    if layer_end is None:
        layer_end = len(blocks)

    s_cond = None
    s_tail = None
    n_layers = 0

    for li in range(layer_start, layer_end):
        layer = blocks[li]
        if not hasattr(layer, "attn1"):
            raise RuntimeError(f"LTX block {li} has no attn1")

        attn1 = layer.attn1
        heads = getattr(attn1, "heads", None)
        if heads is None:
            raise RuntimeError(f"Layer {li}.attn1 has no heads attr")

        q_raw = getattr(attn1, "temp_query", None)
        k_raw = getattr(attn1, "temp_key", None)
        if q_raw is None or k_raw is None:
            raise RuntimeError(
                f"Layer {li}.attn1.temp_query/temp_key is None. "
                "Forward must run first and self-attn q/k must be stored."
            )

        q = _ltx_ensure_bhsd(q_raw, heads=heads)  # (B,H,seq,d)
        k = _ltx_ensure_bhsd(k_raw, heads=heads)  # (B,H,seq,d)

        B, Hh, seq_q, d = q.shape
        _, Hk, seq_k, dk = k.shape
        if Hh != Hk or d != dk:
            raise RuntimeError(f"Q/K mismatch at layer {li}: q={q.shape}, k={k.shape}")

        sp_q = _ltx_infer_sp_from_q(q, latent_frames)
        sp_k = _ltx_infer_sp_from_q(k, latent_frames)
        if sp_q != sp_k:
            raise RuntimeError(f"sp mismatch at layer {li}: {sp_q} vs {sp_k}")
        sp = sp_q

        scale = 1.0 / math.sqrt(d)

        # frame-major slicing assumption
        # frame0: condition key
        cond_s, cond_e = 0, sp
        q_cond_s, q_cond_e = sp, latent_frames * sp
        tail_s, tail_e = sp, latent_frames * sp

        def _clamp(v, lo, hi):
            return max(lo, min(v, hi))

        cond_s = _clamp(cond_s, 0, seq_k)
        cond_e = _clamp(cond_e, 0, seq_k)
        q_cond_s = _clamp(q_cond_s, 0, seq_q)
        q_cond_e = _clamp(q_cond_e, 0, seq_q)

        q_tail_s = _clamp(tail_s, 0, seq_q)
        q_tail_e = _clamp(tail_e, 0, seq_q)

        k_tail_s = _clamp(tail_s, 0, seq_k)
        k_tail_e = _clamp(tail_e, 0, seq_k)

        acc_cond = torch.zeros((), device=q.device, dtype=q.dtype)
        acc_tail = torch.zeros((), device=q.device, dtype=q.dtype)
        valid_cond = 0
        valid_tail = 0

        for h in range(Hh):
            # Match original k=0 behavior:
            # minimize attention from all queries to the condition key slice.
            qh_eot = q[:, h, q_cond_s:q_cond_e, :]
            kh_cond = k[:, h, cond_s:cond_e, :]
            if qh_eot.shape[1] > 0 and kh_cond.shape[1] > 0:
                log_cond = (qh_eot @ kh_cond.transpose(-1, -2)) * scale
                acc_cond = acc_cond + (log_cond.abs().mean() if use_abs else log_cond.mean())
                valid_cond += 1

            # q_tail vs k_tail (maximize)
            qh_tail = q[:, h, q_tail_s:q_tail_e, :]
            kh_tail = k[:, h, k_tail_s:k_tail_e, :]
            if qh_tail.shape[1] > 0 and kh_tail.shape[1] > 0:
                log_tail = (qh_tail @ kh_tail.transpose(-1, -2)) * scale
                acc_tail = acc_tail + (log_tail.abs().mean() if use_abs else log_tail.mean())
                valid_tail += 1

        acc_cond = acc_cond / max(valid_cond, 1)
        acc_tail = acc_tail / max(valid_tail, 1)

        if s_cond is None:
            s_cond = acc_cond
            s_tail = acc_tail
        else:
            s_cond = s_cond + acc_cond
            s_tail = s_tail + acc_tail

        n_layers += 1

    if n_layers == 0:
        raise RuntimeError("No layers processed in two_logits_ltx_self_split")

    s_cond = s_cond / n_layers
    s_tail = s_tail / n_layers
    return None, s_cond, s_tail

def make_noisy_video_latents_all_es_ltx(
    pipe,
    *,
    base_img01: torch.Tensor,
    latent_frames: int,
    latent_repeat_frames: int,
    t0: torch.Tensor,
    it=0,
    Xv=None,
    apply_ES: bool = True,
):
    device = base_img01.device
    dtype = pipe.transformer.dtype

    assert base_img01.shape[0] == 1 and base_img01.ndim == 4
    assert latent_frames >= 1
    assert latent_repeat_frames >= 1

    base01 = base_img01.to(device=device, dtype=dtype)

    # LTX latent normalization stats
    with torch.no_grad():
        tmp = pipe.vae.encode((base01 * 2 - 1).unsqueeze(2)).latent_dist.mode()
    latents_mean = pipe.vae.latents_mean.view(1, -1, 1, 1, 1).to(device=tmp.device, dtype=tmp.dtype)
    latents_std = pipe.vae.latents_std.view(1, -1, 1, 1, 1).to(device=tmp.device, dtype=tmp.dtype)

    def encode_video_3d_from_f3hw_ltx(x_f3hw: torch.Tensor):
        # x_f3hw: (T,3,H,W) in [0,1]
        x_m11 = x_f3hw * 2.0 - 1.0
        
        x_m11 = x_m11.permute(1, 0, 2, 3).unsqueeze(0)      # (1,3,T,H,W)
        # print(x_m11.shape)
        z = pipe.vae.encode(x_m11).latent_dist.mean         # (1,C,T',Hp,Wp)  # same as your ES code uses mean
        z = (z - latents_mean) / latents_std                # LTX latent normalization
        return z                                            # (1,C,T',Hp,Wp)

    H, W = base01.shape[-2], base01.shape[-1]
    aux_samples = base01.new_zeros((0, 3, H, W))

    # -------------------------
    # (1) EditShield optimize for all video latents.
    # The first latent frame will be overwritten with the condition latent later.
    # -------------------------
    Fr = latent_frames
    eps_set = torch.tensor([16.0 * (i + 1) for i in range(Fr)], device=device, dtype=torch.float32) / 255.0
    step_v = 2.0 / 255.0
    steps_v = 1

    z_base = encode_video_3d_from_f3hw_ltx(base01)[:, :, :1]     # (1,C,1,Hp,Wp)

    if not apply_ES:
        Xv = torch.zeros((Fr, 3, base01.shape[-2], base01.shape[-1]), device=device, dtype=dtype)

    else:
        if Fr > 0:
            if Xv is None:
                Xv = base01.repeat(Fr, 1, 1, 1)
            noise_std = 1.0 / 255.0
            Xv = (Xv + noise_std * torch.randn_like(Xv)).clamp(0, 1).detach()
            Xv = Xv.to(device=device, dtype=dtype).detach()
            assert Xv.shape[0] == Fr, f"Xv has {Xv.shape[0]} frames but Fr={Fr}"

            delta_v = torch.zeros_like(Xv)
            loss_log = []

            for _ in range(steps_v):
                z_refs = []
                with torch.no_grad():
                    for j in range(Fr):
                        xj = Xv[j:j+1].to(dtype=dtype)
                        zj = encode_video_3d_from_f3hw_ltx(xj)[:, :, :1]
                        z_refs.append(zj.detach())

                for k in range(Fr):
                    xk = Xv[k:k+1].to(dtype=dtype)
                    dk = delta_v[k:k+1].detach().clone().to(dtype=dtype).requires_grad_(True)

                    xk_adv = (xk + dk).clamp(0, 1)
                    z = encode_video_3d_from_f3hw_ltx(xk_adv)[:, :, :1]

                    loss_base = ((z - z_base) ** 2).mean()

                    loss_nbr = 0.0
                    cnt = 0
                    loss_v = 1024 * (loss_base + loss_nbr)

                    gk = torch.autograd.grad(loss_v, dk, retain_graph=False, create_graph=False)[0]

                    with torch.no_grad():
                        dk = dk + step_v * gk.sign()

                        base_k = base01
                        delta_k = (xk + dk - base_k)

                        Xv[k:k+1] = (base_k + delta_k).clamp(0, 1).detach()
                        delta_v[k:k+1] = delta_k.detach()
                        loss_log.append(float(loss_v.detach().item()))
        else:
            Xv = base01.new_zeros((0, 3, base01.shape[-2], base01.shape[-1]))

    # -------------------------
    # (2) encode ES latents
    # -------------------------
    if Fr > 0:
        # Original:
        # x_tail_f3hw = Xv.repeat_interleave(latent_repeat_frames, dim=0)
        # x_tail_f3hw = torch.cat([x_tail_f3hw[:1].clone(), x_tail_f3hw], dim=0)
        # z_tail = encode_video_3d_from_f3hw_ltx(x_tail_f3hw)[:, :, :Fr]
        x_tail_short = Xv[:1].repeat(latent_repeat_frames, 1, 1, 1)
        x_tail_short = torch.cat([x_tail_short[:1].clone(), x_tail_short], dim=0)
        z_tail_unit = encode_video_3d_from_f3hw_ltx(x_tail_short)[:, :, :1]
        z_tail = z_tail_unit.repeat(1, 1, Fr, 1, 1)
    else:
        C = z_base.shape[1]
        Hp, Wp = z_base.shape[-2], z_base.shape[-1]
        z_tail = z_base.new_zeros((1, C, Fr, Hp, Wp))

    z_init = z_tail

    # -------------------------
    # (5) add noise at timestep t0
    # -------------------------
    eps_noise = torch.randn_like(z_init, device=device, dtype=dtype)

    noisy = pipe.scheduler.scale_noise(z_init, t0, eps_noise)

    noisy_in = noisy

    return noisy_in, z_base[:, :, :0], aux_samples, Xv

def i2vdetach_ltx(
    pipe,
    img01: torch.Tensor,
    *,
    steps: int,
    eps: float,
    step_size: float,
    num_frames: int,
    height: int,
    width: int,
    timesteps_total: int,
    noise_mode: str,
    fixed_t_index: int,
    latent_repeat_frames: int = 8,
    apply_ES: bool = True,
    apply_black: bool = False,
    use_only_first_video_frame: bool = False,
    args=None,
    prompt: str = "",
    use_abs: bool = False,
):
    device = img01.device
    dtype = pipe.transformer.dtype

    # prompt embeddings (same as before)
    with torch.no_grad():
        (
            prompt_embeds,
            prompt_attention_mask,
            _neg_prompt_embeds,
            _neg_prompt_attention_mask,
        ) = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt="",
            do_classifier_free_guidance=False,
            num_videos_per_prompt=1,
            max_sequence_length=128,
            device=device,
        )

    latent_num_frames = (num_frames - 1) // pipe.vae_temporal_compression_ratio + 1
    latent_height = height // pipe.vae_spatial_compression_ratio
    latent_width = width // pipe.vae_spatial_compression_ratio

    # scheduler timesteps (LTX)
    video_sequence_length = latent_num_frames * latent_height * latent_width
    sigmas = np.linspace(1.0, 1 / timesteps_total, timesteps_total)
    mu = calculate_shift(
        video_sequence_length,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.15),
    )
    pipe.scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
    ts = pipe.scheduler.timesteps

    # layer range (same)
    n_layers = len(pipe.transformer.transformer_blocks)
    layer_mid = pipe.transformer.transformer_blocks[n_layers // 2]
    layer_start = n_layers // 4
    layer_end = n_layers * 3 // 4

    # rope interpolation scale (same)
    rope_interpolation_scale = (
        pipe.vae_temporal_compression_ratio / 25,
        pipe.vae_spatial_compression_ratio,
        pipe.vae_spatial_compression_ratio,
    )

    # PGD variables
    img_f = img01.to(dtype=dtype)
    delta = torch.zeros_like(img_f, device=device, dtype=dtype, requires_grad=True)

    X_v = None

    # helper: LTX-style image -> normalized latent (B,C,1,H',W')
    def _encode_first_frame_ltx(img01_bchw: torch.Tensor) -> torch.Tensor:
        # img01_bchw: (B,3,H,W) in [0,1]
        x = img01_bchw * 2.0 - 1.0
        z = pipe.vae.encode(x.unsqueeze(2)).latent_dist.mode()  # (B,C,1,H',W')
        z = pipe._normalize_latents(
            z,
            pipe.vae.latents_mean,
            pipe.vae.latents_std,
            pipe.vae.config.scaling_factor,
        )
        return z

    for it in range(steps):
        img_adv01 = (img_f + delta).clamp(0, 1)

        # choose timestep
        seed_everything(42+it, deterministic=True)
        if noise_mode == "fixed_t":
            ti = max(0, min(fixed_t_index, len(ts) - 1))
            t0 = ts[ti : ti + 1]
        else:
            hi = max(1, min(fixed_t_index, len(ts)))
            ti = torch.randint(0, hi, (1,), device=device).item()
            t0 = ts[ti : ti + 1]

        # ------------------------------------------------------------
        # 1) Build ES video latents for every video slot.
        # The condition latent is injected after this step.
        # ------------------------------------------------------------
        base_for_eot = img_f 
        noisy_in, z_ori, aux_samples, X_v = make_noisy_video_latents_all_es_ltx(
            pipe,
            base_img01=base_for_eot,
            latent_frames=latent_num_frames,
            latent_repeat_frames=latent_repeat_frames,
            t0=t0,
            it=it,
            Xv=X_v,
            apply_ES=args.apply_ES,
        )
        # noisy_in: (B,C,F,H',W') normalized LTX latents (your helper output)

        # ------------------------------------------------------------
        # 2) LTX-style conditioning: overwrite FIRST latent frame with adv image latent
        #    (instead of Cog-style concat)
        # ------------------------------------------------------------
        adv_lat_1 = _encode_first_frame_ltx(img_adv01).to(noisy_in.dtype)  # (1,C,1,H',W')
        noisy_in = noisy_in.clone()
        noisy_in[:, :, :1, :, :] = adv_lat_1[:, :, :1, :, :]

        # ------------------------------------------------------------
        # 3) Pack latents exactly like LTX pipeline
        # ------------------------------------------------------------
        latent_model_input = pipe._pack_latents(
            noisy_in.to(prompt_embeds.dtype),
            pipe.transformer_spatial_patch_size,
            pipe.transformer_temporal_patch_size,
        )

        # ------------------------------------------------------------
        # 4) Build conditioning_mask exactly like LTX pipeline and mask timestep
        # ------------------------------------------------------------
        cond_mask = torch.zeros(
            (noisy_in.shape[0], 1, noisy_in.shape[2], noisy_in.shape[3], noisy_in.shape[4]),
            device=noisy_in.device,
            dtype=noisy_in.dtype,
        )
        cond_mask[:, :, 0] = 1.0  # first latent frame is conditioning frame

        conditioning_mask = pipe._pack_latents(
            cond_mask,
            pipe.transformer_spatial_patch_size,
            pipe.transformer_temporal_patch_size,
        ).squeeze(-1)  # (B,S)

        # LTX pipeline style timestep broadcast + mask
        timestep = t0.expand(latent_model_input.shape[0])
        timestep = timestep.unsqueeze(-1) * (1 - conditioning_mask.to(timestep.dtype))

        # transformer forward (same structure)
        _ = pipe.transformer(
            hidden_states=latent_model_input,
            encoder_hidden_states=prompt_embeds,
            timestep=timestep,
            encoder_attention_mask=prompt_attention_mask,
            num_frames=latent_num_frames,
            height=latent_height,
            width=latent_width,
            rope_interpolation_scale=rope_interpolation_scale,
            attention_kwargs=None,
            return_dict=False,
        )[0]

        # loss (same)
        loss_text, loss_img, loss_video = two_logits_ltx_self_split(
            pipe,
            latent_frames=latent_num_frames,
            layer_start=layer_start,
            layer_end=layer_end,
            use_abs=use_abs,
        )
        loss = loss_img - loss_video#*2

        # debug-safe grad call (recommend keep during debugging)
        grad = torch.autograd.grad(
            loss, delta, retain_graph=False, create_graph=False, allow_unused=False
        )[0]

        with torch.no_grad():
            delta -= step_size * grad.sign()
            delta.clamp_(-eps, eps)
            # optional valid-range clamp wrt original image:
            # delta.clamp_(-img_f, 1.0 - img_f)

        delta.grad = None
        torch.cuda.empty_cache()

        # cleanup (same spirit)
        if hasattr(layer_mid, "attn1") and hasattr(layer_mid.attn1, "temp_query"):
            layer_mid.attn1.temp_query = None
        if hasattr(layer_mid, "attn1") and hasattr(layer_mid.attn1, "temp_key"):
            layer_mid.attn1.temp_key = None
        if hasattr(layer_mid, "attn1") and hasattr(layer_mid.attn1, "temp_value"):
            layer_mid.attn1.temp_value = None

        for blk in pipe.transformer.transformer_blocks:
            if hasattr(blk, "attn1"):
                if hasattr(blk.attn1, "temp_query"):
                    blk.attn1.temp_query = None
                if hasattr(blk.attn1, "temp_key"):
                    blk.attn1.temp_key = None
                if hasattr(blk.attn1, "temp_value"):
                    blk.attn1.temp_value = None

        print(
            f"[{it+1:04d}/{steps}] total_loss={loss.item():.6f} | "
            f"img_loss={loss_img.item():.6f}, video_loss={loss_video.item():.6f}, "
            f"|delta|_inf={delta.abs().max().item():.5f} t_idx={ti}"
        )

    img_out = (img_f + delta).clamp(0, 1).detach()
    return img_out, delta.detach(), aux_samples, X_v
