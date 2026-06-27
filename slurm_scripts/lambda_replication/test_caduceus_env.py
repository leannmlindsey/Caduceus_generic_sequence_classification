#!/usr/bin/env python3
"""
test_caduceus_env.py — verify a conda env can actually RUN Caduceus on this node,
not just import the packages. Run it with the env's python ON A GPU NODE:

    conda activate /u/llindsey1/miniconda3/envs/<caduceus_env_or_CADUCEUS_3>
    python test_caduceus_env.py
    # optional: also test loading the real checkpoint
    python test_caduceus_env.py --ckpt-dir /work/hdd/bfzj/llindsey1/LAMBDA_REPLICATION/Caduceus_generic_sequence_classification/checkpoints/caduceus-ps_seqlen-8k_d_model-256_n_layer-4_lr-8e-3

Why a forward pass: mamba-ssm / causal-conv1d ship compiled CUDA kernels. On the
wrong arch (e.g. not built for GH200 sm_90) they IMPORT fine but crash at the
first kernel call. A tiny Mamba forward on GPU is the only honest check.

Exit code = number of FAILED checks (0 = env is good to run Caduceus).
"""
import argparse
import os
import sys
import traceback

fails = 0
def ok(msg):  print(f"  [ OK ] {msg}")
def bad(msg):
    global fails
    fails += 1
    print(f"  [FAIL] {msg}")


def imp(modname, attr_for_version="__version__"):
    try:
        m = __import__(modname)
        v = getattr(m, attr_for_version, "?")
        ok(f"import {modname:<18} {v}")
        return m
    except Exception as e:
        bad(f"import {modname:<18} {type(e).__name__}: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default=os.environ.get("CKPT_DIR", ""),
                    help="dir with model_config.json + checkpoints/last.ckpt (optional)")
    args = ap.parse_args()

    print("=" * 70)
    print("Caduceus env check")
    print(f"  python: {sys.executable}")
    print(f"  version: {sys.version.split()[0]}")
    print(f"  CONDA_PREFIX: {os.environ.get('CONDA_PREFIX', '(none)')}")
    print("=" * 70)

    print("\n[1] core imports")
    torch = imp("torch")
    imp("numpy"); imp("pandas"); imp("sklearn", "__version__")
    imp("transformers")
    pl = imp("pytorch_lightning")
    imp("mamba_ssm"); imp("causal_conv1d")
    # the model package itself (may be the local repo, not pip)
    try:
        import caduceus  # noqa
        ok("import caduceus          (package present)")
    except Exception as e:
        print(f"  [warn] import caduceus     {type(e).__name__}: {e} "
              "(ok if loaded from the repo via PYTHONPATH at runtime)")

    if torch is None:
        print("\ntorch missing — cannot continue."); return fails

    print("\n[2] CUDA / GPU")
    if torch.cuda.is_available():
        dev = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        ok(f"cuda available: {dev}  sm_{cap[0]}{cap[1]}  (torch cuda {torch.version.cuda})")
        if cap[0] < 9:
            print(f"  [warn] compute capability {cap} < 9.0 — not a GH200 (run on a ghx4 node?)")
    else:
        bad("torch.cuda.is_available() is False — run this on a GPU node (srun ... --gpus-per-node=1)")
        print("       (skipping the kernel forward-pass test without a GPU)")
        return fails

    print("\n[3] Mamba forward pass on GPU (exercises mamba-ssm + causal-conv1d kernels)")
    try:
        from mamba_ssm import Mamba
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            try:
                x = torch.randn(2, 16, 64, device="cuda", dtype=dtype)
                blk = Mamba(d_model=64, d_state=16, d_conv=4, expand=2).to("cuda", dtype=dtype)
                y = blk(x)
                torch.cuda.synchronize()
                assert y.shape == x.shape
                ok(f"Mamba forward works in {str(dtype).replace('torch.','')}  (out {tuple(y.shape)})")
                break
            except Exception as e:
                print(f"  [warn] Mamba forward failed in {dtype}: {type(e).__name__}: {e}")
        else:
            bad("Mamba forward failed in ALL dtypes — kernels not usable on this arch")
    except Exception:
        bad("could not even construct a Mamba block:")
        traceback.print_exc()

    if args.ckpt_dir:
        print("\n[4] checkpoint load")
        cfg = os.path.join(args.ckpt_dir, "model_config.json")
        ckpt = os.path.join(args.ckpt_dir, "checkpoints", "last.ckpt")
        if os.path.isfile(cfg):
            ok(f"model_config.json present ({cfg})")
        else:
            bad(f"model_config.json MISSING ({cfg})")
        if os.path.isfile(ckpt):
            try:
                sd = torch.load(ckpt, map_location="cpu")
                keys = list(sd.keys()) if isinstance(sd, dict) else []
                is_lightning = "state_dict" in keys or "pytorch-lightning_version" in keys
                ok(f"last.ckpt loads ({os.path.getsize(ckpt)//1024//1024} MB; "
                   f"lightning_ckpt={is_lightning}; top keys={keys[:5]})")
            except Exception as e:
                bad(f"last.ckpt failed to load: {type(e).__name__}: {e}")
        else:
            bad(f"last.ckpt MISSING ({ckpt})")

    print("\n" + "=" * 70)
    print(f"RESULT: {'ALL CHECKS PASSED — env is good for Caduceus' if fails == 0 else str(fails)+' CHECK(S) FAILED'}")
    print("=" * 70)
    return fails


if __name__ == "__main__":
    sys.exit(main())
