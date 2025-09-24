#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
07_scan_sigma_grid_v5.py
- Scanne une grille de triplets (sigma1 ≤ sigma2 ≤ sigma3)
- Lance 06_weil_kernel_gaussian_arb_v5.py pour chaque triplet
- Récupère la marge Cholesky, la borne de Gershgorin et les chemins CSV/certificat
- Supporte --resume et --live
"""

from __future__ import annotations
import sys, os, re, csv, time, itertools, argparse, subprocess
from pathlib import Path

SCRIPT_V5 = Path(__file__).parent / "06_weil_kernel_gaussian_arb_v5.py"
OUT_DIR   = Path("outputs")
OUT_LOGS  = OUT_DIR / "logs"
OUT_MASTER= OUT_DIR / "scan_master_v5.csv"

def canon_triplet(sigmas):
    s = tuple(sorted(sigmas))
    return s

def name_from_sigmas(sigmas):
    return "v5_" + "_".join(str(s).replace(".", "p") for s in sigmas)

def load_done_set(csv_path: Path):
    done = set()
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    sig = (float(row["sigma1"]), float(row["sigma2"]), float(row["sigma3"]))
                    done.add(canon_triplet(sig))
                except Exception:
                    pass
    return done

def build_cmd(python_exe, script, sigmas, params, csv_path, cert_path):
    cmd = [
        python_exe, str(script),
        "--sigmas", *(str(s) for s in sigmas),
        "--flip", str(params.flip),
        "--u_scale", str(params.u_scale),
        "--dps", str(params.dps),
        "--max_dps", str(params.max_dps),
        "--quad_Lsig", str(params.quad_Lsig),
        "--quad_ppu", str(params.quad_ppu),
        "--pmax", str(params.pmax),
        "--tau", str(params.tau),
        "--csv", str(csv_path),
        "--cert", str(cert_path)
    ]
    if params.log_entries: cmd.append("--log_entries")
    if params.dump_terms:  cmd.append("--dump_terms")
    if params.progress:    cmd.append("--progress")
    if params.progress_bar: cmd.append("--progress_bar")
    return cmd

def parse_outputs(log_text: str):
    ok = None; chol_lb = None; g_lb = None
    m1 = re.search(r"Cholesky PSD certified\?\s+(True|False).*?chol_min_pivot_lb\s*=\s*([\-0-9.eE]+)", log_text)
    m2 = re.search(r"Best scaled Gershgorin lower bound:\s*([\-0-9.eE]+)", log_text)
    if m1:
        ok = (m1.group(1) == "True")
        chol_lb = float(m1.group(2))
    if m2:
        g_lb = float(m2.group(1))
    return ok, chol_lb, g_lb

def run_one(sigmas, params, idx, total):
    base = name_from_sigmas(sigmas)
    out_csv_one = OUT_DIR / f"gram_{base}.csv"
    out_cert    = OUT_DIR / f"cert_{base}.json"
    log_path    = OUT_LOGS / f"scan_{base}.log"
    cmd = build_cmd(sys.executable, SCRIPT_V5, sigmas, params, out_csv_one, out_cert)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    print(f"[{idx}/{total}] sigmas={sigmas} -> running...", flush=True)

    if getattr(params, "live", False):
        out_chunks = []
        OUT_LOGS.mkdir(parents=True, exist_ok=True)
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace",
                              bufsize=1, env=env) as proc:
            try:
                for line in iter(proc.stdout.readline, ""):
                    print(line, end="", flush=True)
                    out_chunks.append(line)
            finally:
                if proc.stdout: proc.stdout.close()
        text = "".join(out_chunks)
    else:
        text = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env).stdout

    OUT_LOGS.mkdir(parents=True, exist_ok=True)
    log_path.write_text(text, encoding="utf-8")

    ok, chol_lb, g_lb = parse_outputs(text)
    return ok, chol_lb, g_lb, out_csv_one, log_path, out_cert

def fmt_min(x):
    if x == float("inf"): return "∞"
    m = int(x // 60); s = int(x % 60)
    return f"{m:02d}:{s:02d}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--range", type=float, nargs=3, default=[0.6, 1.8, 0.2], help="start, stop, step")
    ap.add_argument("--dps", type=int, default=200)
    ap.add_argument("--max_dps", type=int, default=320)
    ap.add_argument("--quad_Lsig", type=float, default=12.0)
    ap.add_argument("--quad_ppu", type=int, default=600)
    ap.add_argument("--pmax", type=int, default=800000)
    ap.add_argument("--tau", type=float, default=1e-20)
    ap.add_argument("--flip", type=float, default=-1.0)
    ap.add_argument("--u_scale", type=float, default=1.0)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--log_entries", action="store_true")
    ap.add_argument("--dump_terms", action="store_true")
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--progress_bar", action="store_true")
    args = ap.parse_args()

    start, stop, step = args.range
    grid = []
    s_vals = []
    x = start
    while x <= stop + 1e-12:
        s_vals.append(round(x, 10))
        x += step
    for a in s_vals:
        for b in s_vals:
            for c in s_vals:
                t = canon_triplet((a,b,c))
                grid.append(t)
    grid = sorted(set(grid))
    done = load_done_set(OUT_MASTER) if args.resume else set()
    grid_run = [t for t in grid if t not in done]
    total = len(grid_run)
    if total == 0:
        print("[Info] Rien à faire (grille vide ou tout déjà présent).")
        return

    new_file = not OUT_MASTER.exists()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    f = OUT_MASTER.open("a", newline="", encoding="utf-8")
    wr = csv.writer(f)
    if new_file:
        wr.writerow(["sigma1","sigma2","sigma3","chol_ok","chol_min_pivot_lb",
                     "gersh_scaled_lb","gram_csv","log_file","cert_file"])

    class P: pass
    params = P()
    params.dps=args.dps; params.max_dps=args.max_dps
    params.quad_Lsig=args.quad_Lsig; params.quad_ppu=args.quad_ppu
    params.pmax=args.pmax; params.tau=args.tau
    params.flip=args.flip; params.u_scale=args.u_scale
    params.dump_terms=args.dump_terms; params.log_entries=args.log_entries
    params.progress=args.progress; params.progress_bar=args.progress_bar
    params.live=args.live

    print(f"-> Lancement {total} runs ; range={args.range} ; u_scale={args.u_scale} ; jobs={args.jobs}")
    start_t = time.time()

    # Sequential (jobs=1) to keep things simple; you can add ThreadPoolExecutor if desired.
    done_ct = 0
    for i, sig in enumerate(grid_run, 1):
        ok, chol_lb, g_lb, csv_one, log_one, cert_one = run_one(sig, params, i, total)
        wr.writerow([sig[0], sig[1], sig[2], ok, chol_lb, g_lb, str(csv_one), str(log_one), str(cert_one)])
        f.flush(); os.fsync(f.fileno())
        done_ct += 1
        elapsed = time.time() - start_t
        rate = done_ct / elapsed if elapsed > 0 else 0.0
        remaining = (total - done_ct) / rate if rate > 0 else float("inf")
        pct = 100.0 * done_ct / total
        print(f"[global] {done_ct}/{total} ({pct:5.1f}%) | elapsed {fmt_min(elapsed)} | ETA ~ {fmt_min(remaining)}", flush=True)

    f.close()
    print(f"\nRésumé écrit dans : {OUT_MASTER}")

if __name__ == "__main__":
    main()
