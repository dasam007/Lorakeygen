#!/usr/bin/env python3
# keygen_gui_indoor.py
# Double-clickable GUI for indoor multipath keygen metrics (no terminal flags).
import io, os, re, sys, json, math
from typing import Tuple, Optional
import numpy as np

# --- soft deps ---
try:
    import pandas as pd
except Exception as e:
    raise SystemExit("This app needs pandas. Install with: pip install pandas\n" + str(e))
try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

APP_TITLE = "Indoor Multipath Keygen — GUI"

# ---------------------- I/O helpers ----------------------
def robust_read_csv(path: str) -> Tuple["pd.DataFrame", str]:
    """Reads an idx,rssi CSV even if encoding/bytes are messy. Returns (DataFrame, status_string)."""
    enc_try = ["utf-8", "utf-8-sig", "cp1252", "latin1", "utf-16", "utf-16le", "utf-16be"]
    for enc in enc_try:
        try:
            df = pd.read_csv(path, encoding=enc)
            if set(df.columns) >= {"idx", "rssi"}:
                return df, f"ok:{enc}"
        except Exception:
            pass
    # Fallback byte-level salvage
    with open(path, "rb") as f:
        raw = f.read()
    cleaned = re.sub(rb"[^\d,\.\-\r\n]", b" ", raw)
    if not cleaned.startswith(b"idx"):
        cleaned = b"idx,rssi\n" + cleaned
    df = pd.read_csv(io.BytesIO(cleaned))
    return df, "salvaged-bytes"

def read_bits_txt(path: str) -> np.ndarray:
    """Reads a file containing 0/1 characters and returns np.uint8 array."""
    with open(path, "rb") as f:
        s = f.read().decode("utf-8", errors="ignore")
    s = "".join(ch for ch in s if ch in "01")
    if not s:
        return np.array([], dtype=np.uint8)
    return np.fromiter((1 if ch == "1" else 0 for ch in s), dtype=np.uint8, count=len(s))

# ---------------------- stats + metrics ----------------------
def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

def kdr_from_bits(a_bits: np.ndarray, b_bits: np.ndarray):
    """Returns dict with len_a, len_b, common, mismatches, kdr."""
    n = min(len(a_bits), len(b_bits))
    if n == 0:
        return dict(len_a=len(a_bits), len_b=len(b_bits), common=0, mismatches=0, kdr=float("nan"))
    mism = int(np.count_nonzero(a_bits[:n] ^ b_bits[:n]))
    return dict(len_a=len(a_bits), len_b=len(b_bits), common=n, mismatches=mism, kdr=(mism / n))

def guardband_quantize_midpoint(a_rssi: np.ndarray, b_rssi: np.ndarray, k: float) -> dict:
    """
    Device-matched guard-band with midpoint rule:
    - Keep when Alice va <= lo -> bitA=0, or va >= hi -> bitA=1.
    - Bob:
        vb >= hi -> 1
        vb <= lo -> 0
        else use midpoint mid=(lo+hi)/2: vb >= mid -> 1, else 0
    """
    vals = a_rssi[~np.isnan(a_rssi)]
    mu = float(np.mean(vals))
    sd = float(np.std(vals, ddof=0))  # population σ
    lo, hi = mu - k * sd, mu + k * sd
    mid = 0.5 * (lo + hi)

    kept_idx, bitsA, bitsB = [], [], []
    for i, va in enumerate(a_rssi):
        vb = b_rssi[i]
        if np.isnan(va) or np.isnan(vb):
            continue
        if va <= lo:
            kept_idx.append(i)
            bitsA.append(0)
            bitsB.append(0 if vb <= lo else (1 if vb >= hi else (1 if vb >= mid else 0)))
        elif va >= hi:
            kept_idx.append(i)
            bitsA.append(1)
            bitsB.append(1 if vb >= hi else (0 if vb <= lo else (1 if vb >= mid else 0)))
        # else: inside guard-band -> skip

    return {
        "mu": mu, "sd": sd, "lo": lo, "hi": hi,
        "kept_idx": np.array(kept_idx, dtype=int),
        "bitsA": np.array(bitsA, dtype=np.uint8),
        "bitsB": np.array(bitsB, dtype=np.uint8),
    }

# -------- Differential helpers: smoothing & lag (+ quantizer) --------
def moving_average_same(x: np.ndarray, w: int) -> np.ndarray:
    """Centered moving average with 'same' length; w=0 or 1 returns x."""
    if w is None or w <= 1:
        return x.copy()
    kernel = np.ones(int(w), dtype=float) / float(w)
    return np.convolve(x, kernel, mode="same")

def align_with_lag(a: np.ndarray, b: np.ndarray, lag: int):
    """
    Align arrays so that b is shifted by 'lag' relative to a.
    lag > 0: b earlier -> drop last 'lag' from b, first 'lag' from a.
    lag < 0: b later  -> drop first '-lag' from b, last '-lag' from a.
    """
    if lag > 0:
        return a[lag:], b[:-lag]
    elif lag < 0:
        L = -lag
        return a[:-L], b[L:]
    else:
        return a, b

def differential_quantize(a_rssi: np.ndarray, b_rssi: np.ndarray, eps_db: float):
    """Δ-based sign quantizer with ε: keep only when |ΔA|>=ε and |ΔB|>=ε."""
    a_bits, b_bits = [], []
    for i in range(1, len(a_rssi)):
        va0, vb0 = a_rssi[i-1], b_rssi[i-1]
        va1, vb1 = a_rssi[i],   b_rssi[i]
        if any(np.isnan([va0, vb0, va1, vb1])):
            continue
        da, db = va1 - va0, vb1 - vb0
        if abs(da) < eps_db or abs(db) < eps_db:
            continue
        a_bits.append(1 if da > 0 else 0)
        b_bits.append(1 if db > 0 else 0)
    return np.array(a_bits, dtype=np.uint8), np.array(b_bits, dtype=np.uint8)

def kgr_bits(len_bits: int, probes: Optional[int], duration_s: Optional[float]):
    """Returns (bits_per_probe, bits_per_second). Matches your device logging: / (probes - 1)."""
    bpp = float("nan")
    bps = float("nan")
    if probes is not None and probes > 1:
        bpp = len_bits / (probes - 1)
    if duration_s is not None and duration_s > 0:
        bps = len_bits / duration_s
    return bpp, bps

def expected_drop_rate(p: float, n: int, t: int) -> float:
    """Binomial drop probability for blocks with > t errors (CRC fail / uncorrectable)."""
    from math import comb
    if not (0.0 <= p <= 1.0):
        return float("nan")
    keep = 0.0
    for i in range(0, t + 1):
        keep += comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    return 1.0 - keep

# ---------------------- plotting ----------------------
def save_plots(a_rssi, b_rssi, outdir):
    if not HAVE_MPL:
        return None, None
    os.makedirs(outdir, exist_ok=True)
    x = np.arange(len(a_rssi))
    m = ~(np.isnan(a_rssi) | np.isnan(b_rssi))
    a = a_rssi[m]; b = b_rssi[m]; xi = x[m]

    plt.figure()
    plt.plot(xi, a, label="Alice RSSI")
    plt.plot(xi, b, label="Bob RSSI")
    plt.xlabel("Index"); plt.ylabel("RSSI (dBm)")
    plt.title("RSSI vs Index — Alice & Bob")
    plt.legend(); plt.tight_layout()
    p1 = os.path.join(outdir, "rssi_vs_index.png")
    plt.savefig(p1); plt.close()

    plt.figure()
    plt.scatter(a, b, s=6)
    plt.xlabel("Alice RSSI (dBm)"); plt.ylabel("Bob RSSI (dBm)")
    plt.title("RSSI Scatter — Reciprocity")
    plt.tight_layout()
    p2 = os.path.join(outdir, "rssi_scatter.png")
    plt.savefig(p2); plt.close()
    return p1, p2

# ---------------------- GUI App ----------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("860x720")
        self.minsize(860, 720)

        # Defaults (you can change in the UI)
        self.default_outdir = os.path.join(os.getcwd(), "out")
        os.makedirs(self.default_outdir, exist_ok=True)

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 6, "pady": 4}

        frm = ttk.Frame(self)
        frm.pack(fill="both", expand=True)

        # File pickers
        row = 0
        self.alice_var = tk.StringVar(value=self._auto_path("alice_rssi.csv"))
        self.bob_var   = tk.StringVar(value=self._auto_path("bob_rssi.csv"))
        self.alice_pre_var = tk.StringVar(value=self._auto_path("alice_pre_bits.txt", must_exist=False))
        self.bob_pre_var   = tk.StringVar(value=self._auto_path("bob_pre_bits.txt", must_exist=False))
        self.outdir_var    = tk.StringVar(value=self.default_outdir)

        def add_file_row(label, var, types, must_exist=True):
            nonlocal row
            ttk.Label(frm, text=label).grid(row=row, column=0, sticky="e", **pad)
            ent = ttk.Entry(frm, textvariable=var, width=70)
            ent.grid(row=row, column=1, sticky="we", **pad)
            def browse():
                p = filedialog.askopenfilename(title=label, filetypes=types)
                if p: var.set(p)
            ttk.Button(frm, text="Browse…", command=browse).grid(row=row, column=2, **pad)
            row += 1

        add_file_row("Alice CSV (idx,rssi):", self.alice_var, [("CSV files","*.csv"), ("All","*.*")])
        add_file_row("Bob CSV (idx,rssi):",   self.bob_var,   [("CSV files","*.csv"), ("All","*.*")])
        add_file_row("Alice pre-bits (optional):", self.alice_pre_var, [("Text","*.txt"), ("All","*.*")], must_exist=False)
        add_file_row("Bob pre-bits (optional):",   self.bob_pre_var,   [("Text","*.txt"), ("All","*.*")], must_exist=False)

        # Parameters
        self.k_var        = tk.StringVar(value="0.60")
        self.eps_var      = tk.StringVar(value="2.0")
        self.diff_win_var = tk.StringVar(value="9")     # NEW: smoothing window (samples)
        self.diff_lag_var = tk.StringVar(value="-1")    # NEW: lag in samples (Bob vs Alice)
        self.n_var        = tk.StringVar(value="15")
        self.t_var        = tk.StringVar(value="1")
        self.probes_var   = tk.StringVar(value="2000")
        self.duration_var = tk.StringVar(value="1210")

        ttk.Label(frm, text="Guard-band k (μ±kσ):").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.k_var, width=12).grid(row=row, column=1, sticky="w", **pad); row+=1

        # Differential parameters block
        ttk.Label(frm, text="Differential ε (dB):").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.eps_var, width=12).grid(row=row, column=1, sticky="w", **pad); row+=1

        ttk.Label(frm, text="Diff smoothing window (samples, 0=off):").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.diff_win_var, width=12).grid(row=row, column=1, sticky="w", **pad); row+=1

        ttk.Label(frm, text="Diff lag (samples, Bob vs Alice):").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.diff_lag_var, width=12).grid(row=row, column=1, sticky="w", **pad); row+=1

        ttk.Label(frm, text="ECC n:").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.n_var, width=12).grid(row=row, column=1, sticky="w", **pad); row+=1
        ttk.Label(frm, text="ECC t:").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.t_var, width=12).grid(row=row, column=1, sticky="w", **pad); row+=1
        ttk.Label(frm, text="Probes (optional):").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.probes_var, width=12).grid(row=row, column=1, sticky="w", **pad); row+=1
        ttk.Label(frm, text="Duration sec (optional):").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.duration_var, width=12).grid(row=row, column=1, sticky="w", **pad); row+=1

        # Outdir
        ttk.Label(frm, text="Output folder:").grid(row=row, column=0, sticky="e", **pad)
        out_entry = ttk.Entry(frm, textvariable=self.outdir_var, width=70)
        out_entry.grid(row=row, column=1, sticky="we", **pad)
        ttk.Button(frm, text="Choose…", command=self._choose_outdir).grid(row=row, column=2, **pad)
        row += 1

        # Options
        self.make_plots_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text="Generate plots (requires matplotlib)", variable=self.make_plots_var).grid(row=row, column=1, sticky="w", **pad)
        row += 1

        # Buttons
        btn_frame = ttk.Frame(frm)
        btn_frame.grid(row=row, column=0, columnspan=3, sticky="we", **pad)
        ttk.Button(btn_frame, text="Run Analysis", command=self.run_analysis).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Open Output Folder", command=self._open_outdir).pack(side="left", padx=4)
        row += 1

        # Results area
        ttk.Label(frm, text="Results:").grid(row=row, column=0, sticky="ne", **pad)
        self.out_text = ScrolledText(frm, height=18, wrap="word")
        self.out_text.grid(row=row, column=1, columnspan=2, sticky="nsew", **pad)
        row += 1

        # Grid weights
        frm.columnconfigure(1, weight=1)
        frm.rowconfigure(row-1, weight=1)

    def _auto_path(self, name: str, must_exist: bool = True) -> str:
        p = os.path.join(os.getcwd(), name)
        if must_exist and not os.path.exists(p):
            return ""
        return p

    def _choose_outdir(self):
        p = filedialog.askdirectory(title="Choose output folder", initialdir=self.default_outdir)
        if p:
            self.outdir_var.set(p)

    def _open_outdir(self):
        p = self.outdir_var.get().strip() or self.default_outdir
        os.makedirs(p, exist_ok=True)
        if sys.platform.startswith("win"):
            os.startfile(p)  # type: ignore
        elif sys.platform == "darwin":
            os.system(f'open "{p}"')
        else:
            os.system(f'xdg-open "{p}"')

    def _append(self, line: str = ""):
        self.out_text.insert("end", line + "\n")
        self.out_text.see("end")
        self.update_idletasks()

    def run_analysis(self):
        # Read params
        alice = self.alice_var.get().strip()
        bob   = self.bob_var.get().strip()
        alice_pre = self.alice_pre_var.get().strip()
        bob_pre   = self.bob_pre_var.get().strip()
        outdir = self.outdir_var.get().strip() or self.default_outdir
        os.makedirs(outdir, exist_ok=True)

        if not alice or not os.path.exists(alice):
            messagebox.showerror(APP_TITLE, "Please select a valid Alice CSV (idx,rssi).")
            return
        if not bob or not os.path.exists(bob):
            messagebox.showerror(APP_TITLE, "Please select a valid Bob CSV (idx,rssi).")
            return

        def parse_float(var, default=None):
            try:
                return float(var.get().strip())
            except Exception:
                return default

        def parse_int(var, default=None):
            try:
                return int(float(var.get().strip()))
            except Exception:
                return default

        k = parse_float(self.k_var, 0.60)
        eps = parse_float(self.eps_var, 2.0)
        diff_win = parse_int(self.diff_win_var, 0)   # NEW
        diff_lag = parse_int(self.diff_lag_var, 0)   # NEW
        n = parse_int(self.n_var, 15)
        t = parse_int(self.t_var, 1)
        probes = parse_int(self.probes_var, None)
        duration = parse_float(self.duration_var, None)

        # Load CSVs
        try:
            a_df, a_stat = robust_read_csv(alice)
            b_df, b_stat = robust_read_csv(bob)
        except Exception as e:
            messagebox.showerror(APP_TITLE, f"Failed to read CSVs:\n{e}")
            return

        m = min(len(a_df), len(b_df))
        a_df = a_df.iloc[:m].copy()
        b_df = b_df.iloc[:m].copy()

        a_rssi = pd.to_numeric(a_df["rssi"], errors="coerce").to_numpy()
        b_rssi = pd.to_numeric(b_df["rssi"], errors="coerce").to_numpy()

        mask = ~(np.isnan(a_rssi) | np.isnan(b_rssi))
        a_clean = a_rssi[mask]; b_clean = b_rssi[mask]
        pairs = len(a_clean)

        # Compute metrics
        r = pearson(a_clean, b_clean)

        # Guard-band (device-matched)
        gb = guardband_quantize_midpoint(a_clean, b_clean, k=k)
        kept = len(gb["kept_idx"])
        kdr_gb = kdr_from_bits(gb["bitsA"], gb["bitsB"])
        bpp_gb, bps_gb = kgr_bits(kept, probes if probes else pairs, duration)

        # Differential with smoothing + lag (NEW)
        a_s = moving_average_same(a_clean, diff_win)
        b_s = moving_average_same(b_clean, diff_win)
        a_al, b_al = align_with_lag(a_s, b_s, diff_lag)
        a_diff, b_diff = differential_quantize(a_al, b_al, eps_db=eps)
        kdr_diff = kdr_from_bits(a_diff, b_diff)
        bpp_diff, bps_diff = kgr_bits(len(a_diff), probes if probes else pairs, duration)

        # Pre-bits (authoritative), if provided
        kdr_pre = None
        if alice_pre and bob_pre and os.path.exists(alice_pre) and os.path.exists(bob_pre):
            a_pre = read_bits_txt(alice_pre); b_pre = read_bits_txt(bob_pre)
            kdr_pre = kdr_from_bits(a_pre, b_pre)

        # ECC expectation (uses pre-bits KDR if available, else guard-band)
        if kdr_pre is not None and not math.isnan(kdr_pre["kdr"]):
            p_err = kdr_pre["kdr"]
            pre_bits_len = kdr_pre["common"]
        else:
            p_err = kdr_gb["kdr"]
            pre_bits_len = kept

        drop_prob = expected_drop_rate(p_err, n, t)
        blocks = math.ceil(pre_bits_len / n)
        exp_final_blocks = int(round(blocks * (1.0 - drop_prob)))
        exp_final_bits   = exp_final_blocks * n

        # Plots
        p1 = p2 = None
        if self.make_plots_var.get() and HAVE_MPL:
            p1, p2 = save_plots(a_clean, b_clean, outdir)

        # Build summaries
        summary = {
            "csv_parse": {"alice": a_stat, "bob": b_stat},
            "pairs_after_clean": pairs,
            "pearson_r": r,
            "guardband": {
                "mu": gb["mu"], "sd": gb["sd"], "lo": gb["lo"], "hi": gb["hi"],
                "kept_count": kept,
                "kept_fraction": (kept / pairs) if pairs else float("nan"),
                "kdr_percent": (kdr_gb["kdr"] * 100.0) if not math.isnan(kdr_gb["kdr"]) else float("nan"),
                "bits_per_probe": bpp_gb,
                "bits_per_second": bps_gb,
            },
            "differential": {
                "epsilon_db": eps,
                "smooth_win": diff_win,
                "lag": diff_lag,
                "bitsA_len": int(len(a_diff)), "bitsB_len": int(len(b_diff)),
                "kdr_percent": (kdr_diff["kdr"] * 100.0) if not math.isnan(kdr_diff["kdr"]) else float("nan"),
                "bits_per_probe": bpp_diff,
                "bits_per_second": bps_diff,
            },
            "pre_bits_if_provided": (
                None if kdr_pre is None else {
                    "alice_len": kdr_pre["len_a"], "bob_len": kdr_pre["len_b"], "common_len": kdr_pre["common"],
                    "kdr_percent": kdr_pre["kdr"] * 100.0
                }
            ),
            "ecc_expectation": {
                "n": n, "t": t,
                "p_err_used": p_err,
                "drop_prob_estimate": drop_prob,
                "blocks_from_pre": blocks,
                "expected_final_blocks": exp_final_blocks,
                "expected_final_bits": exp_final_bits
            },
            "outputs": {
                "summary_csv": os.path.join(outdir, "indoor_metrics_summary.csv"),
                "summary_json": os.path.join(outdir, "indoor_metrics_summary.json"),
                "rssi_vs_index_plot": p1,
                "rssi_scatter_plot": p2
            }
        }

        # Save outputs
        rows = [
            ("pairs_after_clean", pairs),
            ("pearson_r", r),
            ("gb_mu", gb["mu"]), ("gb_sd", gb["sd"]),
            ("gb_lo", gb["lo"]), ("gb_hi", gb["hi"]),
            ("gb_kept", kept),
            ("gb_kept_fraction", (kept / pairs) if pairs else float("nan")),
            ("gb_kdr_percent", (kdr_gb["kdr"] * 100.0) if not math.isnan(kdr_gb["kdr"]) else float("nan")),
            ("gb_bits_per_probe", summary["guardband"]["bits_per_probe"]),
            ("gb_bits_per_second", summary["guardband"]["bits_per_second"]),
            ("diff_eps_db", eps),
            ("diff_smooth_win", diff_win),
            ("diff_lag", diff_lag),
            ("diff_bitsA_len", len(a_diff)),
            ("diff_bitsB_len", len(b_diff)),
            ("diff_kdr_percent", (kdr_diff["kdr"] * 100.0) if not math.isnan(kdr_diff["kdr"]) else float("nan")),
            ("diff_bits_per_probe", summary["differential"]["bits_per_probe"]),
            ("diff_bits_per_second", summary["differential"]["bits_per_second"]),
            ("ecc_n", n), ("ecc_t", t),
            ("ecc_p_err_used", p_err),
            ("ecc_drop_prob_estimate", drop_prob),
            ("ecc_blocks_from_pre", blocks),
            ("ecc_expected_final_blocks", exp_final_blocks),
            ("ecc_expected_final_bits", exp_final_bits),
        ]
        if kdr_pre is not None:
            rows += [
                ("pre_alice_len", kdr_pre["len_a"]),
                ("pre_bob_len",   kdr_pre["len_b"]),
                ("pre_common_len",kdr_pre["common"]),
                ("pre_kdr_percent", kdr_pre["kdr"] * 100.0),
            ]
        df_out = pd.DataFrame(rows, columns=["Metric", "Value"])
        os.makedirs(outdir, exist_ok=True)
        df_out.to_csv(summary["outputs"]["summary_csv"], index=False)
        with open(summary["outputs"]["summary_json"], "w") as f:
            json.dump(summary, f, indent=2)

        # Print to UI
        self.out_text.delete("1.0", "end")
        self._append("=== Indoor Multipath Keygen Summary ===")
        self._append(f"Pairs after clean: {pairs}")
        self._append(f"Pearson r: {r:.3f}")
        self._append(f"[Guard-band k={k:.2f}] mu={gb['mu']:.2f}, sd={gb['sd']:.2f}, lo={gb['lo']:.2f}, hi={gb['hi']:.2f}")
        kept_pct = (kept / pairs * 100) if pairs else float("nan")
        self._append(f"  Kept: {kept}/{pairs} ({kept_pct:.2f}%)")
        if not math.isnan(kdr_gb["kdr"]):
            self._append(f"  KDR_gb: {kdr_gb['kdr']*100:.3f}%")
        if probes or duration:
            bpp, bps = summary["guardband"]["bits_per_probe"], summary["guardband"]["bits_per_second"]
            self._append(f"  KGR_gb: {bpp:.4f} bits/probe, {bps:.2f} bits/s")

        if not math.isnan(kdr_diff["kdr"]):
            self._append(f"[Differential eps={eps:.2f} dB, win={diff_win}, lag={diff_lag}] "
                         f"KDR_diff: {kdr_diff['kdr']*100:.3f}% (bits: {kdr_diff['common']})")
            if probes or duration:
                self._append(f"  KGR_diff: {bpp_diff:.4f} bits/probe, {bps_diff:.2f} bits/s")
        else:
            self._append(f"[Differential eps={eps:.2f} dB, win={diff_win}, lag={diff_lag}] produced 0 bits — "
                         f"relax ε or adjust window/lag.")

        if kdr_pre is not None:
            self._append(f"[Pre-bits] common={kdr_pre['common']}, KDR_pre={kdr_pre['kdr']*100:.3f}%")

        self._append(f"[ECC expectation] n={n}, t={t}, p_err_used={p_err:.5f}")
        self._append(f"  drop_prob≈{drop_prob*100:.2f}% → expected final bits ≈ {exp_final_bits} (from {blocks} blocks × {n})")
        self._append("")
        self._append("Saved:")
        self._append(f"  {summary['outputs']['summary_csv']}")
        self._append(f"  {summary['outputs']['summary_json']}")
        if p1: self._append(f"  {summary['outputs']['rssi_vs_index_plot']}")
        if p2: self._append(f"  {summary['outputs']['rssi_scatter_plot']}")

        messagebox.showinfo(APP_TITLE, "Analysis complete.\nSee results area and output files.")

# ---------------------- main ----------------------
if __name__ == "__main__":
    # lazy import here to avoid hard fail before GUI shows
    import pandas as pd  # type: ignore
    app = App()
    app.mainloop()
