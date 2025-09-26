#!/usr/bin/env python3
import threading, time, math, struct, hashlib, os, sys, queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

# project deps
import bch
import sx126x

# --------------------- constants / defaults ---------------------
PROBE_COUNT_DEFAULT = 50
PROBE_GAP_DEFAULT   = 0.50
IDLE_LIMIT_ECC      = 40.0
END_GRACE_SEC       = 0.6
PORT_DEFAULT        = "/dev/ttyS0"
FREQ_MHZ_DEFAULT    = 868
AIR_SPEED_DEFAULT   = 1200
POWER_DBM_DEFAULT   = 22
BUFFER_SIZE_DEFAULT = 240
ADDR_ALICE_DEFAULT  = 0x0000
ADDR_BOB_DEFAULT    = 0x0001
HAM_N               = 31

# --------------------- FAST, THROTTLED, THREAD-SAFE PLOT ---------------------
from collections import deque

class PlotCanvas(ttk.Frame):
    def __init__(self, parent, width=860, height=260, y_pad=2.0, fps=15, rescale_hysteresis=1.5):
        super().__init__(parent)
        self.w, self.h = width, height
        self.margin_l, self.margin_r = 48, 12
        self.margin_t, self.margin_b = 20, 34
        self.c = tk.Canvas(self, width=self.w, height=self.h, bg="#ffffff", highlightthickness=0)
        self.c.pack(fill="x", expand=False)

        # style
        self.col_grid  = "#e6e6e6"
        self.col_frame = "#222222"
        self.col_text  = "#111111"
        self.col_pts   = "#d0021b"   # red
        self.col_line  = "#0050c8"   # blue
        self.r_pt      = 3.2

        # data & caches
        self.n_total = 0
        self.points  = []     # floats or None
        self.items_p = []     # point ovals
        self.items_l = []     # line segments
        self.ymin = None
        self.ymax = None
        self.y_pad = float(y_pad)

        # perf: throttle & queue
        self.fps = max(5, int(fps))
        self.rescale_hysteresis = float(rescale_hysteresis)
        self._q = deque()
        self._q_lock = threading.Lock()
        self._scheduled = False

        self._draw_axes(first=True)

    # public api (thread-safe)
    def reset(self, n_total):
        self.n_total = max(1, int(n_total))
        self.points  = [None] * self.n_total
        self.items_p = [None] * self.n_total
        self.items_l = [None] * max(0, self.n_total - 1)
        self.ymin, self.ymax = None, None
        self.c.delete("all")
        self._draw_axes(first=True)

    def post_point(self, idx, rssi):
        if not (0 <= idx < max(1, self.n_total)):
            return
        with self._q_lock:
            self._q.append((idx, float(rssi)))
            if not self._scheduled:
                self._scheduled = True
                self.after(int(1000 / self.fps), self._drain)

    def _drain(self):
        with self._q_lock:
            batch = list(self._q)
            self._q.clear()
            self._scheduled = False
        if not batch:
            return
        for idx, rssi in batch:
            self._add_or_update(idx, rssi)

    # ---------- drawing ----------
    def _x(self, i):
        L, R = self.margin_l, self.w - self.margin_r
        frac = 0 if self.n_total <= 1 else i / (self.n_total - 1)
        return L + frac * (R - L)

    def _y(self, v):
        T, B = self.margin_t, self.h - self.margin_b
        if self.ymin is None or self.ymax is None or self.ymax == self.ymin:
            return (T + B) / 2
        frac = (v - self.ymin) / (self.ymax - self.ymin)
        return B - frac * (B - T)

    def _draw_axes(self, first=False):
        c = self.c
        L, R = self.margin_l, self.w - self.margin_r
        T, B = self.margin_t, self.h - self.margin_b
        for k in range(11):
            x = L + (R - L) * (k / 10)
            c.create_line(x, T, x, B, fill="#e6e6e6")
        for k in range(6):
            y = T + (B - T) * (k / 5)
            c.create_line(L, y, R, y, fill="#e6e6e6")
        c.create_rectangle(L, T, R, B, outline="#222222")
        c.create_text((L + R) // 2, self.h - 10, text="Index", fill="#111111", font=("TkDefaultFont", 10))
        c.create_text(L - 40, T - 4, text="RSSI (dBm)", fill="#111111", font=("TkDefaultFont", 10), anchor="w")
        if self.ymin is not None and self.ymax is not None:
            c.create_text(L - 6, B, text=f"{self.ymin:.1f}", fill="#111111", anchor="e", font=("TkDefaultFont", 9), tags="ylmin")
            c.create_text(L - 6, T, text=f"{self.ymax:.1f}", fill="#111111", anchor="e", font=("TkDefaultFont", 9), tags="ylmax")
        else:
            c.create_text(L - 6, B, text="min", fill="#111111", anchor="e", font=("TkDefaultFont", 9), tags="ylmin")
            c.create_text(L - 6, T, text="max", fill="#111111", anchor="e", font=("TkDefaultFont", 9), tags="ylmax")
        c.create_text((L + R) // 2, T - 6, text="idx vs RSSI (live)", fill="#111111", font=("TkDefaultFont", 10))

    def _update_y_labels(self):
        L = self.margin_l
        T, B = self.margin_t, self.h - self.margin_b
        self.c.itemconfigure("ylmin", text=f"{self.ymin:.1f}")
        self.c.coords("ylmin", L - 6, B)
        self.c.itemconfigure("ylmax", text=f"{self.ymax:.1f}")
        self.c.coords("ylmax", L - 6, T)

    def _add_or_update(self, idx, rssi):
        if self.ymin is None or self.ymax is None:
            self.ymin = rssi - self.y_pad
            self.ymax = rssi + self.y_pad
            self._update_y_labels()

        need_rescale = False
        if rssi < self.ymin - 1.5:
            self.ymin = rssi - self.y_pad
            need_rescale = True
        if rssi > self.ymax + 1.5:
            self.ymax = rssi + self.y_pad
            need_rescale = True

        self.points[idx] = rssi

        if need_rescale:
            self._update_y_labels()
            for i, v in enumerate(self.points):
                if v is None: continue
                x, y, r = self._x(i), self._y(v), self.r_pt
                if self.items_p[i] is None:
                    self.items_p[i] = self.c.create_oval(x-r, y-r, x+r, y+r,
                                                         outline=self.col_pts, fill=self.col_pts, width=1.0)
                else:
                    self.c.coords(self.items_p[i], x-r, y-r, x+r, y+r)
            for i in range(self.n_total - 1):
                v1 = self.points[i]
                v2 = self.points[i+1]
                if v1 is None or v2 is None: 
                    continue
                x1, y1 = self._x(i),   self._y(v1)
                x2, y2 = self._x(i+1), self._y(v2)
                if self.items_l[i] is None:
                    self.items_l[i] = self.c.create_line(x1, y1, x2, y2, fill=self.col_line, width=1.6)
                else:
                    self.c.coords(self.items_l[i], x1, y1, x2, y2)
        else:
            x, y, r = self._x(idx), self._y(rssi), self.r_pt
            if self.items_p[idx] is None:
                self.items_p[idx] = self.c.create_oval(x-r, y-r, x+r, y+r,
                                                       outline=self.col_pts, fill=self.col_pts, width=1.0)
            else:
                self.c.coords(self.items_p[idx], x-r, y-r, x+r, y+r)
            if idx > 0 and self.points[idx-1] is not None:
                self._ensure_line(idx-1)
            if idx + 1 < self.n_total and self.points[idx+1] is not None:
                self._ensure_line(idx)

    def _ensure_line(self, i):
        if not (0 <= i < self.n_total - 1): return
        v1, v2 = self.points[i], self.points[i+1]
        if v1 is None or v2 is None: return
        x1, y1 = self._x(i),   self._y(v1)
        x2, y2 = self._x(i+1), self._y(v2)
        if self.items_l[i] is None:
            self.items_l[i] = self.c.create_line(x1, y1, x2, y2, fill=self.col_line, width=1.6)
        else:
            self.c.coords(self.items_l[i], x1, y1, x2, y2)

# --------------------- radio helpers (Bob) ---------------------
class BobRadio:
    def __init__(self, ui, port, freq_mhz, air_speed, power_dbm, buffer_size, my_addr, peer_addr):
        self.ui = ui
        self.node = sx126x.sx126x(
            port, freq_mhz, addr=my_addr, power=power_dbm,
            rssi=True, air_speed=air_speed, buffer_size=buffer_size
        )
        self.my_addr = my_addr
        self.peer_addr = peer_addr

    def _hdr(self, dst):
        off = getattr(self.node, "offset_freq", 0)
        return bytes([(dst>>8)&0xFF, dst&0xFF, off,
                      (self.my_addr>>8)&0xFF, self.my_addr&0xFF, off])

    def send(self, ptype, payload, dst=None):
        if dst is None: dst = self.peer_addr
        self.node.send(self._hdr(dst) + bytes([ptype]) + payload)

    def recv(self, expect_types, timeout=5.0):
        t0 = time.time()
        types = {expect_types} if isinstance(expect_types, int) else set(expect_types)
        while time.time() - t0 < timeout:
            if self.node.receive(timeout=0.5):
                p = self.node.last_payload or b""
                if len(p) >= 1 and p[0] in types:
                    return p[0], p[1:]
            time.sleep(0.02)
        return None, None

# --------------------- GUI app (Bob) ---------------------------
class BobApp(tk.Tk):
    TYPE_PROBE=0xA1; TYPE_REPLY=0xB2; TYPE_KEEP=0xC0; TYPE_THR=0xC3
    TYPE_READY_ECC=0xC9; TYPE_SYN=0xD0; TYPE_SYN_CHUNK=0xD1; TYPE_SYN_END=0xD2
    TYPE_DROP=0xD5; TYPE_DONE=0xEE

    def __init__(self):
        super().__init__()
        self.title("Bob — Dasam LoRa Keygen (GUI)")
        self.geometry("920x820"); self.resizable(True, True)
        self.stop_flag = threading.Event(); self.log_q = queue.Queue()
        self.stage_frames = {}; self.stage1_plot = None
        self._build_ui(); self._poll_logs()

    def _build_ui(self):
        cfg = ttk.LabelFrame(self, text="Radio Config"); cfg.pack(fill="x", padx=10, pady=6)
        self.port     = self._add_ent(cfg, "Port", PORT_DEFAULT, 0, 0, 16)
        self.freq     = self._add_ent(cfg, "Freq MHz", str(FREQ_MHZ_DEFAULT), 0, 2, 8)
        self.airs     = self._add_ent(cfg, "Air bps", str( AIR_SPEED_DEFAULT), 0, 4, 8)
        self.power    = self._add_ent(cfg, "Power dBm", str(POWER_DBM_DEFAULT), 0, 6, 6)
        self.bufsize  = self._add_ent(cfg, "Buffer sz", str(BUFFER_SIZE_DEFAULT), 1, 0, 8)
        self.addr_me  = self._add_ent(cfg, "Addr Bob",   hex(ADDR_BOB_DEFAULT),  1, 2, 10)
        self.addr_peer= self._add_ent(cfg, "Addr Alice", hex(ADDR_ALICE_DEFAULT),1, 4, 10)
        self.probes   = self._add_ent(cfg, "Probes", str(PROBE_COUNT_DEFAULT), 1, 6, 6)
        self.probe_gap= self._add_ent(cfg, "Probe gap s", f"{PROBE_GAP_DEFAULT:.2f}", 2, 0, 8)

        btns = ttk.Frame(self); btns.pack(fill="x", padx=10, pady=2)
        self.btn_start = ttk.Button(btns, text="Start Full Run", command=self._start)
        self.btn_stop  = ttk.Button(btns, text="Abort", command=self._abort, state="disabled")
        self.btn_start.pack(side="left", padx=4); self.btn_stop.pack(side="left", padx=4)
        ttk.Button(btns, text="Save Log…", command=self._save_log).pack(side="right", padx=4)

        f1 = self._stage("Stage 1 — Probing")
        self.stage1_plot = PlotCanvas(f1, width=880, height=260); self.stage1_plot.pack(fill="x", padx=6, pady=(0,6))
        self._stage("Stage 2 — Indices & Thresholds")
        self._stage("Stage 3 — ECC Receive & IR")
        self._stage("Stage 4 — Final Key")

        logf = ttk.LabelFrame(self, text="Log"); logf.pack(fill="both", expand=True, padx=10, pady=6)
        self.log_box = ScrolledText(logf, height=12, wrap="word"); self.log_box.pack(fill="both", expand=True)

    def _add_ent(self, parent, label, default, r, c, width):
        ttk.Label(parent, text=label).grid(row=r, column=c, sticky="w", padx=4, pady=2)
        var = tk.StringVar(value=default); ent = ttk.Entry(parent, textvariable=var, width=width)
        ent.grid(row=r, column=c+1, sticky="w", padx=4, pady=2); return var

    def _stage(self, title):
        f = ttk.LabelFrame(self, text=title); f.pack(fill="x", padx=10, pady=4)
        bar = ttk.Progressbar(f, mode="determinate", length=560); bar.pack(side="left", padx=8, pady=6)
        lbl = ttk.Label(f, text="idle"); lbl.pack(side="left", padx=8)
        self.stage_frames[title] = (bar, lbl, f); return f

    # logs
    def log(self, msg): self.log_q.put(msg)
    def _poll_logs(self):
        try:
            while True:
                msg = self.log_q.get_nowait()
                self.log_box.insert("end", msg.rstrip()+"\n"); self.log_box.see("end")
        except queue.Empty: pass
        self.after(80, self._poll_logs)

    def _set_stage(self, title, value, total=None, extra=""):
        bar, lbl, _ = self.stage_frames[title]
        if total is None:
            bar.config(mode="indeterminate"); bar.start(40); lbl.config(text=extra or "working…")
        else:
            bar.config(mode="determinate", maximum=total, value=value); lbl.config(text=f"{value}/{total} {extra}".strip())
    def _done_stage(self, title, note="done"):
        bar, lbl, _ = self.stage_frames[title]
        bar.stop(); bar.config(mode="determinate", value=bar["maximum"] if bar["maximum"] else 1); lbl.config(text=note)

    # run / abort
    def _start(self):
        self.stop_flag.clear(); self.btn_start.config(state="disabled"); self.btn_stop.config(state="normal")
        threading.Thread(target=self._run_all, daemon=True).start()
    def _abort(self): self.stop_flag.set(); self.log("⚠️  Abort requested")
    def _save_log(self):
        fn = filedialog.asksaveasfilename(defaultextension=".txt", initialfile="bob_gui_log.txt")
        if not fn: return
        with open(fn, "w") as f: f.write(self.log_box.get("1.0", "end"))
        messagebox.showinfo("Saved", f"Log written to {fn}")

    # pipeline
    def _run_all(self):
        try:
            node = BobRadio(self, self.port.get(), int(self.freq.get()), int(self.airs.get()),
                            int(self.power.get()), int(self.bufsize.get()),
                            int(self.addr_me.get(), 0), int(self.addr_peer.get(), 0))
            self.log(f"✅ Radio up (buffer={getattr(node.node,'buffer_size','?')})")
            bits_corr = self._run_pipeline(node)
            if bits_corr is None: return
            key = hashlib.sha256(self._bits_to_bytes(bits_corr)).hexdigest()
            self._done_stage("Stage 4 — Final Key", f"bits={len(bits_corr)}")
            self.log(f"✅ SHA-256 = {key}")
            node.send(self.TYPE_DONE, b"")
        except Exception as e:
            self.log(f"❌ ERROR: {e}")
        finally:
            self.btn_start.config(state="normal"); self.btn_stop.config(state="disabled")

    def _run_pipeline(self, node: 'BobRadio'):
        # Stage 1 — Probing
        N = int(self.probes.get()); gap = float(self.probe_gap.get())
        rssi = [math.nan] * N; ok = 0
        self._set_stage("Stage 1 — Probing", 0, N, ""); self.stage1_plot.reset(N)
        t_end = time.time() + (N * (gap + 1.6) + 6.0); count = 0
        while count < N and time.time() < t_end and not self.stop_flag.is_set():
            if node.node.receive(timeout=0.4):
                p = node.node.last_payload or b""
                if p and len(p) >= 3 and p[0] == self.TYPE_PROBE:
                    idx = struct.unpack("<H", p[1:3])[0]
                    if 0 <= idx < len(rssi):
                        rssi[idx] = node.node.last_rssi
                        self.stage1_plot.post_point(idx, rssi[idx])   # << fast, throttled
                        ok += 1
                        node.send(self.TYPE_REPLY, struct.pack("<H", idx) + b"B")
                        self.log(f"📥 Probe #{idx}  RSSI -{node.node.last_rssi} dBm; ✅ reply")
                        count += 1
                        self._set_stage("Stage 1 — Probing", count, N, f"ok={ok}")
                        time.sleep(gap)
            else:
                time.sleep(0.02)
        self._done_stage("Stage 1 — Probing", f"ok={ok}/{N}")

        # Stage 2 — Indices & Thresholds
        self._set_stage("Stage 2 — Indices & Thresholds", 0, 1, "waiting indices")
        kept, total, got = None, None, 0; deadline = time.time() + 60.0
        while time.time() < deadline and not self.stop_flag.is_set():
            t, pl = node.recv(self.TYPE_KEEP, timeout=5.0)
            if t is None: continue
            if len(pl) < 6: self.log("⚠️  Short TYPE_KEEP header"); continue
            total_count, offset_count, chunk_count = struct.unpack("<HHH", pl[:6])
            need = 2*chunk_count
            if len(pl) < 6 + need: self.log("⚠️  TYPE_KEEP payload short"); continue
            if total is None: total = total_count; kept = [None]*total
            idxs = list(struct.unpack("<" + "H"*chunk_count, pl[6:6+need]))
            for i, v in enumerate(idxs):
                pos = offset_count + i
                if 0 <= pos < total:
                    if kept[pos] is None: got += 1
                    kept[pos] = v
            self._set_stage("Stage 2 — Indices & Thresholds", got, total, "indices")
            if got == total: break
            deadline = time.time() + 20.0
        if kept is None:
            self.log("❌ No indices received"); return None
        t, pl = node.recv(self.TYPE_THR, timeout=20.0)
        if t is None or len(pl) < 8:
            self.log("❌ Thresholds not received"); return None
        lo, hi = struct.unpack("<ff", pl[:8])
        self._done_stage("Stage 2 — Indices & Thresholds", f"kept={len(kept)}  lo/hi={lo:.2f}/{hi:.2f}")

        # Quantise with thresholds
        bits = []; mid = 0.5 * (lo + hi)
        for i in kept:
            v = rssi[i]
            if v >= hi: bits.append(1)
            elif v <= lo: bits.append(0)
            else: bits.append(1 if v >= mid else 0)

        # Stage 3 — ECC
        self._set_stage("Stage 3 — ECC Receive & IR", 0, 1, "send READY")
        node.send(self.TYPE_READY_ECC, b""); self.log("✅ READY_ECC sent")

        total_bits, n, syn_list, crc_list = self._recv_ecc(node)
        if total_bits == 0:
            self.log("❌ No usable ECC"); return None

        blocksB = bch.chunk_bits(bits[:total_bits], n)
        nblocks = min(len(blocksB), len(syn_list), len(crc_list))
        self._set_stage("Stage 3 — ECC Receive & IR", 0, nblocks, "IR")

        kept_blocks = []; dropped_idx = []; flips_total = 0
        for i in range(nblocks):
            blk = blocksB[i]
            syn_err = syn_list[i] ^ bch.syndrome(blk)
            corr_blk, flips = bch.correct_with_syndrome(blk, syn_err)
            flips_total += flips
            ok = (bch.crc8(corr_blk) == crc_list[i])
            if ok: kept_blocks.append(corr_blk)
            else:  dropped_idx.append(i)
            self._set_stage("Stage 3 — ECC Receive & IR", i+1, nblocks, f"drops={len(dropped_idx)}")

        # Report drops (+ sentinel empty)
        MAX_DROP_CHUNK = 10
        sent = 0; total_drops = len(dropped_idx)
        while sent < total_drops and not self.stop_flag.is_set():
            chunk = dropped_idx[sent:sent+MAX_DROP_CHUNK]
            payload = struct.pack("<H", len(chunk))
            if chunk:
                payload += struct.pack("<" + "H"*len(chunk), *chunk)
            node.send(self.TYPE_DROP, payload); sent += len(chunk); time.sleep(0.05)
        node.send(self.TYPE_DROP, struct.pack("<H", 0))
        self._done_stage("Stage 3 — ECC Receive & IR", f"kept={len(kept_blocks)}/{nblocks}, flips={flips_total}")

        # Stage 4 — Final Key
        bits_corr = [b for blk in kept_blocks for b in blk]
        self._set_stage("Stage 4 — Final Key", len(bits_corr), len(bits_corr) or 1, "final bits")
        return bits_corr

    # ECC receive helper
    def _recv_ecc(self, node: 'BobRadio'):
        self.log("ℹ️  Receiving chunked ECC…")
        t, pl = node.recv((self.TYPE_SYN, self.TYPE_SYN_CHUNK, self.TYPE_SYN_END), timeout=12.0)

        # Monolithic
        if t == self.TYPE_SYN and pl and len(pl) >= 5:
            total_bits, n, nblocks = struct.unpack("<HBH", pl[:5])
            syn_bytes_len = (nblocks + 1) // 2
            syn_bytes = pl[5:5 + syn_bytes_len]
            crc_bytes = pl[5 + syn_bytes_len: 5 + syn_bytes_len + nblocks]
            self.log(f"✅ ECC monolithic: blocks={nblocks}")
            return total_bits, n, bch.unpack_syndromes(syn_bytes, nblocks), list(crc_bytes)

        # Chunked
        buf = {}; declared_parts = None; declared_total = None
        last_activity = time.time(); parts_seen = 0
        end_seen = False; end_deadline = None

        def done():
            if declared_parts is not None and len(buf) >= declared_parts: return True
            if declared_total is not None:
                total_len = sum(len(v) for v in buf.values())
                if total_len >= declared_total: return True
            if end_seen and end_deadline is not None and time.time() >= end_deadline: return True
            return False

        if t == self.TYPE_SYN_CHUNK and pl and len(pl) >= 2:
            seq = struct.unpack("<H", pl[:2])[0]
            buf[seq] = pl[2:]; parts_seen = 1; last_activity = time.time()
            self._set_stage("Stage 3 — ECC Receive & IR", parts_seen, parts_seen, "chunks")
        elif t == self.TYPE_SYN_END and pl and len(pl) >= 4:
            declared_parts, declared_total = struct.unpack("<HH", pl[:4])
            last_activity = time.time(); end_seen = True; end_deadline = time.time() + END_GRACE_SEC
            self.log(f"ℹ️  ECC END announced: parts={declared_parts}, total={declared_total}")

        while (time.time() - last_activity) < IDLE_LIMIT_ECC and not self.stop_flag.is_set():
            if done(): break
            t, pl = node.recv((self.TYPE_SYN_CHUNK, self.TYPE_SYN_END), timeout=5.0)
            if t is None: continue
            if t == self.TYPE_SYN_CHUNK:
                if len(pl) < 2: self.log("⚠️  short SYN_CHUNK; skipping"); continue
                seq = struct.unpack("<H", pl[:2])[0]
                if seq not in buf: parts_seen += 1
                buf[seq] = pl[2:]; last_activity = time.time()
                self._set_stage("Stage 3 — ECC Receive & IR", len(buf), declared_parts or len(buf), "chunks")
            elif t == self.TYPE_SYN_END and len(pl) >= 4:
                declared_parts, declared_total = struct.unpack("<HH", pl[:4])
                last_activity = time.time(); end_seen = True; end_deadline = time.time() + END_GRACE_SEC
                self.log(f"ℹ️  ECC END received: parts={declared_parts}, total={declared_total}")

        if not buf:
            self.log("❌ No ECC chunks received"); return 0, 0, [], []

        payload = b"".join(buf[i] for i in sorted(buf))
        if declared_parts is not None and len(buf) != declared_parts:
            self.log(f"⚠️  Missing chunks: got {len(buf)} / {declared_parts}")
        if declared_total is not None and len(payload) != declared_total:
            self.log(f"⚠️  Size mismatch: got {len(payload)} vs declared {declared_total}")

        if len(payload) < 5:
            self.log("❌ ECC buffer too short after chunking"); return 0, 0, [], []

        total_bits, n, nblocks = struct.unpack("<HBH", payload[:5])
        syn_bytes_len = (nblocks + 1) // 2
        syn_bytes = payload[5:5 + syn_bytes_len]
        crc_bytes  = payload[5 + syn_bytes_len: 5 + syn_bytes_len + nblocks]
        self.log(f"✅ ECC chunked assembled: blocks={nblocks}")
        return total_bits, n, bch.unpack_syndromes(syn_bytes, nblocks), list(crc_bytes)

    @staticmethod
    def _bits_to_bytes(bits):
        if not bits: return b""
        pad = (-len(bits)) % 8
        if pad: bits = bits + [0]*pad
        v = 0
        for b in bits: v = (v<<1) | (b & 1)
        return v.to_bytes(len(bits)//8, "big")

# --------------------- main ---------------------
if __name__ == "__main__":
    BobApp().mainloop()
