#!/usr/bin/env python3
# bob_differential.py — Differential quantization (ε, window, lag) + ECC receiver
import sx126x, time, csv, math, struct, hashlib, os, sys
import bch
from typing import List
import pathlib
import numpy as np

# ========= Parameters =========
DEBUG = True
VERBOSE_PROBES = True

PORT         = "/dev/ttyS0"
FREQ_MHZ     = 868
ADDR_ALICE   = 0x0000
ADDR_BOB     = 0x0001
AIR_SPEED    = 9600
POWER_DBM    = 22
BUFFER_SIZE  = 240

PROBE_COUNT  = 100
PROBE_GAP    = 0.50

IDLE_LIMIT_ECC = 40.0
HAM_N          = 15

# Packet types
TYPE_PROBE      = 0xA1
TYPE_PROBE_END  = 0xA2   # NEW: Alice → Bob, indicates end of probing
TYPE_REPLY      = 0xB2
TYPE_KEEP       = 0xC0
TYPE_THR        = 0xC3   # (epsilon, win, lag)
TYPE_READY_ECC  = 0xC9
TYPE_SYN        = 0xD0
TYPE_SYN_CHUNK  = 0xD1
TYPE_SYN_END    = 0xD2
TYPE_DROP       = 0xD5
TYPE_DONE       = 0xEE
TYPE_BAD        = 0xC6   # NEW: Bob→Alice: indices Bob cannot compute

# ========= Pretty logger =========
class Log:
    def __init__(self, enable=True):
        self.enable = enable
        self.use_color = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
        self._last_progress_len = 0
    def _c(self, code): return f"\033[{code}m" if self.use_color else ""
    def _r(self): return self._c("0")
    def color(self, s, code): return f"{self._c(code)}{s}{self._r()}"
    def bold(self, s): return self.color(s, "1")
    def cyan(self, s): return self.color(s, "36")
    def green(self, s): return self.color(s, "32")
    def yellow(self, s): return self.color(s, "33")
    def red(self, s): return self.color(s, "31")
    def grey(self, s): return self.color(s, "90")
    def banner(self, t, sub=None):
        line = "─" * max(10, len(t) + 2)
        print(self.cyan(f"\n┏{line}┓")); print(self.cyan("┃ ") + self.bold(t) + self.cyan(" ┃"))
        if sub: print(self.cyan("┃ ") + sub + self.cyan(" ┃"))
        print(self.cyan(f"┗{line}┛"))
    def info(self, msg): print("  " + msg)
    def ok(self, msg):   print("  ✅ " + self.green(msg))
    def warn(self, msg): print("  ⚠️  " + self.yellow(msg))
    def err(self, msg):  print("  ❌ " + self.red(msg))
    def kv(self, k, v):  print(f"    {self.grey(k+':'):20} {v}")
    def progress(self, label, i, n, extra=""):
        width = 28; i=min(i,n)
        frac = 0 if n==0 else i/n
        filled = int(frac*width)
        bar = "█"*filled + "░"*(width-filled)
        pct = int(frac*100)
        line = f"  {label:18} [{bar}] {pct:3d}%  {i}/{n}"
        if extra: line += f"  {self.grey(extra)}"
        print("\r" + " " * max(self._last_progress_len, len(line)), end="")
        print("\r" + line, end="", flush=True); self._last_progress_len = len(line)
        if i>=n: print("")

log = Log(True)

# ========= Utils =========
def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d: pathlib.Path(d).mkdir(parents=True, exist_ok=True)

def save_csv_rssi(filename, rssi):
    path = os.path.join(os.path.dirname(__file__), filename)
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["idx","rssi"])
        for i,rv in enumerate(rssi): w.writerow([i, rv])
    log.ok(f"Saved {filename} (N={len(rssi)})")

def save_bits_ascii01(path: str, bits):
    _ensure_dir(path)
    with open(path, "w") as f:
        f.write("".join("1" if b else "0" for b in bits))

def save_bits_bin(path: str, bits):
    _ensure_dir(path)
    out = bytearray(); acc=0; k=0
    for b in bits:
        acc=(acc<<1)|(1 if b else 0); k+=1
        if k==8: out.append(acc); acc=0; k=0
    if k: out.append(acc<<(8-k))
    with open(path, "wb") as f: f.write(out)

def kgr_bits(len_bits: int, probes: int, duration_s: float):
    if (probes or 0)<=1 or (duration_s or 0)<=0: return float("nan"), float("nan")
    return (len_bits/(probes-1)), (len_bits/duration_s)

def movavg_nan(x: List[float], w: int):
    if w is None or w <= 1: return np.array(x, dtype=float)
    x = np.array(x, dtype=float)
    m = ~np.isnan(x)
    k = np.ones(w, dtype=float)
    num = np.convolve(np.where(m, x, 0.0), k, mode="same")
    den = np.convolve(m.astype(float), k, mode="same")
    out = np.divide(num, den, out=np.full_like(x, np.nan), where=den>0)
    return out

# ========= Radio =========
node = sx126x.sx126x(PORT, FREQ_MHZ, addr=ADDR_BOB, power=POWER_DBM,
                     rssi=True, air_speed=AIR_SPEED, buffer_size=BUFFER_SIZE)
try:
    log.info(f"LoRa buffer size = {node.buffer_size}")
except Exception:
    log.warn("LoRa buffer size attribute not exposed; continuing.")

def _hdr(dst):
    off = getattr(node, "offset_freq", 0)
    return bytes([(dst>>8)&0xFF, dst&0xFF, off, (ADDR_BOB>>8)&0xFF, ADDR_BOB&0xFF, off])

def _send(ptype: int, payload: bytes, dst=ADDR_ALICE):
    node.send(_hdr(dst) + bytes([ptype]) + payload)
    if DEBUG and not VERBOSE_PROBES:
        if ptype in (TYPE_SYN, TYPE_SYN_CHUNK, TYPE_SYN_END):
            log.info(f"SENT type=0x{ptype:02X} (ECC) len={len(payload)}")
        elif ptype in (TYPE_KEEP, TYPE_THR, TYPE_READY_ECC, TYPE_DROP, TYPE_DONE, TYPE_REPLY, TYPE_BAD):
            log.info(f"SENT type=0x{ptype:02X} len={len(payload)}")

def _recv(expect_types, timeout=5.0):
    t0 = time.time(); types = {expect_types} if isinstance(expect_types, int) else set(expect_types)
    while time.time() - t0 < timeout:
        if node.receive(timeout=0.5):
            p = getattr(node, "last_payload", b"") or b""
            if len(p) >= 1 and p[0] in types:
                if DEBUG and not VERBOSE_PROBES:
                    log.info(f"RECV type=0x{p[0]:02X} len={len(p)-1}")
                return p[0], p[1:]
        time.sleep(0.01)
    return None, None

# ========= Probing (Bob replies) =========
def recv_probe(rssi_arr, timeout=2.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if node.receive(timeout=0.4):
            p = getattr(node, "last_payload", b"") or b""
            # End-of-probing signal from Alice
            if p and len(p) >= 1 and p[0] == TYPE_PROBE_END:
                if VERBOSE_PROBES: log.ok("Got PROBE_END from Alice")
                return -1
            # Regular probe
            if p and len(p) >= 3 and p[0] == TYPE_PROBE:
                idx = struct.unpack("<H", p[1:3])[0]
                if 0 <= idx < len(rssi_arr):
                    rssi_arr[idx] = node.last_rssi
                if VERBOSE_PROBES:
                    log.info(f"📥 Probe #{idx}  RSSI -{node.last_rssi} dBm")
                return idx
        time.sleep(0.02)
    return None

def send_reply(idx):
    payload = struct.pack("<H", idx) + b"B"
    _send(TYPE_REPLY, payload)
    if VERBOSE_PROBES: log.ok(f"Reply #{idx}")
    time.sleep(PROBE_GAP)

def run_probing():
    log.banner("Bob — Differential Keygen",
               f"Freq={FREQ_MHZ}MHz  Air={AIR_SPEED}bps  Buffer={BUFFER_SIZE}  Power={POWER_DBM}dBm")
    log.banner("Stage 1 — Probing"); log.kv("Probes (target)", PROBE_COUNT)

    rssi = [math.nan] * PROBE_COUNT
    seen = set()
    last_idx = -1
    last_rx_time = None
    t0 = time.perf_counter()

    # Heuristics: if PROBE_END is lost, stop after last index + quiet window
    wait_per_iter    = max(0.3, PROBE_GAP * 1.25)
    quiet_after_last = max(0.5, PROBE_GAP * 1.10)
    soft_deadline    = PROBE_COUNT * PROBE_GAP + 10.0

    while True:
        idx = recv_probe(rssi, timeout=wait_per_iter)
        now = time.perf_counter()

        if idx == -1:  # PROBE_END
            break

        if idx is not None and 0 <= idx < PROBE_COUNT:
            if idx not in seen:
                seen.add(idx)
            last_idx = max(last_idx, idx)
            last_rx_time = now

        # progress (no misleading "Missed probe #count")
        if not VERBOSE_PROBES:
            ok = len(seen)
            rate = (ok / max(1e-6, (now - t0)))
            extra = f"ok={ok}  last_idx={last_idx}  {rate:.1f}/s"
            log.progress("Probing", ok, PROBE_COUNT, extra=extra)

        # Fallback stop conditions if PROBE_END not received
        if last_idx >= PROBE_COUNT - 1 and last_rx_time and (now - last_rx_time) >= quiet_after_last:
            break
        if (now - t0) >= soft_deadline:
            log.warn("Probing soft-deadline reached; continuing with what we have.")
            break

    dt = time.perf_counter() - t0
    ok_count = len(seen)
    log.ok(f"Probing finished in {dt:.1f}s  (ok={ok_count}/{PROBE_COUNT})")
    save_csv_rssi("bob_rssi.csv", rssi)
    global _BOB_PROBE_DURATION; _BOB_PROBE_DURATION = dt
    return rssi

# ========= Stage 2: receive indices & params =========
def recv_kept_indices():
    log.banner("Stage 2 — Indices & Params")
    kept=None; total=None; got=0; deadline=time.time()+60.0
    while time.time() < deadline:
        t,pl=_recv(TYPE_KEEP, timeout=5.0)
        if t is None: continue
        if len(pl) < 6:
            log.warn("Short kept-index header; skip"); continue
        total_count, offset_count, chunk_count = struct.unpack("<HHH", pl[:6])
        need = 2*chunk_count
        if len(pl) < 6+need:
            log.warn(f"Expected {need} index bytes, got {len(pl)-6}; skip"); continue
        if kept is None:
            total = total_count; kept=[None]*total
        if total_count != total:
            log.warn(f"Mismatched total ({total_count} != {total}); skip"); continue
        idx_list = list(struct.unpack("<" + "H"*chunk_count, pl[6:6+need]))
        for i,v in enumerate(idx_list):
            pos = offset_count + i
            if 0 <= pos < total:
                if kept[pos] is None: got += 1
                kept[pos] = v
        log.progress("Recv kept idx", got, total)
        if got == total:
            log.ok(f"Kept indices received ({total})"); return kept
        deadline = time.time() + 20.0
    raise TimeoutError("Kept indices reception timed out")

def recv_diff_params():
    t,pl=_recv(TYPE_THR, timeout=20.0)
    if t is None or len(pl) < 8:
        raise TimeoutError("Params not received/short")
    eps, win, lag = struct.unpack("<fhh", pl[:8])
    log.ok(f"Params received  eps={eps:.2f} dB  win={win}  lag={lag}")
    return float(eps), int(win), int(lag)

# === NEW: compute & send BAD indices (missing/timeouts) ===
def compute_bad_indices_bob(rssi_raw, kept_idx, win, lag):
    b = movavg_nan(rssi_raw, int(win))
    n = len(b); bad=[]
    for i in kept_idx:
        j = i + lag
        if not (1 <= j < n):
            bad.append(i); continue
        x0, x1 = b[j-1], b[j]
        if math.isnan(x0) or math.isnan(x1):
            # try raw fallback; if still NaN, mark bad
            try:
                x0r, x1r = float(rssi_raw[j-1]), float(rssi_raw[j])
                if math.isnan(x0r) or math.isnan(x1r):
                    bad.append(i); continue
            except Exception:
                bad.append(i); continue
    return bad

def send_bad_indices(bad_idx):
    total=len(bad_idx); sent=0; CHUNK=24
    while sent < total:
        chunk = bad_idx[sent:sent+CHUNK]
        payload = struct.pack("<H", len(chunk))
        if chunk: payload += struct.pack("<" + "H"*len(chunk), *chunk)
        _send(TYPE_BAD, payload); sent += len(chunk)
        if not VERBOSE_PROBES:
            log.progress("Send BAD idx", sent, total)

# ========= ECC receive =========
def recv_ecc():
    log.banner("Stage 3 — ECC")
    t,pl=_recv((TYPE_SYN, TYPE_SYN_CHUNK, TYPE_SYN_END), timeout=12.0)

    if t == TYPE_SYN and pl and len(pl) >= 5:
        total_bits, n, nblocks = struct.unpack("<HBH", pl[:5])
        syn_bytes_len = (nblocks + 1)//2
        syn_bytes = pl[5:5+syn_bytes_len]
        crc_bytes = pl[5+syn_bytes_len:5+syn_bytes_len+nblocks]
        log.kv("ECC blocks", nblocks); log.ok("ECC received (monolithic)")
        return total_bits, n, bch.unpack_syndromes(syn_bytes, nblocks), list(crc_bytes)

    log.info("Receiving chunked ECC…")
    buf={}; declared_parts=None; declared_total=None; last_activity=time.time(); parts_seen=0
    if t == TYPE_SYN_CHUNK and pl and len(pl) >= 2:
        seq=struct.unpack("<H", pl[:2])[0]; buf[seq]=pl[2:]; parts_seen=1
        last_activity=time.time(); log.progress("ECC chunks", parts_seen, parts_seen)
    elif t == TYPE_SYN_END and pl and len(pl) >= 4:
        declared_parts, declared_total = struct.unpack("<HH", pl[:4])
        last_activity=time.time(); log.info(f"ECC END announced: parts={declared_parts}, total={declared_total}")

    while time.time() - last_activity < IDLE_LIMIT_ECC:
        t,pl=_recv((TYPE_SYN_CHUNK, TYPE_SYN_END), timeout=5.0)
        if t is None: continue
        if t == TYPE_SYN_CHUNK:
            if len(pl) < 2: log.warn("short SYN_CHUNK; skipping"); continue
            seq=struct.unpack("<H", pl[:2])[0]
            if seq not in buf: parts_seen += 1
            buf[seq]=pl[2:]; last_activity=time.time()
            log.progress("ECC chunks", len(buf), declared_parts or len(buf))
        elif t == TYPE_SYN_END and len(pl) >= 4:
            declared_parts, declared_total = struct.unpack("<HH", pl[:4])
            last_activity=time.time(); log.info(f"ECC END received: parts={declared_parts}, total={declared_total}")
            break

    if not buf:
        log.err("No ECC chunks received"); return 0,0,[],[]

    payload = b"".join(buf[i] for i in sorted(buf))
    if len(payload) < 5:
        log.err("ECC buffer too short after chunking"); return 0,0,[],[]

    total_bits, n, nblocks = struct.unpack("<HBH", payload[:5])
    syn_bytes_len = (nblocks + 1)//2
    syn_bytes = payload[5:5+syn_bytes_len]
    crc_bytes = payload[5+syn_bytes_len:5+syn_bytes_len+nblocks]
    log.kv("ECC blocks", nblocks); log.ok("ECC chunked payload assembled")
    return total_bits, n, bch.unpack_syndromes(syn_bytes, nblocks), list(crc_bytes)

# ========= Differential bits (Bob) =========
def diff_bits_bob(rssi_raw, kept_idx, eps, win, lag):
    b = movavg_nan(rssi_raw, int(win))
    bits=[]
    n = len(b)
    for i in kept_idx:
        j = i + lag
        if not (1 <= j < n):
            bits.append(0); continue
        x0, x1 = b[j-1], b[j]
        if math.isnan(x0) or math.isnan(x1):
            # Fallback: try raw; if missing, force 0 (it will be dropped by ECC if misaligned)
            try:
                x0r, x1r = float(rssi_raw[j-1]), float(rssi_raw[j])
                d = (0.0 if (math.isnan(x0r) or math.isnan(x1r)) else (x1r - x0r))
            except Exception:
                d = 0.0
        else:
            d = x1 - x0
        bits.append(1 if d > 0 else 0)
    return bits

def bits_to_bytes(bits):
    if not bits: return b""
    pad = (-len(bits)) % 8
    if pad: bits = bits + [0]*pad
    v=0
    for b in bits: v=(v<<1)|(b&1)
    return v.to_bytes(len(bits)//8, "big")

# ========= Main =========
if __name__ == "__main__":
    # 1) Probing
    rssi = run_probing()

    # 2) Receive indices + diff params from Alice
    kept_idx = recv_kept_indices()
    eps, win, lag = recv_diff_params()

    # 2b) Compute and send BAD indices; signal READY_ECC afterwards
    bad_idx = compute_bad_indices_bob(rssi, kept_idx, win, lag)
    if bad_idx:
        log.ok(f"Reporting {len(bad_idx)} bad indices (missing/timeouts) to Alice")
        send_bad_indices(bad_idx)
    else:
        log.info("No bad indices to report")
    _send(TYPE_READY_ECC, b""); log.ok("Sent READY_ECC to Alice")

    # 3) Receive ECC and run IR
    total_bits, n, syn_list, crc_list = recv_ecc()
    if total_bits == 0:
        log.err("No usable ECC received; aborting IR")
        _send(TYPE_DROP, struct.pack("<H", 0))
        _send(TYPE_DONE, b"")
        raise SystemExit(0)

    # Build Bob's bits aligned to COMMON kept indices (Alice pruned on her side)
    badset = set(bad_idx)
    kept_common = [i for i in kept_idx if i not in badset]
    bitsB = diff_bits_bob(rssi, kept_common, eps, win, lag)
    bitsB = bitsB[:total_bits]
    blocksB = bch.chunk_bits(bitsB, n)

    # --- Metrics/NIST: pre-ECC dump + KGR (Bob side) ---
    duration = globals().get("_BOB_PROBE_DURATION", 0.0)
    _ensure_dir("logs")
    save_bits_ascii01("logs/bob_pre_bits.txt", bitsB[:total_bits])
    save_bits_bin("logs/bob_pre_bits.bin", bitsB[:total_bits])
    bpp, bps = kgr_bits(len(bitsB[:total_bits]), PROBE_COUNT, duration)
    log.kv("Pre-ECC bits (Bob)", len(bitsB[:total_bits]))
    log.ok(f"[KGR] (Bob) bits/probe={bpp:.4f}  bits/s={bps:.2f}  duration={duration:.2f}s")

    # IR per block
    kept_blocks=[]; dropped_idx=[]; flips_total=0
    block_count = min(len(blocksB), len(crc_list), len(syn_list))
    for i in range(block_count):
        blk = blocksB[i]
        syn_err = syn_list[i] ^ bch.syndrome(blk)
        corr_blk, flips = bch.correct_with_syndrome(blk, syn_err)
        flips_total += flips
        ok = (bch.crc8(corr_blk) == crc_list[i])
        if ok: kept_blocks.append(corr_blk)
        else:  dropped_idx.append(i)
        if DEBUG and VERBOSE_PROBES and i < 6:
            log.info(f"IR blk#{i}: syn_err={syn_err} flips={flips} crc_ok={ok}")

    # Report dropped blocks
    MAX_DROP_CHUNK=10; total_drops=len(dropped_idx); sent=0
    while sent < total_drops:
        chunk = dropped_idx[sent:sent+MAX_DROP_CHUNK]
        payload = struct.pack("<H", len(chunk))
        if chunk: payload += struct.pack("<" + "H"*len(chunk), *chunk)
        _send(TYPE_DROP, payload); sent += len(chunk); time.sleep(0.05)
        if not VERBOSE_PROBES: log.progress("Send drops", sent, total_drops)

    log.ok(f"IR summary: flips_total={flips_total}, dropped_blocks={len(dropped_idx)}")

    # 5) Key
    bits_corr = [b for blk in kept_blocks for b in blk]
    key = hashlib.sha256(bits_to_bytes(bits_corr)).hexdigest()
    log.banner("Stage 4 — Key")
    log.kv("Final kept blocks", f"{len(kept_blocks)} / {block_count}")
    log.kv("Final bits", len(bits_corr))
    log.ok(f"SHA-256 = {log.bold(key)}")

    _send(TYPE_DONE, b""); log.ok("DONE")
