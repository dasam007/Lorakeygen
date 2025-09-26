#!/usr/bin/env python3
# alice_differential.py — Differential quantization (ε, window, lag) + ECC sender
import sx126x, time, csv, math, struct, hashlib, os, sys
import statistics
import bch  # needs: chunk_bits, syndrome, correct_with_syndrome, crc8, pack_syndromes, unpack_syndromes
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

PROBE_COUNT  = 100     # set to 100 for your last run; change as needed
PROBE_GAP    = 0.50    # increase to 0.75–1.00s at AIR_SPEED=1200 for long range

# --- Differential preset ---
DIFF_EPS_DB      = 2.0   # epsilon threshold (dB)
DIFF_WIN         = 4     # moving-average window (samples); 1 disables smoothing
DIFF_LAG         = -1    # Bob index = i + lag (negative means Bob earlier)
DIFF_GUARD_MULT  = 1.0   # guard = ±(GUARD_MULT * eps). Use 3.0 for ±6 dB guard, 1.0 to disable

# ECC
HAM_N        = 15     # block length for bch.chunk_bits
ECC_CHUNKED  = True
ECC_CHUNK_BYTES = 20
ECC_PRE_DELAY_SEC = 1.00
ECC_INTER_CHUNK_GAP = 0.08

# Packet types
TYPE_PROBE      = 0xA1
TYPE_PROBE_END  = 0xA2   # NEW: Alice → Bob, indicates end of probing
TYPE_REPLY      = 0xB2
TYPE_KEEP       = 0xC0
TYPE_THR        = 0xC3   # carries (epsilon, win, lag)
TYPE_READY_ECC  = 0xC9
TYPE_SYN        = 0xD0
TYPE_SYN_CHUNK  = 0xD1
TYPE_SYN_END    = 0xD2
TYPE_DROP       = 0xD5
TYPE_DONE       = 0xEE
TYPE_BAD        = 0xC6   # NEW: Bob → Alice: kept indices Bob cannot compute (missing/timeouts)

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
        width = 28; i = min(i, n)
        frac = 0 if n == 0 else i/n
        filled = int(frac*width)
        bar = "█"*filled + "░"*(width-filled)
        pct = int(frac*100)
        line = f"  {label:18} [{bar}] {pct:3d}%  {i}/{n}"
        if extra: line += f"  {self.grey(extra)}"
        print("\r" + " " * max(self._last_progress_len, len(line)), end="")
        print("\r" + line, end="", flush=True); self._last_progress_len = len(line)
        if i >= n: print("")

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
        acc = (acc<<1) | (1 if b else 0); k += 1
        if k == 8: out.append(acc); acc=0; k=0
    if k: out.append(acc << (8-k))
    with open(path, "wb") as f: f.write(out)

def kgr_bits(len_bits: int, probes: int, duration_s: float):
    if (probes or 0) <= 1 or (duration_s or 0) <= 0: return float("nan"), float("nan")
    return (len_bits / (probes - 1)), (len_bits / duration_s)

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
node = sx126x.sx126x(PORT, FREQ_MHZ, addr=ADDR_ALICE, power=POWER_DBM,
                     rssi=True, air_speed=AIR_SPEED, buffer_size=BUFFER_SIZE)

def _hdr(dst):
    off = getattr(node, "offset_freq", 0)
    return bytes([(dst>>8)&0xFF, dst&0xFF, off, (ADDR_ALICE>>8)&0xFF, ADDR_ALICE&0xFF, off])

def _send(ptype: int, payload: bytes, dst=ADDR_BOB):
    node.send(_hdr(dst) + bytes([ptype]) + payload)
    if DEBUG and not VERBOSE_PROBES:
        if ptype in (TYPE_SYN, TYPE_SYN_CHUNK, TYPE_SYN_END):
            log.info(f"SENT type=0x{ptype:02X} (ECC) len={len(payload)}")
        elif ptype in (TYPE_KEEP, TYPE_THR, TYPE_READY_ECC, TYPE_DROP, TYPE_DONE, TYPE_BAD, TYPE_PROBE_END):
            log.info(f"SENT type=0x{ptype:02X} len={len(payload)}")

def _recv(expect_types, timeout=5.0):
    t0 = time.time(); types = {expect_types} if isinstance(expect_types, int) else set(expect_types)
    while time.time() - t0 < timeout:
        if node.receive(timeout=0.5):
            p = node.last_payload or b""
            if len(p) >= 1 and p[0] in types:
                if DEBUG and not VERBOSE_PROBES:
                    log.info(f"RECV type=0x{p[0]:02X} len={len(p)-1}")
                return p[0], p[1:]
        time.sleep(0.01)
    return None, None

# ========= Probing =========
def send_probe(idx: int):
    node.send(_hdr(ADDR_BOB) + bytes([TYPE_PROBE]) + struct.pack("<H", idx) + b"A")
    if VERBOSE_PROBES: log.info(f"📨 Probe #{idx}")
    time.sleep(PROBE_GAP)

def send_probe_end():
    # signal "no more probes" to Bob so he can stop immediately
    node.send(_hdr(ADDR_BOB) + bytes([TYPE_PROBE_END]))
    if VERBOSE_PROBES: log.ok("Sent PROBE_END")

def recv_reply(expect_idx: int, rssi_arr, timeout=1.8):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if node.receive(timeout=0.5):
            p = node.last_payload or b""
            if p and len(p) >= 3 and p[0] == TYPE_REPLY and struct.unpack("<H", p[1:3])[0] == expect_idx:
                rssi_arr[expect_idx] = node.last_rssi
                if VERBOSE_PROBES: log.ok(f"Reply #{expect_idx}  RSSI -{node.last_rssi} dBm")
                return True
        time.sleep(0.02)
    if VERBOSE_PROBES: log.warn(f"Reply timeout for #{expect_idx}")
    return False

def run_probing():
    log.banner("Alice — Differential Keygen",
               f"Freq={FREQ_MHZ}MHz  Air={AIR_SPEED}bps  Buffer={BUFFER_SIZE}  Power={POWER_DBM}dBm")
    log.banner("Stage 1 — Probing")
    log.kv("Probes", PROBE_COUNT); log.kv("Gap", f"{PROBE_GAP:.2f}s")
    ok=0; rssi=[math.nan]*PROBE_COUNT; t0=time.perf_counter()
    for i in range(PROBE_COUNT):
        send_probe(i)
        if recv_reply(i, rssi): ok += 1
        if not VERBOSE_PROBES and ((i%10==9) or (i+1==PROBE_COUNT)):
            rate = (i+1)/(time.perf_counter()-t0+1e-6)
            log.progress("Probing", i+1, PROBE_COUNT, extra=f"ok={ok}  {rate:.1f}/s")
    send_probe_end()
    dt = time.perf_counter()-t0
    log.ok(f"Probing finished in {dt:.1f}s  (ok={ok}/{PROBE_COUNT})")
    save_csv_rssi("alice_rssi.csv", rssi)
    global _ALICE_PROBE_DURATION; _ALICE_PROBE_DURATION = dt
    return rssi

# ========= Differential Quantization (Alice) =========
def diff_quantise_alice(rssi_raw, eps=DIFF_EPS_DB, win=DIFF_WIN, lag=DIFF_LAG):
    log.banner("Stage 2 — Differential Quantisation")
    a = movavg_nan(rssi_raw, int(win))
    kept_idx=[]; bits=[]
    n = len(a)
    thr = float(eps) * float(DIFF_GUARD_MULT)  # guard band (±dB)

    # Only keep i where both a[i-1], a[i] are finite AND i+lag in [1, n-1]
    for i in range(1, n):
        j = i + lag
        if j < 1 or j >= n:
            continue
        x0, x1 = a[i-1], a[i]
        if math.isnan(x0) or math.isnan(x1):
            continue
        d = x1 - x0
        if abs(d) >= thr:
            kept_idx.append(i)
            bits.append(1 if d > 0 else 0)

    log.kv("ε (dB)", eps); log.kv("window", win); log.kv("lag", lag)
    log.kv("Guard (±dB)", f"{thr:.2f}")
    log.kv("Kept indices", f"{len(kept_idx)} / {n}")
    return bits, kept_idx

def send_kept_indices(kept_idx):
    total=len(kept_idx); sent=0; chunks=0; t0=time.perf_counter(); CHUNK=24
    while sent < total:
        chunk = kept_idx[sent:sent+CHUNK]
        payload = struct.pack("<HHH", total, sent, len(chunk))
        if chunk: payload += struct.pack("<" + "H"*len(chunk), *chunk)
        _send(TYPE_KEEP, payload)
        sent += len(chunk); chunks += 1; time.sleep(0.02)
        if not VERBOSE_PROBES:
            log.progress("Send kept idx", sent, total)
    dt=time.perf_counter()-t0; log.ok(f"Kept indices sent in {dt:.2f}s ({chunks} chunks)")

def send_diff_params(eps, win, lag):
    payload = struct.pack("<fhh", float(eps), int(win), int(lag))
    _send(TYPE_THR, payload)
    log.ok(f"Sent params  eps={eps:.2f} dB  win={win}  lag={lag}")

# === NEW: receive Bob's BAD indices (missing/timeouts) and READY_ECC ===
def recv_bad_and_ready(timeout=30.0):
    bad=[]; ready=False
    deadline = time.time() + timeout
    while time.time() < deadline:
        t,pl = _recv((TYPE_BAD, TYPE_READY_ECC), timeout=5.0)
        if t is None:
            continue
        if t == TYPE_BAD:
            if len(pl) >= 2:
                cnt = struct.unpack("<H", pl[:2])[0]
                need = 2*cnt
                if len(pl) >= 2+need and cnt>0:
                    bad.extend(struct.unpack("<" + "H"*cnt, pl[2:2+need]))
                else:
                    pairs = max(0, (len(pl)-2)//2)
                    if pairs:
                        bad.extend(struct.unpack("<" + "H"*pairs, pl[2:2+2*pairs]))
            deadline = time.time() + 5.0  # extend a bit after each BAD chunk
        elif t == TYPE_READY_ECC:
            ready = True
            break
    bad = sorted(set(bad))
    if bad:
        log.ok(f"Received {len(bad)} bad indices from Bob (will prune)")
    else:
        log.info("No bad indices received from Bob")
    return bad, ready

def wait_ready_ecc(timeout=10.0):
    log.banner("Stage 3 — ECC"); log.info("Waiting for READY_ECC from Bob…")
    t0=time.perf_counter(); t,_=_recv(TYPE_READY_ECC, timeout)
    if t is None: log.warn("Timed out waiting for READY_ECC")
    else: log.ok(f"Got READY_ECC after {time.perf_counter()-t0:.2f}s")
    if ECC_PRE_DELAY_SEC>0:
        log.info(f"Pause {ECC_PRE_DELAY_SEC:.2f}s before sending ECC…")
        time.sleep(ECC_PRE_DELAY_SEC)

# ========= ECC send =========
def send_ecc(bits):
    blocks = bch.chunk_bits(bits, HAM_N)
    syn_bytes = bch.pack_syndromes([bch.syndrome(b) for b in blocks])
    crc_bytes = bytes(bch.crc8(b) for b in blocks)
    header = struct.pack("<HBH", len(bits), HAM_N, len(blocks))
    payload = header + syn_bytes + crc_bytes
    log.kv("ECC blocks", len(blocks))
    log.kv("Syndrome bytes", len(syn_bytes))
    log.kv("CRC bytes", len(crc_bytes))
    if not ECC_CHUNKED:
        _send(TYPE_SYN, payload); log.ok(f"ECC sent monolithic  total={len(payload)} bytes"); return blocks
    buf_cap  = getattr(node, "buffer_size", 64)
    wire_cap = min(64, buf_cap)
    part_max = max(1, min(ECC_CHUNK_BYTES, wire_cap - 9))
    log.kv("Chunk bytes", part_max)
    seq=0; total_parts = (len(payload)+part_max-1)//part_max
    for off in range(0, len(payload), part_max):
        part = payload[off:off+part_max]
        _send(TYPE_SYN_CHUNK, struct.pack("<H", seq) + part)
        seq += 1
        if seq % 5 == 0 or seq == total_parts:
            log.progress("Send ECC", seq, total_parts)
        time.sleep(ECC_INTER_CHUNK_GAP)
    _send(TYPE_SYN_END, struct.pack("<HH", seq, len(payload)))
    log.ok(f"ECC chunked: parts={seq}, total={len(payload)} bytes")
    return blocks

def recv_dropped_blocks(timeout=20.0):
    drops=[]; t_end=time.time()+timeout; got_any=False
    while time.time() < t_end:
        t,pl=_recv(TYPE_DROP, timeout=3.0)
        if t is None: break
        got_any=True
        if len(pl) < 2:
            log.warn("Short TYPE_DROP"); continue
        count=struct.unpack("<H", pl[:2])[0]
        have=len(pl)-2
        if have < 2*count: count = have//2
        if count>0:
            idxs=struct.unpack("<" + "H"*count, pl[2:2+2*count])
            drops.extend(idxs)
        log.progress("Recv drops", len(drops), len(drops))
    if not got_any: log.warn("No dropped-block report received")
    else: log.ok(f"Dropped blocks reported: {len(drops)}")
    return set(drops)

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

    # 2) Differential quantise & share
    bitsA, kept_idx = diff_quantise_alice(rssi, DIFF_EPS_DB, DIFF_WIN, DIFF_LAG)
    send_kept_indices(kept_idx)
    send_diff_params(DIFF_EPS_DB, DIFF_WIN, DIFF_LAG)

    # 2c) Receive Bob's BAD indices and READY_ECC, then prune to COMMON
    bad_idx, got_ready = recv_bad_and_ready(timeout=30.0)
    if bad_idx:
        badset = set(bad_idx)
        pruned = [(i,b) for (i,b) in zip(kept_idx, bitsA) if i not in badset]
        kept_idx = [p[0] for p in pruned]
        bitsA    = [p[1] for p in pruned]
        log.ok(f"Pruned to common indices: {len(bitsA)} bits remain after BAD filter")
    if not got_ready:
        wait_ready_ecc()

    # Metrics + NIST dumps (pre-ECC)
    duration = globals().get("_ALICE_PROBE_DURATION", 0.0)
    _ensure_dir("logs")
    save_bits_ascii01("logs/alice_pre_bits.txt", bitsA)
    save_bits_bin("logs/alice_pre_bits.bin", bitsA)
    bpp, bps = kgr_bits(len(bitsA), PROBE_COUNT, duration)
    log.kv("Pre-ECC bits", len(bitsA))
    log.ok(f"[KGR] bits/probe={bpp:.4f}  bits/s={bps:.2f}  duration={duration:.2f}s")

    # 3) ECC
    blocksA = send_ecc(bitsA)
    drops = recv_dropped_blocks()

    # 4) Final key
    kept_blocksA = [blk for i, blk in enumerate(blocksA) if i not in drops]
    bits_final = [b for blk in kept_blocksA for b in blk]
    key = hashlib.sha256(bits_to_bytes(bits_final)).hexdigest()

    log.banner("Stage 4 — Key")
    log.kv("Final kept blocks", f"{len(kept_blocksA)} / {len(blocksA)}")
    log.kv("Final bits", len(bits_final))
    log.ok(f"SHA-256 = {log.bold(key)}")
    save_bits_ascii01("logs/alice_final_bits.txt", bits_final)
    save_bits_bin("logs/alice_final_bits.bin", bits_final)

    _send(TYPE_DONE, b""); log.ok("DONE")
