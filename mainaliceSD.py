#!/usr/bin/env python3
import sx126x, time, csv, math, struct, hashlib, os, sys, statistics
import bch  # chunk_bits, syndrome, crc8, pack_syndromes, unpack_syndromes

# ======== PRETTY LOGS ========
class Log:
    def __init__(self, enable=True):
        self.enable = enable
        self.use_color = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
        self._last_progress_len = 0
        self._last_progress_label = None
    def _c(self, code): return f"\033[{code}m" if self.use_color else ""
    def _r(self): return self._c("0")
    def color(self, s, code): return f"{self._c(code)}{s}{self._r()}"
    def bold(self, s): return self.color(s, "1")
    def cyan(self, s): return self.color(s, "36")
    def green(self, s): return self.color(s, "32")
    def yellow(self, s): return self.color(s, "33")
    def red(self, s): return self.color(s, "31")
    def grey(self, s): return self.color(s, "90")
    def blue(self, s): return self.color(s, "34")
    def banner(self, title, subtitle=None):
        line = "─" * max(10, len(title) + 2)
        print(self.cyan(f"\n┏{line}┓"))
        print(self.cyan(f"┃ ") + self.bold(title) + self.cyan(f" ┃"))
        if subtitle:
            print(self.cyan("┃ ") + subtitle + self.cyan(f" ┃"))
        print(self.cyan(f"┗{line}┛"))
    def info(self, msg):  print("  " + msg)
    def ok(self, msg):    print("  ✅ " + self.green(msg))
    def warn(self, msg):  print("  ⚠️  " + self.yellow(msg))
    def err(self, msg):   print("  ❌ " + self.red(msg))
    def kv(self, k, v): print(f"    {self.grey(k+':'):20} {v}")
    def progress(self, label, i, n, extra=""):
        width = 28
        i = min(i, n)
        frac = 0 if n == 0 else i / n
        filled = int(frac * width)
        bar = "█" * filled + "░" * (width - filled)
        pct = int(frac * 100)
        line = f"  {label:18} [{bar}] {pct:3d}%  {i}/{n}"
        if extra:
            line += f"  {self.grey(extra)}"
        print("\r" + " " * max(self._last_progress_len, len(line)), end="")
        print("\r" + line, end="", flush=True)
        self._last_progress_len = len(line)
        self._last_progress_label = label
        if i >= n: print("")

log = Log(enable=True)

# ======== METRICS HELPERS ========
import pathlib
def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d: pathlib.Path(d).mkdir(parents=True, exist_ok=True)

def save_bits_ascii01(path: str, bits):
    _ensure_dir(path)
    with open(path, "w") as f:
        f.write("".join("1" if b else "0" for b in bits))

def save_bits_bin(path: str, bits):
    _ensure_dir(path)
    out = bytearray(); acc = 0; k = 0
    for b in bits:
        acc = (acc << 1) | (1 if b else 0)
        k += 1
        if k == 8: out.append(acc); acc = 0; k = 0
    if k: out.append(acc << (8-k))
    with open(path, "wb") as f: f.write(out)

def kgr_bits(len_bits: int, probes: int, duration_s: float):
    if probes <= 1 or duration_s <= 0:
        return float("nan"), float("nan")
    return (len_bits / (probes - 1)), (len_bits / duration_s)

# ======== DEBUG / BEHAVIOR TOGGLES ========
DEBUG = True
PROBE_COUNT = 500
PROBE_GAP   = 0.40
ECC_CHUNKED = True
CHUNK_SIZE_IDX = 8
ECC_CHUNK_BYTES = 20         # keep chunks comfortably within cap
ECC_PRE_DELAY_SEC = 1.00
ECC_INTER_CHUNK_GAP = 0.08
VERBOSE_PROBES = True

# ======== RADIO / PROTOCOL CONFIG ========
PORT         = "/dev/ttyS0"
FREQ_MHZ     = 868
ADDR_ALICE   = 0x0000
ADDR_BOB     = 0x0001
AIR_SPEED    = 1200
POWER_DBM    = 13
BUFFER_SIZE  = 240

# On-air conservative payload cap (total bytes, including type/seq headers)
WIRE_PAYLOAD_CAP = 32

GUARD_K      = 0.60
HAM_N        = 31            # BCH(31,21)

# Packet types
TYPE_PROBE      = 0xA1
TYPE_REPLY      = 0xB2
TYPE_KEEP       = 0xC0
TYPE_THR        = 0xC3
TYPE_READY_ECC  = 0xC9
TYPE_SYN        = 0xD0
TYPE_SYN_CHUNK  = 0xD1
TYPE_SYN_END    = 0xD2
TYPE_DROP       = 0xD5
TYPE_DONE       = 0xEE

# ======== LoRa driver ========
node = sx126x.sx126x(PORT, FREQ_MHZ, addr=ADDR_ALICE, power=POWER_DBM,
                     rssi=True, air_speed=AIR_SPEED, buffer_size=BUFFER_SIZE)

def _hdr(dst):
    off = getattr(node, "offset_freq", 0)
    return bytes([(dst>>8)&0xFF, dst&0xFF, off,
                  (ADDR_ALICE>>8)&0xFF, ADDR_ALICE&0xFF, off])

def _send(ptype: int, payload: bytes, dst=ADDR_BOB):
    node.send(_hdr(dst) + bytes([ptype]) + payload)
    if DEBUG and not VERBOSE_PROBES:
        if ptype in (TYPE_SYN, TYPE_SYN_CHUNK, TYPE_SYN_END):
            log.info(f"SENT type=0x{ptype:02X} (ECC) len={len(payload)}")
        elif ptype in (TYPE_KEEP, TYPE_THR, TYPE_READY_ECC, TYPE_DROP, TYPE_DONE):
            log.info(f"SENT type=0x{ptype:02X} len={len(payload)}")

def _recv(expect_types, timeout=5.0):
    t0 = time.time()
    types = {expect_types} if isinstance(expect_types, int) else set(expect_types)
    while time.time() - t0 < timeout:
        if node.receive(timeout=0.5):
            p = node.last_payload or b""
            if len(p) >= 1 and p[0] in types:
                if DEBUG and not VERBOSE_PROBES:
                    log.info(f"RECV type=0x{p[0]:02X} len={len(p)-1}")
                return p[0], p[1:]
        time.sleep(0.01)
    return None, None

# ======== CSV ========
def save_csv_rssi(filename, rssi):
    path = os.path.join(os.path.dirname(__file__), filename)
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["idx","rssi"])
        for i,rv in enumerate(rssi): w.writerow([i, rv])
    log.ok(f"Saved {filename} (N={len(rssi)})")

# ======== Probing ========
def send_probe(idx: int):
    payload = struct.pack("<H", idx) + b"A"
    _send(TYPE_PROBE, payload)
    if VERBOSE_PROBES: log.info(f"📨 Probe #{idx}")
    time.sleep(PROBE_GAP)

def recv_reply(expect_idx: int, rssi_arr, timeout=1.5):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if node.receive(timeout=0.4):
            p = node.last_payload or b""
            if p and len(p) >= 3 and p[0] == TYPE_REPLY \
               and struct.unpack("<H", p[1:3])[0] == expect_idx:
                rssi_arr[expect_idx] = node.last_rssi
                if VERBOSE_PROBES:
                    log.ok(f"Reply #{expect_idx}  RSSI -{node.last_rssi} dBm")
                return True
        time.sleep(0.02)
    if VERBOSE_PROBES: log.warn(f"Reply timeout for #{expect_idx}")
    return False

def run_probing():
    log.banner("Alice — Dasam LoRa Keygen",
               f"Freq={FREQ_MHZ}MHz  Air={AIR_SPEED}bps  Buffer={BUFFER_SIZE}  Power={POWER_DBM}dBm")
    log.banner("Stage 1 — Probing")
    log.kv("Probes", PROBE_COUNT)
    log.kv("Gap", f"{PROBE_GAP:.2f}s")
    ok = 0
    rssi = [math.nan] * PROBE_COUNT
    t0 = time.perf_counter()
    for i in range(PROBE_COUNT):
        send_probe(i)
        if recv_reply(i, rssi): ok += 1
        if not VERBOSE_PROBES:
            if (i % 10 == 9) or (i+1 == PROBE_COUNT):
                rate = (i+1)/(time.perf_counter()-t0+1e-6)
                log.progress("Probing", i+1, PROBE_COUNT, extra=f"ok={ok}  {rate:.1f}/s")
    dt = time.perf_counter() - t0
    log.ok(f"Probing finished in {dt:.1f}s  (ok={ok}/{PROBE_COUNT})")
    save_csv_rssi("alice_rssi.csv", rssi)
    try:
        global _ALICE_PROBE_DURATION
    except NameError:
        pass
    _ALICE_PROBE_DURATION = dt
    return rssi

# ======== Quantization ========
def quantise_guard(rssi, k=GUARD_K):
    vals = [v for v in rssi if not math.isnan(v)]
    mu = statistics.mean(vals) if vals else 0.0
    sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    lo, hi = mu - k*sd, mu + k*sd
    kept_idx, bits = [], []
    for i, v in enumerate(rssi):
        if math.isnan(v): 
            continue
        if v >= hi:
            kept_idx.append(i); bits.append(1)
        elif v <= lo:
            kept_idx.append(i); bits.append(0)
    log.banner("Stage 2 — Quantisation")
    log.kv("Mean RSSI", f"{mu:.2f}")
    log.kv("Std Dev", f"{sd:.2f}")
    log.kv("lo/hi", f"{lo:.2f} / {hi:.2f}")
    log.kv("Kept indices", f"{len(kept_idx)} / {len(rssi)}")
    return bits, kept_idx, (lo, hi)

# ======== Index & thresholds ========
def send_kept_indices(kept_idx):
    total = len(kept_idx)
    sent = 0
    chunks = 0
    t0 = time.perf_counter()
    while sent < total:
        chunk = kept_idx[sent:sent+CHUNK_SIZE_IDX]
        payload = struct.pack("<HHH", total, sent, len(chunk))
        payload += struct.pack("<" + "H"*len(chunk), *chunk)
        _send(TYPE_KEEP, payload)
        sent += len(chunk); chunks += 1
        time.sleep(0.03)
        log.progress("Send kept idx", sent, total)
    dt = time.perf_counter() - t0
    log.ok(f"Kept indices sent in {dt:.2f}s ({chunks} chunks)")

def send_thresholds(lo, hi):
    _send(TYPE_THR, struct.pack("<ff", float(lo), float(hi)))
    log.ok(f"Thresholds sent  lo={lo:.2f}  hi={hi:.2f}")

def wait_ready_ecc(timeout=10.0):
    log.banner("Stage 3 — ECC")
    log.info("Waiting for READY_ECC from Bob…")
    t0 = time.perf_counter()
    t, _ = _recv(TYPE_READY_ECC, timeout)
    if t is None: log.warn("Timed out waiting for READY_ECC")
    else:         log.ok(f"Got READY_ECC after {time.perf_counter()-t0:.2f}s")
    if ECC_PRE_DELAY_SEC > 0:
        log.info(f"Pause {ECC_PRE_DELAY_SEC:.2f}s before sending ECC…")
        time.sleep(ECC_PRE_DELAY_SEC)

# ======== ECC send (cap-aware; auto-chunk if needed) ========
def send_ecc(bits):
    blocks = bch.chunk_bits(bits, HAM_N)
    syn_bytes = bch.pack_syndromes([bch.syndrome(b) for b in blocks])
    crc_bytes = bytes(bch.crc8(b) for b in blocks)

    header = struct.pack("<HBH", len(bits), HAM_N, len(blocks))
    payload = header + syn_bytes + crc_bytes
    log.kv("ECC blocks", len(blocks))
    log.kv("Syndrome bytes", len(syn_bytes))
    log.kv("CRC bytes", len(crc_bytes))

    # ---- CAP-AWARE SENDING ----
    # Overheads: TYPE_SYN has +1 type; TYPE_SYN_CHUNK has +1 type +2 seq; radio adds its own hdr
    mono_cap  = WIRE_PAYLOAD_CAP - 1          # bytes of payload allowed with TYPE_SYN
    chunk_cap = WIRE_PAYLOAD_CAP - 1 - 2      # payload allowed with TYPE_SYN_CHUNK
    buf_cap   = getattr(node, "buffer_size", 64)
    wire_cap  = min(64, buf_cap)              # conservative wire cap on this API call

    if not ECC_CHUNKED and len(payload) <= mono_cap and len(payload) <= wire_cap:
        _send(TYPE_SYN, payload)
        log.ok(f"ECC sent monolithic  total={len(payload)} bytes")
        return blocks

    # Force chunked if larger than monolithic cap or if ECC_CHUNKED=True
    part_max = max(1, min(ECC_CHUNK_BYTES, chunk_cap, wire_cap - 3))  # -3 for type+seq
    log.kv("Chunk bytes", part_max)

    seq = 0
    total_parts = (len(payload) + part_max - 1) // part_max
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

# ======== Dropped-blocks (chunked receive) ========
def recv_dropped_blocks(timeout=20.0):
    drops = []
    t_end = time.time() + timeout
    got_any = False
    while time.time() < t_end:
        t, pl = _recv(TYPE_DROP, timeout=3.0)
        if t is None: break
        got_any = True
        if len(pl) < 2:
            log.warn("Short TYPE_DROP packet")
            continue
        count = struct.unpack("<H", pl[:2])[0]
        if len(pl) < 2 + 2*count:
            log.warn(f"Incomplete TYPE_DROP: count={count}, got {len(pl)-2} bytes")
            count = (len(pl)-2)//2
        if count > 0:
            idxs = struct.unpack("<" + "H"*count, pl[2:2+2*count])
            drops.extend(idxs)
        log.progress("Recv drops", len(drops), len(drops))
    if not got_any: log.warn("No dropped-block report received")
    else:           log.ok(f"Dropped blocks reported: {len(drops)}")
    return drops

# ======== Bits helper ========
def bits_to_bytes(bits):
    if not bits: return b""
    pad = (-len(bits)) % 8
    if pad: bits = bits + [0]*pad
    v = 0
    for b in bits: v = (v<<1) | (b & 1)
    return v.to_bytes(len(bits)//8, "big")

# ======== Main ========
if __name__ == "__main__":
    log.banner("Alice — Dasam LoRa Keygen",
               f"Freq={FREQ_MHZ}MHz  Air={AIR_SPEED}bps  Buffer={BUFFER_SIZE}  Power={POWER_DBM}dBm")

    # 1) Probing
    rssi = run_probing()

    # 2) Quantize & share
    bitsA, kept_idx, (lo, hi) = quantise_guard(rssi)
    send_kept_indices(kept_idx)
    send_thresholds(lo, hi)

    # --- Metrics/NIST: pre-ECC dump + KGR ---
    try:
        duration = _ALICE_PROBE_DURATION
    except Exception:
        duration = 0.0
    save_bits_ascii01("logs/alice_pre_bits.txt", bitsA)
    save_bits_bin("logs/alice_pre_bits.bin", bitsA)
    bpp, bps = kgr_bits(len(bitsA), PROBE_COUNT, duration)
    log.kv("Pre-ECC bits", len(bitsA))
    log.ok(f"[KGR] bits/probe={bpp:.4f}  bits/s={bps:.2f}  duration={duration:.2f}s")

    # 3) ECC
    wait_ready_ecc()
    blocksA = send_ecc(bitsA)
    drops = set(recv_dropped_blocks())

    # 4) Final key
    kept_blocksA = [blk for i, blk in enumerate(blocksA) if i not in drops]
    bits_final = [b for blk in kept_blocksA for b in blk]
    key = hashlib.sha256(bits_to_bytes(bits_final)).hexdigest()

    log.banner("Stage 4 — Key")
    log.kv("Final kept blocks", f"{len(kept_blocksA)} / {len(blocksA)}")
    log.kv("Final bits", len(bits_final))
    log.ok(f"SHA-256 = {log.bold(key)}")

    # --- NIST (optional): final post-IR bitstring (pre-hash) ---
    save_bits_ascii01("logs/alice_final_bits.txt", bits_final)
    save_bits_bin("logs/alice_final_bits.bin", bits_final)
    log.ok("Wrote NIST inputs: logs/alice_pre_bits.txt and logs/alice_final_bits.txt")

    _send(TYPE_DONE, b"")
    log.ok("DONE")
