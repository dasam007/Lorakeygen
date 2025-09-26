#!/usr/bin/env python3
import sx126x, time, csv, math, struct, hashlib, os, sys
import bch

# ======== PRETTY LOGS ========
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
    def banner(self, title, subtitle=None):
        line = "─" * max(10, len(title) + 2)
        print(self.cyan(f"\n┏{line}┓")); print(self.cyan(f"┃ ") + self.bold(title) + self.cyan(f" ┃"))
        if subtitle: print(self.cyan("┃ ") + subtitle + self.cyan(f" ┃"))
        print(self.cyan(f"┗{line}┛"))
    def info(self, msg):  print("  " + msg)
    def ok(self, msg):    print("  ✅ " + self.green(msg))
    def warn(self, msg):  print("  ⚠️  " + self.yellow(msg))
    def err(self, msg):   print("  ❌ " + self.red(msg))
    def kv(self, k, v):   print(f"    {self.grey(k+':'):20} {v}")
    def progress(self, label, i, n, extra=""):
        width = 28; i = min(i, n); frac = 0 if n == 0 else i / n
        filled = int(frac * width); bar = "█" * filled + "░" * (width - filled)
        pct = int(frac * 100)
        line = f"  {label:18} [{bar}] {pct:3d}%  {i}/{n}"
        if extra: line += f"  {self.grey(extra)}"
        print("\r" + " " * max(self._last_progress_len, len(line)), end="")
        print("\r" + line, end="", flush=True); self._last_progress_len = len(line)
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
        acc = (acc << 1) | (1 if b else 0); k += 1
        if k == 8: out.append(acc); acc = 0; k = 0
    if k: out.append(acc << (8-k))
    with open(path, "wb") as f: f.write(out)

def kgr_bits(len_bits: int, probes: int, duration_s: float):
    if probes <= 0 or duration_s <= 0: return float("nan"), float("nan")
    return (len_bits / max(1, probes-1)), (len_bits / duration_s)

# ======== DEBUG / BEHAVIOR TOGGLES ========
DEBUG = True
PROBE_COUNT = 500
PROBE_GAP   = 0.40
IDLE_LIMIT_ECC = 40.0
END_GRACE_SEC  = 0.6   # short wait after SYN_END to catch stragglers
VERBOSE_PROBES = True

# ======== RADIO / PROTOCOL CONFIG ========
PORT         = "/dev/ttyS0"
FREQ_MHZ     = 868
ADDR_ALICE   = 0x0000
ADDR_BOB     = 0x0001
AIR_SPEED    = 1200
POWER_DBM    = 13
BUFFER_SIZE  = 240
HAM_N        = 31   # BCH(31,21)

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
node = sx126x.sx126x(PORT, FREQ_MHZ, addr=ADDR_BOB, power=POWER_DBM,
                     rssi=True, air_speed=AIR_SPEED, buffer_size=BUFFER_SIZE)
try:
    log.info(f"LoRa buffer size = {node.buffer_size}")
except Exception:
    log.warn("LoRa buffer size attribute not exposed; continuing.")

def _hdr(dst):
    off = getattr(node, "offset_freq", 0)
    return bytes([(dst>>8)&0xFF, dst&0xFF, off,
                  (ADDR_BOB>>8)&0xFF, ADDR_BOB&0xFF, off])

def _send(ptype: int, payload: bytes, dst=ADDR_ALICE):
    node.send(_hdr(dst) + bytes([ptype]) + payload)
    if DEBUG and not VERBOSE_PROBES:
        if ptype in (TYPE_SYN, TYPE_SYN_CHUNK, TYPE_SYN_END):
            log.info(f"SENT type=0x{ptype:02X} (ECC) len={len(payload)}")
        elif ptype in (TYPE_KEEP, TYPE_THR, TYPE_READY_ECC, TYPE_DROP, TYPE_DONE, TYPE_REPLY):
            log.info(f"SENT type=0x{ptype:02X} len={len(payload)}")

def _recv(expect_types, timeout=5.0):
    t0 = time.time()
    types = {expect_types} if isinstance(expect_types, int) else set(expect_types)
    while time.time() - t0 < timeout:
        if node.receive(timeout=0.5):
            p = getattr(node, "last_payload", b"") or b""
            if len(p) >= 1 and p[0] in types:
                if DEBUG and not VERBOSE_PROBES:
                    log.info(f"RECV type=0x{p[0]:02X} len={len(p)-1}")
                return p[0], p[1:]
        time.sleep(0.01)
    return None, None

# ======== CSV saver ========
def save_csv_rssi(filename, rssi):
    path = os.path.join(os.path.dirname(__file__), filename)
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["idx","rssi"])
        for i,rv in enumerate(rssi): w.writerow([i, rv])
    log.ok(f"Saved {filename} (N={len(rssi)})")

# ======== Probing (ping-pong) ========
def recv_probe(rssi_arr, timeout=2.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if node.receive(timeout=0.4):
            p = getattr(node, "last_payload", b"") or b""
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
    log.banner("Bob — Dasam LoRa Keygen",
               f"Freq={FREQ_MHZ}MHz  Air={AIR_SPEED}bps  Buffer={BUFFER_SIZE}  Power={POWER_DBM}dBm")
    log.banner("Stage 1 — Probing"); log.kv("Probes", PROBE_COUNT)
    ok = 0; rssi = [math.nan] * PROBE_COUNT; t0 = time.perf_counter()
    for count in range(PROBE_COUNT):
        idx = recv_probe(rssi)
        if idx is None:
            if VERBOSE_PROBES: log.warn(f"Missed probe #{count}")
            continue
        send_reply(idx); ok += 1
        if not VERBOSE_PROBES and ((count % 10 == 9) or (count+1 == PROBE_COUNT)):
            rate = (count+1)/(time.perf_counter()-t0+1e-6)
            log.progress("Probing", count+1, PROBE_COUNT, extra=f"ok={ok}  {rate:.1f}/s")
    dt = time.perf_counter() - t0
    log.ok(f"Probing finished in {dt:.1f}s  (ok={ok}/{PROBE_COUNT})")
    save_csv_rssi("bob_rssi.csv", rssi)
    try: global _BOB_PROBE_DURATION
    except NameError: pass
    _BOB_PROBE_DURATION = dt
    return rssi

# ======== Kept indices / thresholds ========
def recv_kept_indices():
    log.banner("Stage 2 — Indices & Thresholds")
    kept = None; total = None; got = 0; deadline = time.time() + 60.0
    while time.time() < deadline:
        t, pl = _recv(TYPE_KEEP, timeout=5.0)
        if t is None: continue
        if len(pl) < 6: log.warn("Short kept-index header; skip"); continue
        total_count, offset_count, chunk_count = struct.unpack("<HHH", pl[:6])
        need = 2 * chunk_count
        if len(pl) < 6 + need:
            log.warn(f"Expected {need} index bytes, got {len(pl)-6}; skip"); continue
        if total is None:
            total = total_count; kept = [None] * total
        if total_count != total:
            log.warn(f"Mismatched total ({total_count} != {total}); skip"); continue
        idx_list = list(struct.unpack("<" + "H"*chunk_count, pl[6:6+need]))
        for i, v in enumerate(idx_list):
            pos = offset_count + i
            if 0 <= pos < total:
                if kept[pos] is None: got += 1
                kept[pos] = v
        log.progress("Recv kept idx", got, total)
        if got == total:
            log.ok(f"Kept indices received ({total})"); return kept
        deadline = time.time() + 20.0
    raise TimeoutError("Kept indices reception timed out")

def recv_thresholds():
    t, pl = _recv(TYPE_THR, timeout=20.0)
    if t is None or len(pl) < 8: raise TimeoutError("Thresholds not received/short")
    lo, hi = struct.unpack("<ff", pl[:8])
    log.ok(f"Thresholds received  lo={lo:.2f}  hi={hi:.2f}")
    return lo, hi

# ======== ECC receive (cap-safe with END grace) ========
def recv_ecc():
    log.banner("Stage 3 — ECC")
    t, pl = _recv((TYPE_SYN, TYPE_SYN_CHUNK, TYPE_SYN_END), timeout=12.0)

    # Monolithic path
    if t == TYPE_SYN and pl and len(pl) >= 5:
        total_bits, n, nblocks = struct.unpack("<HBH", pl[:5])
        syn_bytes_len = bch.syn_bytes_len(nblocks)
        syn_bytes = pl[5:5 + syn_bytes_len]
        crc_bytes = pl[5 + syn_bytes_len: 5 + syn_bytes_len + nblocks]
        log.kv("ECC blocks", nblocks); log.ok("ECC received (monolithic)")
        return total_bits, n, bch.unpack_syndromes(syn_bytes, nblocks), list(crc_bytes)

    # Chunked path
    log.info("Receiving chunked ECC…")
    buf = {}               # seq -> bytes
    declared_parts = None
    declared_total = None
    last_activity = time.time()
    parts_seen = 0
    end_seen = False
    end_deadline = None

    def done():
        """Return True if we have enough to stop waiting."""
        if declared_parts is not None and len(buf) >= declared_parts:
            return True
        if declared_total is not None:
            total_len = sum(len(v) for v in buf.values())
            if total_len >= declared_total:
                return True
        if end_seen and end_deadline is not None and time.time() >= end_deadline:
            return True
        return False

    # First packet handling
    if t == TYPE_SYN_CHUNK and pl and len(pl) >= 2:
        seq = struct.unpack("<H", pl[:2])[0]
        buf[seq] = pl[2:]; parts_seen = 1
        last_activity = time.time()
        log.progress("ECC chunks", parts_seen, parts_seen)
    elif t == TYPE_SYN_END and pl and len(pl) >= 4:
        declared_parts, declared_total = struct.unpack("<HH", pl[:4])
        last_activity = time.time()
        end_seen = True
        end_deadline = time.time() + END_GRACE_SEC
        log.info(f"ECC END announced: parts={declared_parts}, total={declared_total}")

    # Collect more until done() or idle timeout
    while (time.time() - last_activity) < IDLE_LIMIT_ECC:
        if done():
            break
        t, pl = _recv((TYPE_SYN_CHUNK, TYPE_SYN_END), timeout=5.0)
        if t is None:
            continue
        if t == TYPE_SYN_CHUNK:
            if len(pl) < 2:
                log.warn("short SYN_CHUNK; skipping"); continue
            seq = struct.unpack("<H", pl[:2])[0]
            if seq not in buf: parts_seen += 1
            buf[seq] = pl[2:]
            last_activity = time.time()
            log.progress("ECC chunks", len(buf), declared_parts or len(buf))
        elif t == TYPE_SYN_END and len(pl) >= 4:
            declared_parts, declared_total = struct.unpack("<HH", pl[:4])
            last_activity = time.time()
            end_seen = True
            end_deadline = time.time() + END_GRACE_SEC
            log.info(f"ECC END received: parts={declared_parts}, total={declared_total}")

    if not buf:
        log.err("No ECC chunks received")
        return 0, 0, [], []

    payload = b"".join(buf[i] for i in sorted(buf))
    if declared_parts is not None and len(buf) != declared_parts:
        log.warn(f"Missing chunks: got {len(buf)} / {declared_parts}")
    if declared_total is not None and len(payload) != declared_total:
        log.warn(f"Size mismatch: got {len(payload)} vs declared {declared_total}")

    if len(payload) < 5:
        log.err("ECC buffer too short after chunking")
        return 0, 0, [], []

    total_bits, n, nblocks = struct.unpack("<HBH", payload[:5])
    syn_bytes_len = bch.syn_bytes_len(nblocks)
    syn_bytes = payload[5:5 + syn_bytes_len]
    crc_bytes = payload[5 + syn_bytes_len: 5 + syn_bytes_len + nblocks]

    log.kv("ECC blocks", nblocks)
    log.ok("ECC chunked payload assembled")
    return total_bits, n, bch.unpack_syndromes(syn_bytes, nblocks), list(crc_bytes)

def quantise_with_thresholds(rssi, kept_idx, lo, hi):
    bits = []; mid = 0.5 * (lo + hi)
    for i in kept_idx:
        v = rssi[i]
        if v >= hi: bits.append(1)
        elif v <= lo: bits.append(0)
        else: bits.append(1 if v >= mid else 0)
    return bits

def bits_to_bytes(bits):
    if not bits: return b""
    pad = (-len(bits)) % 8
    if pad: bits = bits + [0]*pad
    v = 0
    for b in bits: v = (v<<1) | (b & 1)
    return v.to_bytes(len(bits)//8, "big")

if __name__ == "__main__":
    # 1) Probing
    rssi = run_probing()

    # 2) Receive indices + thresholds from Alice
    kept_idx = recv_kept_indices()
    lo, hi   = recv_thresholds()

    # 3) Signal ready for ECC
    _send(TYPE_READY_ECC, b""); log.ok("Sent READY_ECC to Alice")

    # 4) Receive ECC and run IR
    total_bits, n, syn_list, crc_list = recv_ecc()
    if total_bits == 0:
        log.err("No usable ECC received; aborting IR")
        _send(TYPE_DROP, struct.pack("<H", 0))
        log.ok(f"Final bits=0  SHA-256={hashlib.sha256(b'').hexdigest()}")
        _send(TYPE_DONE, b""); raise SystemExit(0)

    bitsB = quantise_with_thresholds(rssi, kept_idx, lo, hi)
    blocksB = bch.chunk_bits(bitsB[:total_bits], n)

    # --- Metrics/NIST: pre-ECC dump + KGR (Bob side) ---
    try: duration = _BOB_PROBE_DURATION
    except Exception: duration = 0.0
    save_bits_ascii01("logs/bob_pre_bits.txt", bitsB[:total_bits])
    save_bits_bin("logs/bob_pre_bits.bin", bitsB[:total_bits])
    bpp, bps = kgr_bits(len(bitsB[:total_bits]), PROBE_COUNT, duration)
    log.kv("Pre-ECC bits (Bob)", len(bitsB[:total_bits]))
    log.ok(f"[KGR] (Bob) bits/probe={bpp:.4f}  bits/s={bps:.2f}  duration={duration:.2f}s")

    if len(syn_list) != math.ceil(total_bits / n):
        log.warn(f"syn_list={len(syn_list)} vs ceil(total_bits/n)={math.ceil(total_bits/n)}")
    if len(crc_list) != math.ceil(total_bits / n):
        log.warn(f"crc_list={len(crc_list)} vs ceil(total_bits / n)={math.ceil(total_bits / n)}")

    kept_blocks = []; dropped_idx = []; flips_total = 0
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

    # report dropped block indices to Alice
    MAX_DROP_CHUNK = 10
    total_drops = len(dropped_idx); sent = 0
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
