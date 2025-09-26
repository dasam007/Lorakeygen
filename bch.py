# bch.py — BCH(31,21,t=2) helpers with 10-bit syndromes
# Public API kept: syndrome, correct_with_syndrome, crc8, chunk_bits, pack_syndromes, unpack_syndromes
# Added: syn_bytes_len(nblocks)

from typing import Iterable, List, Tuple

BCH_N  = 31
BCH_K  = 21
SYN_BITS = BCH_N - BCH_K  # 10
BCH_T  = 2

BCH_DEBUG = False

# g(x) = x^10 + x^9 + x^8 + x^6 + x^5 + x^3 + 1  (POCSAG)
G_POLY = (1<<10) | (1<<9) | (1<<8) | (1<<6) | (1<<5) | (1<<3) | 1  # 0b1110101001 = 1897

def _sanitize_bits(bits: Iterable[int], n: int) -> List[int]:
    out = [(b & 1) for b in bits]
    if len(out) < n:
        out.extend([0]*(n - len(out)))
    return out[:n]

def _int_from_bits_le(bits: Iterable[int]) -> int:
    x = 0
    for i, b in enumerate(bits):
        if b & 1: x |= (1 << i)
    return x

def _bits_from_int_le(x: int, n: int) -> List[int]:
    return [(x >> i) & 1 for i in range(n)]

def _poly_mod(msg: int, g: int = G_POLY) -> int:
    deg_g = g.bit_length() - 1
    while msg.bit_length() - 1 >= deg_g:
        shift = (msg.bit_length() - 1) - deg_g
        msg ^= (g << shift)
    return msg  # remainder < 2^deg_g

# ------- precompute syndrome→error mask for weight ≤2 -------
REM_SINGLE: List[int] = []
SYN2MASK   = {}
AMBIGUOUS  = set()

def _build_tables():
    global REM_SINGLE, SYN2MASK, AMBIGUOUS
    if REM_SINGLE: return
    # weight-1
    for i in range(BCH_N):
        rem = _poly_mod(1 << i)
        REM_SINGLE.append(rem)
        if rem not in SYN2MASK:
            SYN2MASK[rem] = (1 << i)
        elif SYN2MASK[rem] != (1 << i):
            AMBIGUOUS.add(rem)
    # weight-2
    for i in range(BCH_N):
        ri = REM_SINGLE[i]
        for j in range(i+1, BCH_N):
            rem = ri ^ REM_SINGLE[j]            # linearity
            mask = (1 << i) | (1 << j)
            if rem not in SYN2MASK:
                SYN2MASK[rem] = mask
            elif SYN2MASK[rem] != mask:
                AMBIGUOUS.add(rem)

_build_tables()

# ---------------- public API ----------------
def syn_bytes_len(nblocks: int) -> int:
    return (SYN_BITS * nblocks + 7) // 8

def syndrome(block: Iterable[int]) -> int:
    b = _sanitize_bits(block, BCH_N)
    return _poly_mod(_int_from_bits_le(b))  # 10-bit remainder

def correct_with_syndrome(bits: Iterable[int], syn_err: int) -> Tuple[List[int], int]:
    b = _sanitize_bits(bits, BCH_N)
    if syn_err == 0: return b, 0
    if syn_err in AMBIGUOUS: return b, 0
    mask = SYN2MASK.get(syn_err, 0)
    if mask == 0: return b, 0
    x = _int_from_bits_le(b) ^ mask
    out = _bits_from_int_le(x, BCH_N)
    flips = 1 if (mask & (mask - 1)) == 0 else 2
    return out, flips

# CRC-8 (poly=0x07) bit-level
def crc8(bits: Iterable[int], poly: int = 0x07, init: int = 0x00) -> int:
    v = init & 0xFF
    for b in bits:
        b &= 1
        msb = ((v ^ (b << 7)) & 0x80) != 0
        v = ((v << 1) & 0xFF)
        if msb: v ^= poly
    return v & 0xFF

def chunk_bits(bits: Iterable[int], n: int) -> List[List[int]]:
    bl = [(b & 1) for b in bits]
    out = [bl[i:i+n] for i in range(0, len(bl), n)]
    if out and len(out[-1]) < n:
        out[-1].extend([0] * (n - len(out[-1])))
    return out

# pack/unpack 10-bit syndromes (LSB-first inside each 10b word)
def pack_syndromes(syn_list: List[int]) -> bytes:
    total_bits = SYN_BITS * len(syn_list)
    out = bytearray((total_bits + 7) // 8)
    bitpos = 0
    for s in syn_list:
        v = s & ((1 << SYN_BITS) - 1)
        for k in range(SYN_BITS):
            if v & (1 << k):
                out[bitpos >> 3] |= (1 << (bitpos & 7))
            bitpos += 1
    return bytes(out)

def unpack_syndromes(data: bytes, nblocks: int) -> List[int]:
    syn = [0] * nblocks
    bitpos = 0
    for i in range(nblocks):
        v = 0
        for k in range(SYN_BITS):
            byte_index = bitpos >> 3
            bit_index  = bitpos & 7
            bit = (data[byte_index] >> bit_index) & 1 if byte_index < len(data) else 0
            v |= (bit << k)
            bitpos += 1
        syn[i] = v
    return syn
