import numpy as np
from dataclasses import dataclass
from pathlib import Path

@dataclass
class RawImage:
    name: str
    path: Path
    width: int
    height: int
    mode: str
    payload: bytes
    extra: dict


class LZ77:
    def __init__(self, window_size: int = 4096, max_match: int = 32, min_match: int = 3, max_candidates: int = 64):
        self.window_size = int(window_size)
        self.max_match = int(max_match)
        self.min_match = int(min_match)
        self.max_candidates = int(max_candidates)

    def compress(self, data: bytes) -> bytes:
        n = len(data)
        out = bytearray()
        pos_by_key = {}

        i = 0
        while i < n:
            best_len = 0
            best_dist = 0

            if i + self.min_match <= n:
                key = data[i:i + self.min_match]
                cand = pos_by_key.get(key)
                if cand:
                    start = max(0, i - self.window_size)
                    kept = []
                    for p in cand:
                        if p >= start:
                            kept.append(p)
                    cand = kept
                    if cand:
                        pos_by_key[key] = cand
                    else:
                        pos_by_key.pop(key, None)

                    if cand:
                        checked = 0
                        for p in reversed(cand):
                            checked += 1
                            if checked > self.max_candidates:
                                break
                            dist = i - p
                            if dist <= 0:
                                continue
                            lim = min(self.max_match, n - i)
                            l = 0
                            while l < lim and data[p + l] == data[i + l]:
                                l += 1
                            if l > best_len:
                                best_len = l
                                best_dist = dist
                                if best_len == self.max_match:
                                    break

            if best_len >= self.min_match:
                out.append(1)
                out.extend(int(best_dist).to_bytes(2, "little", signed=False))
                out.append(int(best_len))
                for k in range(best_len):
                    j = i + k
                    if j + self.min_match <= n:
                        kkey = data[j:j + self.min_match]
                        lst = pos_by_key.get(kkey)
                        if lst is None:
                            pos_by_key[kkey] = [j]
                        else:
                            lst.append(j)
                i += best_len
            else:
                out.append(0)
                out.append(data[i])
                if i + self.min_match <= n:
                    kkey = data[i:i + self.min_match]
                    lst = pos_by_key.get(kkey)
                    if lst is None:
                        pos_by_key[kkey] = [i]
                    else:
                        lst.append(i)
                i += 1

        return bytes(out)

    def decompress(self, blob: bytes) -> bytes:
        out = bytearray()
        i = 0
        n = len(blob)
        while i < n:
            t = blob[i]
            i += 1
            if t == 0:
                out.append(blob[i])
                i += 1
            elif t == 1:
                dist = int.from_bytes(blob[i:i + 2], "little", signed=False)
                i += 2
                ln = blob[i]
                i += 1
                if dist == 0 or ln == 0 or dist > len(out):
                    raise ValueError("LZ77: token invalide")
                start = len(out) - dist
                for _ in range(ln):
                    out.append(out[start])
                    start += 1
            else:
                raise ValueError("LZ77: type token inconnu")
        return bytes(out)


class LZW12:
    def __init__(self, max_dict_size: int = 4096, code_width: int = 12):
        self.max_dict_size = int(max_dict_size)
        self.code_width = int(code_width)

    def compress(self, data: bytes) -> bytes:
        if not data:
            return b""

        next_code = 256
        table = {}
        w = data[0]

        codes = []
        for b in data[1:]:
            key = (w, b)
            c = table.get(key)
            if c is not None:
                w = c
            else:
                codes.append(w)
                if next_code < self.max_dict_size:
                    table[key] = next_code
                    next_code += 1
                w = b
        codes.append(w)

        out = bytearray()
        buf = 0
        bits = 0
        mask = (1 << self.code_width) - 1
        for c in codes:
            buf |= (int(c) & mask) << bits
            bits += self.code_width
            while bits >= 8:
                out.append(buf & 0xFF)
                buf >>= 8
                bits -= 8
        if bits:
            out.append(buf & 0xFF)
        return bytes(out)

    def decompress(self, blob: bytes) -> bytes:
        if not blob:
            return b""

        codes = []
        buf = 0
        bits = 0
        i = 0
        mask = (1 << self.code_width) - 1
        while i < len(blob):
            buf |= blob[i] << bits
            bits += 8
            i += 1
            while bits >= self.code_width:
                codes.append(buf & mask)
                buf >>= self.code_width
                bits -= self.code_width

        prefix = np.full(self.max_dict_size, -1, dtype=np.int32)
        ch = np.zeros(self.max_dict_size, dtype=np.uint8)
        for k in range(256):
            ch[k] = k

        def expand(code: int, stack: bytearray) -> int:
            while code >= 256:
                stack.append(int(ch[code]))
                code = int(prefix[code])
            stack.append(int(ch[code]))
            return stack[-1]

        next_code = 256
        out = bytearray()

        old = codes[0]
        stack = bytearray()
        first = expand(old, stack)
        out.extend(reversed(stack))
        stack.clear()

        for code in codes[1:]:
            if code < next_code:
                first_char = expand(code, stack)
                out.extend(reversed(stack))
                stack.clear()
            elif code == next_code:
                first_char = expand(old, stack)
                stack.reverse()
                stack.append(first_char)
                out.extend(stack)
                first_char = stack[-len(stack)]
                stack.clear()
            else:
                raise ValueError("LZW: code invalide")

            if next_code < self.max_dict_size:
                prefix[next_code] = old
                ch[next_code] = first_char
                next_code += 1

            old = code

        return bytes(out)


class RLE16:
    def __init__(self):
        pass

    def compress(self, arr_u16: np.ndarray) -> bytes:
        x = arr_u16.astype(np.uint16, copy=False).ravel()
        if x.size == 0:
            return b""
        out = bytearray()
        cur = int(x[0])
        run = 1
        for v in x[1:]:
            vv = int(v)
            if vv == cur and run < 65535:
                run += 1
            else:
                out.extend(int(cur).to_bytes(2, "little", signed=False))
                out.extend(int(run).to_bytes(2, "little", signed=False))
                cur = vv
                run = 1
        out.extend(int(cur).to_bytes(2, "little", signed=False))
        out.extend(int(run).to_bytes(2, "little", signed=False))
        return bytes(out)

    def decompress(self, blob: bytes, shape: tuple[int, int]) -> np.ndarray:
        if len(blob) % 4 != 0:
            raise ValueError("RLE: flux invalide")
        out = np.empty(shape[0] * shape[1], dtype=np.uint16)
        k = 0
        i = 0
        n = len(blob)
        while i < n:
            val = int.from_bytes(blob[i:i + 2], "little", signed=False)
            ln = int.from_bytes(blob[i + 2:i + 4], "little", signed=False)
            i += 4
            if ln <= 0:
                raise ValueError("RLE: longueur invalide")
            if k + ln > out.size:
                raise ValueError("RLE: dépassement sortie")
            out[k:k + ln] = val
            k += ln
        if k != out.size:
            raise ValueError("RLE: taille décodée incorrecte")
        return out.reshape(shape)
