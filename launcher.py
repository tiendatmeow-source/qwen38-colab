# -*- coding: utf-8 -*-
"""
Qwen3.8-27B · Colab one-click launcher
======================================
Một lần gọi `run()` sẽ tự động:

  1. Dò GPU / RAM / ổ đĩa và chọn bản GGUF chất lượng cao nhất còn vừa VRAM.
  2. Cài llama.cpp (bản dựng sẵn CUDA; không được thì tự build từ mã nguồn).
  3. Tải model (song song với bước 2) kèm thanh tiến độ.
  4. Nạp model, tự co giãn context cho vừa GPU, chạy thử một câu để đo tốc độ.
  5. Mở Cloudflare Tunnel và trả về link chatbox công khai (Cloudflare từ chối thì chuyển sang localhost.run).
  6. Canh gác: tự khởi động lại model / tunnel nếu chết, hiển thị thống kê trực tiếp.

Chỉ dùng thư viện chuẩn của Python. Nguồn: https://github.com/tiendatmeow-source/qwen38-colab
"""
import atexit
import html
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

__version__ = "1.0.0"

MiB = 1024 * 1024
GiB = 1024 * MiB
PINNED_BUILD = "b12110"  # llama.cpp v0.9.0 — bản dùng khi viết và kiểm thử launcher này
LLAMA_REPO = "ggml-org/llama.cpp"

DEFAULTS = {
    # ---- các lựa chọn hiện trên form của notebook ----
    "quant": "auto",         # "auto" hoặc tên quant, ví dụ "UD-IQ4_XS", "UD-Q3_K_XL"
    "reasoning": "medium",   # "low" | "medium" | "xhigh" | "off"
    "vision": "auto",        # "auto" | "on" | "off"
    "context": "auto",       # "auto" hoặc số token, ví dụ 16384
    "password": "",          # nếu đặt -> thành API key của server
    "mtp": False,            # tăng tốc bằng Multi-Token Prediction (thử nghiệm)
    "llama_build": PINNED_BUILD,  # tag llama.cpp, hoặc "stable"
    "hf_token": "",
    # ---- nâng cao ----
    "repo": "unsloth/Qwen3.8-27B-GGUF",
    "port": 8080,
    "workdir": "",
    "allow_cpu": False,
    "kv_type": "q8_0",       # kiểu nén KV cache
    "fit_margin_mib": 512,   # VRAM chừa trống cho driver / phân mảnh
    "min_ctx": 8192,         # sàn context khi để "auto"
    "target_ctx": 16384,     # context dùng để ước lượng khi chọn quant
    "self_test": True,
    "qr": True,
    "block": True,
    "exit_after_s": 0,       # chỉ dùng cho kiểm thử tự động
}

# Số đo lấy từ metadata GGUF thật của Qwen3.8-27B (kiến trúc qwen35):
# 65 block, chỉ 16 block full-attention (full_attention_interval = 4),
# 4 KV head x 256 chiều -> 16 * 2 * 4 * 256 * 2 byte = 65 536 byte/token ở f16.
# token_embd (521 MiB) luôn nằm trên RAM, không tốn VRAM.
PROFILES = {
    "qwen3.8-27b": {"emb_mib": 521, "kv_f16_bytes": 65536, "overhead_mib": 650,
                    "mmproj_extra_mib": 250, "mtp_extra_mib": 700},
}
GENERIC_PROFILE = {"emb_mib": 0, "kv_f16_bytes": 120_000, "overhead_mib": 500,
                   "mmproj_extra_mib": 250, "mtp_extra_mib": 700}
KV_BITS = {"f32": 32, "f16": 16, "bf16": 16, "q8_0": 8.5, "q5_1": 6, "q5_0": 5.5,
           "q4_1": 5, "q4_0": 4.5, "iq4_nl": 4.5}

SAMPLING = {  # khuyến nghị chính thức của Qwen / Unsloth
    "thinking": ["--temp", "1.0", "--top-p", "0.95", "--top-k", "20", "--min-p", "0.0",
                 "--presence-penalty", "0.0", "--repeat-penalty", "1.0"],
    "instruct": ["--temp", "0.7", "--top-p", "0.8", "--top-k", "20", "--min-p", "0.0",
                 "--presence-penalty", "1.5", "--repeat-penalty", "1.0"],
}


class LauncherError(Exception):
    """Lỗi có thông điệp thân thiện, hiển thị thẳng cho người dùng."""


_CANCEL = threading.Event()  # bật khi người dùng bấm Stop -> các luồng tải/build tự dừng


def _check_cancel():
    if _CANCEL.is_set():
        raise LauncherError("đã huỷ")


# --------------------------------------------------------------------------- #
# Tiện ích nhỏ
# --------------------------------------------------------------------------- #
def _fmt_size(n):
    n = float(n)
    return f"{n / GiB:.1f} GB" if n >= GiB else f"{n / MiB:.0f} MB"


def _fmt_dur(s):
    s = int(max(0, s))
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def _sh(cmd, timeout=60, env=None):
    """Chạy lệnh, trả về (mã thoát, stdout+stderr). Không bao giờ ném lỗi."""
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=timeout, env=env, text=True, errors="replace")
        return p.returncode, p.stdout or ""
    except subprocess.TimeoutExpired as e:
        out = e.stdout if isinstance(e.stdout, str) else (e.stdout or b"").decode("utf-8", "replace")
        return 124, out
    except Exception as e:  # noqa: BLE001
        return 127, str(e)


def _http(url, timeout=15, headers=None, data=None, method=None):
    """GET/POST đơn giản. Trả về (status, bytes); status = 0 nếu không kết nối được."""
    hdrs = {"User-Agent": f"qwen38-colab/{__version__}"}
    hdrs.update(headers or {})
    req = urllib.request.Request(url, headers=hdrs, data=data, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        try:
            body = e.read()
        except Exception:  # noqa: BLE001
            body = b""
        return e.code, body
    except Exception:  # noqa: BLE001
        return 0, b""


def _download(url, dest, progress=None, tries=4):
    """Tải file vừa/nhỏ (binary) bằng urllib, có thử lại và báo tiến độ."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": f"qwen38-colab/{__version__}"})
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                done = 0
                while True:
                    _check_cancel()
                    chunk = r.read(4 * MiB)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(done, total)
            if total and done != total:
                raise IOError(f"tải thiếu dữ liệu ({done}/{total} byte)")
            tmp.replace(dest)
            return dest
        except Exception as e:  # noqa: BLE001
            _check_cancel()
            last = e
            time.sleep(2 + 3 * attempt)
    raise LauncherError(f"Không tải được {url}: {last}")


def _free_port(preferred):
    for port in [preferred] + list(range(preferred + 1, preferred + 40)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise LauncherError("Không tìm được cổng trống cho server.")


def _tail(path, n=25):
    try:
        with open(path, "r", errors="replace") as f:
            return [ln.rstrip() for ln in f.readlines()[-n:]]
    except OSError:
        return []


def _allocated_bytes(root):
    """Dung lượng đã ghi thật trên đĩa (đúng cả với file thưa của hf-xet)."""
    total = 0
    for dp, _dn, fns in os.walk(root):
        for fn in fns:
            try:
                st = os.stat(os.path.join(dp, fn))
                total += min(st.st_size, st.st_blocks * 512)
            except OSError:
                pass
    return total


def _net_rx_bytes():
    """Tổng byte đã nhận qua mạng (dùng để biết việc tải còn sống hay đã treo)."""
    total = 0
    try:
        with open("/proc/net/dev") as f:
            for ln in f.readlines()[2:]:
                name, data = ln.split(":", 1)
                if name.strip() != "lo":
                    total += int(data.split()[0])
    except Exception:  # noqa: BLE001
        pass
    return total


def _in_notebook():
    try:
        from IPython import get_ipython  # noqa: PLC0415
        ip = get_ipython()
        return ip is not None and hasattr(ip, "kernel")
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# Bảng điều khiển: thẻ HTML sống trong notebook, hoặc in chữ thường khi chạy CLI
# --------------------------------------------------------------------------- #
_CSS = """
<style>
.q38{font-family:ui-sans-serif,system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;max-width:860px;
 background:linear-gradient(145deg,#0f172a,#1e1b4b);color:#e2e8f0;border-radius:16px;padding:20px 22px;
 border:1px solid #334155;line-height:1.5;font-size:14px;box-sizing:border-box}
.q38 *{box-sizing:border-box}
.q38 .hd{display:flex;align-items:center;gap:12px;margin-bottom:14px}
.q38 .logo{width:42px;height:42px;border-radius:12px;background:linear-gradient(135deg,#8b5cf6,#06b6d4);
 display:flex;align-items:center;justify-content:center;font-size:22px;flex:none}
.q38 .ttl{font-size:17px;font-weight:700;color:#f8fafc}
.q38 .sub{font-size:12px;color:#94a3b8}
.q38 .st{display:flex;gap:10px;align-items:flex-start;padding:5px 0}
.q38 .ic{width:20px;height:20px;flex:none;display:flex;align-items:center;justify-content:center;
 border-radius:50%;font-size:12px;font-weight:700;margin-top:1px}
.q38 .ic.pending{border:2px solid #475569}
.q38 .ic.running{border:2px solid #334155;border-top-color:#22d3ee;animation:q38spin .8s linear infinite}
.q38 .ic.done{background:#10b981;color:#052e16}
.q38 .ic.warn{background:#f59e0b;color:#451a03}
.q38 .ic.fail{background:#ef4444;color:#fff}
@keyframes q38spin{to{transform:rotate(360deg)}}
.q38 .lb{font-weight:600;color:#f1f5f9}
.q38 .st.pending .lb{color:#64748b;font-weight:500}
.q38 .dt{font-size:12.5px;color:#94a3b8;word-break:break-word}
.q38 .bar{height:6px;background:#1e293b;border-radius:99px;margin-top:5px;overflow:hidden;max-width:520px}
.q38 .bar i{display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,#8b5cf6,#22d3ee)}
.q38 .box{margin-top:14px;padding:12px 14px;border-radius:12px;font-size:13px}
.q38 .box.err{background:#450a0a;border:1px solid #b91c1c;color:#fecaca}
.q38 .box.info{background:#0c2a3e;border:1px solid #0e7490;color:#cffafe}
.q38 .box.warn{background:#422006;border:1px solid #b45309;color:#fde68a}
.q38 pre{white-space:pre-wrap;word-break:break-word;font-size:11.5px;background:#020617;color:#cbd5e1;
 padding:10px;border-radius:8px;margin:8px 0 0;max-height:220px;overflow:auto}
.q38 .ready{margin-top:6px;display:flex;gap:18px;flex-wrap:wrap;align-items:center}
.q38 .cta{display:inline-block;background:linear-gradient(135deg,#8b5cf6,#06b6d4);color:#fff!important;
 font-weight:700;font-size:16px;padding:12px 22px;border-radius:12px;text-decoration:none!important}
.q38 .url{margin-top:10px;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px;background:#020617;
 border:1px solid #334155;padding:8px 10px;border-radius:8px;color:#67e8f9;word-break:break-all;user-select:all}
.q38 .qr{background:#fff;padding:8px;border-radius:12px;flex:none;line-height:0}
.q38 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:8px;margin-top:14px}
.q38 .cell{background:rgba(15,23,42,.65);border:1px solid #334155;border-radius:10px;padding:8px 10px}
.q38 .cell b{display:block;font-size:11px;color:#94a3b8;font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.q38 .cell span{font-size:13.5px;color:#f1f5f9}
.q38 .live{margin-top:12px;font-size:12.5px;color:#a5b4fc}
.q38 .dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#10b981;margin-right:6px;
 animation:q38pulse 1.6s ease-in-out infinite}
@keyframes q38pulse{50%{opacity:.35}}
.q38 code{background:#020617;color:#a5f3fc;padding:1px 5px;border-radius:5px;font-size:12px}
.q38 details{margin-top:12px;font-size:12.5px;color:#cbd5e1}
.q38 summary{cursor:pointer;color:#a5b4fc;font-weight:600}
.q38 .ft{margin-top:12px;font-size:12px;color:#94a3b8}
</style>
"""
_ICON_TXT = {"pending": "", "running": "", "done": "✓", "warn": "!", "fail": "✕"}


class Board:
    def __init__(self, title, subtitle):
        self.title, self.subtitle = title, subtitle
        self.steps, self.order = {}, []
        self.boxes = []       # [(kind, html)]
        self.ready = None     # dict khi đã có link
        self.live = ""
        self.rich = _in_notebook()
        self._handle = None
        self._live_handle = None
        self._live_shown = None
        self._dirty = True
        self._printed = {}
        self._lock = threading.RLock()

    # ---- trạng thái (an toàn khi gọi từ luồng phụ) ----
    def add(self, key, label):
        with self._lock:
            self.steps[key] = {"label": label, "state": "pending", "detail": "", "pct": None}
            self.order.append(key)
            self._dirty = True

    def set(self, key, state=None, detail=None, pct=-1):
        with self._lock:
            s = self.steps[key]
            if state is not None:
                s["state"] = state
            if detail is not None:
                s["detail"] = detail
            if pct != -1:
                s["pct"] = pct
            self._dirty = True

    def box(self, kind, body_html, replace=False):
        with self._lock:
            if replace:
                self.boxes = [b for b in self.boxes if b[0] != kind]
            self.boxes.append((kind, body_html))
            self._dirty = True

    def clear_boxes(self, kind=None):
        with self._lock:
            self.boxes = [b for b in self.boxes if kind and b[0] != kind]
            self._dirty = True

    def touch(self):
        self._dirty = True

    def set_live(self, text):
        self.live = text
        if not self.rich:
            self._dirty = True

    # ---- vẽ (chỉ gọi từ luồng chính) ----
    def render(self, force=False):
        with self._lock:
            if self._dirty or force:
                self._dirty = False
                if self.rich:
                    self._render_html()
                else:
                    self._render_text()
            if self.rich and self.live != self._live_shown:
                self._render_live()

    def _render_live(self):
        # Dòng thống kê nằm ở khung riêng: cập nhật 10 giây/lần mà không vẽ lại thẻ chính
        from IPython.display import HTML, display  # noqa: PLC0415
        self._live_shown = self.live
        body = (f'{_CSS}<div class="q38" style="margin-top:8px;padding:10px 16px"><div class="live" '
                f'style="margin:0"><span class="dot"></span>{html.escape(self.live)}</div></div>') if self.live else ""
        if self._live_handle is None:
            if body:
                self._live_handle = display(HTML(body), display_id=True)
        else:
            self._live_handle.update(HTML(body))

    def _render_html(self):
        from IPython.display import HTML, display  # noqa: PLC0415
        doc = HTML(self.html())
        if self._handle is None:
            self._handle = display(doc, display_id=True)
        else:
            self._handle.update(doc)

    def html(self):
        e = html.escape
        out = [_CSS, '<div class="q38">',
               f'<div class="hd"><div class="logo">🧠</div><div><div class="ttl">{e(self.title)}</div>'
               f'<div class="sub">{e(self.subtitle)}</div></div></div>']
        steps = [self.steps[k] for k in self.order]
        if self.ready:
            out.append(self._ready_html())
            out.append("<details><summary>Chi tiết các bước cài đặt</summary>")
        for s in steps:
            st = s["state"]
            bar = ""
            if st == "running" and s["pct"] is not None:
                bar = f'<div class="bar"><i style="width:{max(0, min(100, s["pct"])):.1f}%"></i></div>'
            det = f'<div class="dt">{e(s["detail"])}</div>' if s["detail"] else ""
            out.append(f'<div class="st {st}"><div class="ic {st}">{_ICON_TXT[st]}</div>'
                       f'<div style="flex:1"><div class="lb">{e(s["label"])}</div>{det}{bar}</div></div>')
        if self.ready:
            out.append("</details>")
        for kind, body in self.boxes:
            out.append(f'<div class="box {kind}">{body}</div>')
        out.append("</div>")
        return "".join(out)

    def _ready_html(self):
        e = html.escape
        r = self.ready
        url = e(r["url"])
        qr = f'<div class="qr">{r["qr"]}</div>' if r.get("qr") else ""
        cells = "".join(f'<div class="cell"><b>{e(k)}</b><span>{e(str(v))}</span></div>' for k, v in r["facts"])
        key_note = (f' · mật khẩu / API key: <code>{e(r["password"])}</code> '
                    f'(nhập ở ⚙ Settings → API Key trong chatbox)') if r.get("password") else ""
        return (
            f'<div class="ready"><div style="flex:1;min-width:280px">'
            f'<a class="cta" href="{url}" target="_blank" rel="noopener">💬 Mở chatbox</a>'
            f'<div class="url">{url}</div>'
            f'<div class="ft">API kiểu OpenAI: <code>{url}/v1</code>{key_note}</div>'
            f'</div>{qr}</div>'
            f'<div class="grid">{cells}</div>'
            f'<div class="ft">Link sống chừng nào ô này còn chạy. Bấm ■ (Stop) để tắt. '
            f'Mỗi lần chạy lại sẽ có link mới.</div>'
        )

    def _render_text(self):
        marks = {"pending": " ", "running": "…", "done": "✓", "warn": "!", "fail": "✕"}
        for k in self.order:
            s = self.steps[k]
            pct = s["pct"]
            bucket = None if pct is None else int(pct // 10)
            sig = (s["state"], s["detail"] if s["state"] != "running" else None, bucket)
            if s["state"] != "pending" and self._printed.get(k) != sig:
                self._printed[k] = sig
                extra = f" [{pct:.0f}%]" if (pct is not None and s["state"] == "running") else ""
                print(f"[{marks[s['state']]}] {s['label']}{extra}  {s['detail']}", flush=True)
        for i, (kind, body) in enumerate(self.boxes):
            if self._printed.get(("box", i)) != body:
                self._printed[("box", i)] = body
                print(f"--- {kind.upper()} ---\n{re.sub(r'<[^>]+>', '', html.unescape(body))}", flush=True)
        if self.ready and self._printed.get("ready") != self.ready["url"]:
            self._printed["ready"] = self.ready["url"]
            print("\n" + "=" * 64 + f"\n  CHATBOX: {self.ready['url']}\n" + "=" * 64, flush=True)
            for k, v in self.ready["facts"]:
                print(f"  {k}: {v}", flush=True)
        if self.live and self._printed.get("live") != self.live:
            self._printed["live"] = self.live
            print(f"  · {self.live}", flush=True)


# --------------------------------------------------------------------------- #
# Dò hệ thống
# --------------------------------------------------------------------------- #
def detect_gpu():
    if not shutil.which("nvidia-smi"):
        return None
    rc, out = _sh(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
                   "--format=csv,noheader,nounits"], timeout=30)
    if rc != 0 or not out.strip():
        return None
    try:
        name, total, free, drv = [x.strip() for x in out.strip().splitlines()[0].split(",")[:4]]
        return {"name": name, "total_mib": int(float(total)), "free_mib": int(float(free)), "driver": drv}
    except Exception:  # noqa: BLE001
        return None


def gpu_used_mib():
    rc, out = _sh(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                   "--format=csv,noheader,nounits"], timeout=15)
    try:
        used, util = [x.strip() for x in out.strip().splitlines()[0].split(",")[:2]]
        return int(float(used)), int(float(util))
    except Exception:  # noqa: BLE001
        return None, None


def ram_total_mib():
    try:
        with open("/proc/meminfo") as f:
            for ln in f:
                if ln.startswith("MemTotal:"):
                    return int(ln.split()[1]) // 1024
    except OSError:
        pass
    return 0


# --------------------------------------------------------------------------- #
# Chọn model
# --------------------------------------------------------------------------- #
_QUANT_RE = re.compile(r"[-.]((?:UD-)?(?:I?Q\d\w*|BF16|F16|F32))\.gguf$", re.I)


def hf_list(repo, token=""):
    hdr = {"Authorization": f"Bearer {token}"} if token else {}
    for attempt in range(4):
        st, body = _http(f"https://huggingface.co/api/models/{repo}/tree/main", timeout=30, headers=hdr)
        if st == 200:
            files = []
            for it in json.loads(body):
                if it.get("type") == "file" and it["path"].lower().endswith(".gguf"):
                    files.append({"path": it["path"], "size": int(it.get("size") or (it.get("lfs") or {}).get("size") or 0)})
            return files
        if st in (401, 403, 404):
            raise LauncherError(f"Không đọc được repo Hugging Face “{repo}” (HTTP {st}). "
                                f"Kiểm tra tên repo hoặc HF_TOKEN.")
        time.sleep(2 + 2 * attempt)
    raise LauncherError("Không kết nối được Hugging Face để lấy danh sách file. Thử chạy lại sau ít phút.")


def split_files(files):
    models, mmprojs = [], []
    for f in files:
        name = f["path"]
        if "/" in name or re.search(r"-\d{5}-of-\d{5}", name):
            continue
        if name.lower().startswith("mmproj"):
            mmprojs.append(f)
            continue
        if name.lower().startswith(("imatrix", "mtp-")):
            continue
        m = _QUANT_RE.search(name)
        models.append({**f, "quant": m.group(1) if m else name[:-5]})
    mmproj = None
    for pref in ("mmproj-f16.gguf", "mmproj-bf16.gguf"):
        mmproj = mmproj or next((f for f in mmprojs if f["path"].lower() == pref), None)
    mmproj = mmproj or (mmprojs[0] if mmprojs else None)
    return models, mmproj


def profile_for(repo):
    low = repo.lower()
    for key, prof in PROFILES.items():
        if key in low:
            return prof
    return GENERIC_PROFILE


def need_mib(size_bytes, prof, ctx, kv_type, margin, vision_bytes=0, mtp=False):
    """Ước lượng VRAM (MiB) để chạy trọn trên GPU."""
    weights = size_bytes / MiB - prof["emb_mib"]
    kv = ctx * prof["kv_f16_bytes"] * KV_BITS.get(kv_type, 16) / 16 / MiB
    vis = (vision_bytes / MiB + prof["mmproj_extra_mib"]) if vision_bytes else 0
    return weights + prof["overhead_mib"] + kv + margin + vis + (prof["mtp_extra_mib"] if mtp else 0)


def make_plans(cfg, models, mmproj, gpu, disk_free):
    """Trả về danh sách phương án [chính, dự phòng 1, dự phòng 2 ...]."""
    prof = profile_for(cfg["repo"])
    ctx_explicit = None if str(cfg["context"]).lower() == "auto" else int(cfg["context"])
    ctx_est = ctx_explicit or int(cfg["target_ctx"])
    free = gpu["free_mib"] if gpu else None
    vis_bytes = mmproj["size"] if mmproj else 0
    want_vision = cfg["vision"] if mmproj else "off"
    disk_ok = [m for m in models if m["size"] + vis_bytes + 2 * GiB < disk_free]
    if not disk_ok:
        raise LauncherError("Ổ đĩa không đủ chỗ cho bất kỳ bản model nào.")
    by_size = sorted(disk_ok, key=lambda m: m["size"], reverse=True)

    def est(m, vision, mtp=cfg["mtp"]):
        return need_mib(m["size"], prof, ctx_est, cfg["kv_type"], cfg["fit_margin_mib"],
                        vis_bytes if vision else 0, mtp)

    force_vision = want_vision == "on"
    if str(cfg["quant"]).lower() == "auto":
        pool = [m for m in by_size if m["quant"].upper().startswith("UD-")] or by_size
        if free is None:
            raise LauncherError("Chế độ CPU cần chỉ định quant cụ thể (không dùng được “auto”).")
        fitting = [m for m in pool if est(m, force_vision) <= free]
        chosen = fitting[0] if fitting else pool[-1]
        reason = "tự chọn: bản lớn nhất còn vừa VRAM" if fitting else "GPU quá nhỏ: lấy bản nhẹ nhất"
        smaller = [m for m in pool if m["size"] < chosen["size"] * 0.97]
    else:
        q = str(cfg["quant"]).lower()
        chosen = (next((m for m in by_size if m["quant"].lower() == q), None)
                  or next((m for m in by_size if q in m["path"].lower()), None))
        if not chosen:
            have = ", ".join(sorted(m["quant"] for m in models))
            raise LauncherError(f"Không thấy quant “{cfg['quant']}” trong repo. Các bản có sẵn: {have}")
        reason = "do bạn chỉ định"
        smaller = []

    if want_vision == "on":
        vision = True
    elif want_vision == "auto":
        vision = free is None or est(chosen, True) <= free
    else:
        vision = False

    def plan(m, vision, mtp, margin, note):
        return {"model": m, "vision": vision, "mtp": mtp, "ctx": ctx_explicit, "margin": margin,
                "est_mib": est(m, vision, mtp), "note": note}

    plans = [plan(chosen, vision, bool(cfg["mtp"]), cfg["fit_margin_mib"], reason)]
    if vision or cfg["mtp"]:
        plans.append(plan(chosen, False, False, max(cfg["fit_margin_mib"], 768),
                          "dự phòng: tắt hình ảnh / MTP"))
    if smaller:
        plans.append(plan(smaller[0], force_vision, False, max(cfg["fit_margin_mib"], 768),
                          "dự phòng: hạ một bậc quant"))
    return plans, prof


# --------------------------------------------------------------------------- #
# Tải model
# --------------------------------------------------------------------------- #
def fetch_model_file(repo, filename, size, dest_dir, token, progress, log_path):
    """Tải 1 file GGUF. Ưu tiên huggingface_hub (hf-xet, rất nhanh), dự phòng bằng curl."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / filename
    if target.exists() and target.stat().st_size == size:
        progress(size, size, 0.0)
        return target

    env = dict(os.environ, HF_HUB_DISABLE_PROGRESS_BARS="1", HF_HUB_DISABLE_TELEMETRY="1",
               HF_XET_CHUNK_CACHE_SIZE_BYTES="0")  # ghi thẳng ra file đích, không nhân đôi dung lượng
    if token:
        env["HF_TOKEN"] = token
    hf_code = ("import sys\nfrom huggingface_hub import hf_hub_download\n"
               "hf_hub_download(sys.argv[1], sys.argv[2], local_dir=sys.argv[3])\n")
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}?download=true"
    curl = ["curl", "-L", "--fail", "--silent", "--show-error", "--retry", "8", "--retry-delay", "3",
            "--retry-all-errors", "-C", "-", "-o", str(target) + ".part", url]
    if token:
        curl[1:1] = ["-H", f"Authorization: Bearer {token}"]
    methods = [("huggingface_hub", [sys.executable, "-c", hf_code, repo, filename, str(dest_dir)])]
    if shutil.which("curl"):
        methods.append(("curl", curl))

    errors = []
    for name, cmd in methods:
        with open(log_path, "ab") as log:
            log.write(f"\n=== {name}: {filename} ===\n".encode())
            log.flush()
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
            last_bytes, last_change, t_prev, speed = -1, time.time(), time.time(), 0.0
            last_rx = _net_rx_bytes()
            stalled = False
            while proc.poll() is None:
                time.sleep(0.5)
                if _CANCEL.is_set():
                    proc.kill()
                    proc.wait()
                    raise LauncherError("đã huỷ")
                done = min(_allocated_bytes(dest_dir), size)
                now, rx = time.time(), _net_rx_bytes()
                if done != last_bytes:
                    if last_bytes >= 0 and now > t_prev:
                        inst = (done - last_bytes) / (now - t_prev)
                        speed = inst if speed == 0 else 0.7 * speed + 0.3 * inst
                    last_bytes, last_change = done, now
                elif rx - last_rx > 256 * 1024:  # file chưa lớn lên nhưng mạng vẫn chạy -> chưa treo
                    last_change = now
                elif now - last_change > 240:
                    stalled = True
                    proc.kill()
                    break
                t_prev, last_rx = now, rx
                progress(done, size, speed)
            rc = proc.wait()
        part = Path(str(target) + ".part")
        if name == "curl" and rc == 0 and part.exists():
            part.replace(target)
        if rc == 0 and target.exists() and (size == 0 or target.stat().st_size == size):
            progress(size, size, 0.0)
            return target
        errors.append(f"{name}: " + ("đứng hình quá 4 phút" if stalled else f"mã thoát {rc}"))
    raise LauncherError(f"Tải {filename} thất bại ({'; '.join(errors)}). Xem log: {log_path}")


# --------------------------------------------------------------------------- #
# Cài llama.cpp
# --------------------------------------------------------------------------- #
def resolve_build(build):
    build = (build or PINNED_BUILD).strip()
    if build.lower() in ("stable", "latest"):
        st, body = _http(f"https://github.com/{LLAMA_REPO}/releases/latest/download/nightly-tag.txt", timeout=30)
        tag = body.decode("utf-8", "replace").strip() if st == 200 else ""
        return tag if re.fullmatch(r"b\d+", tag) else PINNED_BUILD
    return build


def list_assets(tag):
    st, body = _http(f"https://github.com/{LLAMA_REPO}/releases/expanded_assets/{tag}", timeout=30)
    names = set(re.findall(rf"releases/download/{re.escape(tag)}/([^\"<>\s]+)", body.decode("utf-8", "replace"))) if st == 200 else set()
    if not names and tag == PINNED_BUILD:  # GitHub chặn/đổi giao diện -> dùng tên đã biết
        names = {f"llama-{tag}-bin-ubuntu-x64.tar.gz",
                 f"llama-{tag}-bin-ubuntu-cuda-12.8-x64.tar.gz", f"cudart-llama-{tag}-bin-ubuntu-cuda-12.8-x64.tar.gz",
                 f"llama-{tag}-bin-ubuntu-cuda-13.4-x64.tar.gz", f"cudart-llama-{tag}-bin-ubuntu-cuda-13.4-x64.tar.gz"}
    return sorted(names)


def _lib_search_dirs():
    dirs = [d for d in os.environ.get("LD_LIBRARY_PATH", "").split(":") if d]
    dirs += ["/usr/lib64-nvidia", "/usr/local/nvidia/lib64", "/usr/lib/x86_64-linux-gnu"]
    for base in Path("/usr/local").glob("cuda*"):
        dirs += [str(base / "lib64"), str(base / "targets/x86_64-linux/lib")]
    for sp in {p for p in sys.path if p.endswith(("site-packages", "dist-packages"))}:
        dirs += [str(p) for p in Path(sp).glob("nvidia/*/lib")]
    seen, out = set(), []
    for d in dirs:
        if d not in seen and os.path.isdir(d):
            seen.add(d)
            out.append(d)
    return out


def _ldd_missing(lib, env):
    rc, out = _sh(["ldd", str(lib)], timeout=30, env=env)
    return sorted(set(re.findall(r"^\s*(\S+)\s+=>\s+not found", out, re.M)))


def _env_with_libs(dirs):
    env = dict(os.environ)
    cur = [d for d in env.get("LD_LIBRARY_PATH", "").split(":") if d]
    merged = []
    for d in list(dirs) + cur:
        if d and d not in merged:
            merged.append(d)
    env["LD_LIBRARY_PATH"] = ":".join(merged)
    return env


def _extract(archive, dest):
    dest = Path(dest)
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)
    rc, out = _sh(["tar", "-xzf", str(archive), "-C", str(dest)], timeout=900)
    if rc != 0:
        raise LauncherError(f"Giải nén {Path(archive).name} lỗi: {out[-300:]}")
    Path(archive).unlink(missing_ok=True)
    return dest


def install_llama(cfg, gpu, bin_root, step):
    """Trả về (đường dẫn llama-server, env, mô tả). Thử lần lượt: CUDA 12 -> CUDA 13 -> build nguồn."""
    tag = resolve_build(cfg["llama_build"])
    bin_root = Path(bin_root)
    marker = bin_root / "ready.json"
    if marker.exists():  # đã cài ở lần chạy trước trong cùng phiên
        try:
            info = json.loads(marker.read_text())
            if info["tag"] == tag and info["gpu"] == bool(gpu) and Path(info["server"]).exists():
                step(detail=f"{info['desc']} (đã cài sẵn)")
                return Path(info["server"]), _env_with_libs(info["lib_dirs"]), info["desc"]
        except Exception:  # noqa: BLE001
            pass

    assets = list_assets(tag)
    notes = []

    def done(server, lib_dirs, desc):
        marker.write_text(json.dumps({"tag": tag, "gpu": bool(gpu), "server": str(server),
                                      "lib_dirs": lib_dirs, "desc": desc}))
        return Path(server), _env_with_libs(lib_dirs), desc

    def fetch(name, label):
        url = f"https://github.com/{LLAMA_REPO}/releases/download/{tag}/{name}"
        archive = _download(url, bin_root / name,
                            lambda d, t: step(detail=f"{label} · {_fmt_size(d)}" + (f" / {_fmt_size(t)}" if t else ""),
                                              pct=(100 * d / t if t else None)))
        step(detail=f"giải nén {label}…", pct=None)
        return _extract(archive, bin_root / name.replace(".tar.gz", ""))

    if not gpu:
        name = f"llama-{tag}-bin-ubuntu-x64.tar.gz"
        root = fetch(name, f"llama.cpp {tag} (CPU)")
        server = next(root.rglob("llama-server"))
        rc, out = _sh([str(server), "--version"], env=_env_with_libs([str(server.parent)]))
        if rc != 0:
            raise LauncherError(f"llama-server (CPU) không chạy được: {out[-300:]}")
        return done(server, [str(server.parent)], f"llama.cpp {tag} · CPU")

    cuda = []
    for a in assets:
        m = re.fullmatch(rf"llama-{re.escape(tag)}-bin-ubuntu-cuda-(\d+)\.(\d+)-x64\.tar\.gz", a)
        if m:
            cuda.append((int(m.group(1)), int(m.group(2)), a))
    cuda.sort(key=lambda x: (x[0] != 12, -x[0], -x[1]))  # CUDA 12 trước: chạy được trên mọi driver mới

    glibc_too_old = False
    for major, minor, name in cuda:
        label = f"llama.cpp {tag} (CUDA {major}.{minor})"
        try:
            root = fetch(name, label)
            server = next(root.rglob("llama-server"))
            lib_dirs = [str(server.parent)]
            rc, out = _sh([str(server), "--version"], env=_env_with_libs(lib_dirs))
            if rc != 0:
                if "GLIBC" in out:
                    glibc_too_old = True
                    notes.append("bản dựng sẵn cần glibc mới hơn hệ điều hành này")
                    break
                notes.append(f"CUDA {major}.{minor}: {out.strip()[-120:]}")
                continue
            cuda_lib = server.parent / "libggml-cuda.so"
            missing = _ldd_missing(cuda_lib, _env_with_libs(lib_dirs)) if cuda_lib.exists() else []
            if missing:  # tìm thư viện CUDA có sẵn trong máy (hệ thống hoặc gói pip nvidia-*)
                for d in _lib_search_dirs():
                    if any((Path(d) / lib).exists() for lib in missing):
                        lib_dirs.append(d)
                missing = _ldd_missing(cuda_lib, _env_with_libs(lib_dirs))
            if any(("cudart" in x or "cublas" in x) for x in missing):
                rt = f"cudart-{name}"
                if rt in assets:
                    rt_root = fetch(rt, f"thư viện CUDA {major}.{minor} runtime")
                    lib_dirs += sorted({str(p.parent) for p in rt_root.rglob("libcudart.so*")})
            rc, out = _sh([str(server), "--list-devices"], timeout=120, env=_env_with_libs(lib_dirs))
            if rc == 0 and re.search(r"CUDA\d", out):
                return done(server, lib_dirs, f"llama.cpp {tag} · CUDA {major}.{minor}")
            notes.append(f"CUDA {major}.{minor}: không thấy GPU ({' '.join(out.split())[-160:]})")
        except (LauncherError, StopIteration) as e:
            notes.append(f"CUDA {major}.{minor}: {e}")

    # ---- Phương án cuối: build từ mã nguồn ----
    why = "; ".join(notes) or ("không tìm thấy bản dựng sẵn" if not glibc_too_old else "")
    step(detail=f"bản dựng sẵn không dùng được ({why}) → build từ mã nguồn, mất khoảng 10–20 phút…", pct=0)
    server = build_from_source(tag, bin_root, step)
    return done(server, [str(server.parent)], f"llama.cpp {tag} · tự build CUDA")


def build_from_source(tag, bin_root, step):
    src = Path(bin_root) / "src"
    log_path = Path(bin_root) / "build.log"
    need = [t for t in ("cmake", "git", "g++") if not shutil.which(t)]
    with open(log_path, "ab") as log:
        if need:
            subprocess.run("apt-get update -qq && apt-get install -y -qq build-essential cmake git libcurl4-openssl-dev",
                           shell=True, stdout=log, stderr=subprocess.STDOUT, timeout=900)
        shutil.rmtree(src, ignore_errors=True)
        rc = subprocess.run(["git", "clone", "--depth", "1", "--branch", tag,
                             f"https://github.com/{LLAMA_REPO}", str(src)],
                            stdout=log, stderr=subprocess.STDOUT, timeout=900).returncode
        if rc != 0:
            raise LauncherError(f"git clone llama.cpp ({tag}) thất bại. Xem {log_path}")
        rc = subprocess.run(["cmake", "-S", str(src), "-B", str(src / "build"), "-DGGML_CUDA=ON",
                             "-DBUILD_SHARED_LIBS=OFF", "-DCMAKE_BUILD_TYPE=Release",
                             "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF"],
                            stdout=log, stderr=subprocess.STDOUT, timeout=1800).returncode
        if rc != 0:
            raise LauncherError(f"cmake cấu hình thất bại. Xem {log_path}")
    t0 = time.time()
    with open(log_path, "ab") as log:
        proc = subprocess.Popen(["cmake", "--build", str(src / "build"), "--config", "Release",
                                 "-j", str(os.cpu_count() or 2), "--target", "llama-server"],
                                stdout=log, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            time.sleep(3)
            if _CANCEL.is_set():
                proc.kill()
                proc.wait()
                raise LauncherError("đã huỷ")
            pcts = re.findall(r"\[\s*(\d+)%\]", "\n".join(_tail(log_path, 40)))
            step(detail=f"đang build llama.cpp từ mã nguồn · {_fmt_dur(time.time() - t0)}",
                 pct=float(pcts[-1]) if pcts else None)
    server = src / "build" / "bin" / "llama-server"
    if proc.returncode != 0 or not server.exists():
        raise LauncherError(f"Build llama.cpp thất bại. Xem {log_path}")
    return server


def install_cloudflared(bin_root, step):
    dest = Path(bin_root) / "cloudflared"
    if not (dest.exists() and _sh([str(dest), "--version"])[0] == 0):
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
        _download(url, dest, lambda d, t: step(pct=(100 * d / t if t else None)))
        dest.chmod(0o755)
    rc, out = _sh([str(dest), "--version"])
    if rc != 0:
        raise LauncherError(f"cloudflared không chạy được: {out[-200:]}")
    m = re.search(r"version\s+(\S+)", out)
    return dest, (m.group(1) if m else "?")


# --------------------------------------------------------------------------- #
# Tiến trình nền: llama-server và cloudflared
# --------------------------------------------------------------------------- #
class Proc:
    def __init__(self, name, cmd, log_path, env=None):
        self.name, self.cmd, self.log_path, self.env = name, cmd, Path(log_path), env
        self.p = None

    def start(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if self.log_path.exists():
            self.log_path.replace(self.log_path.with_suffix(self.log_path.suffix + ".prev"))
        log = open(self.log_path, "ab")
        self.p = subprocess.Popen(self.cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                  env=self.env, start_new_session=True)
        log.close()
        return self

    def alive(self):
        return self.p is not None and self.p.poll() is None

    def stop(self, grace=8):
        if not self.p:
            return
        if self.p.poll() is None:
            try:
                os.killpg(self.p.pid, signal.SIGTERM)
            except OSError:
                pass
            try:
                self.p.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.p.pid, signal.SIGKILL)
                except OSError:
                    pass
                try:
                    self.p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass

    def log_text(self):
        try:
            return self.log_path.read_text(errors="replace")
        except OSError:
            return ""


def kill_leftovers(run_dir):
    """Dọn tiến trình của lần chạy trước (chạy lại ô nhiều lần vẫn sạch)."""
    pid_file = Path(run_dir) / "pids.json"
    try:
        pids = json.loads(pid_file.read_text())
    except Exception:  # noqa: BLE001
        pids = []
    for pid in pids:
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode("utf-8", "replace")
        except OSError:
            continue
        if any(k in cmdline for k in ("llama-server", "cloudflared", "localhost.run")):
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.kill(pid, sig)
                    time.sleep(1.0)
                except OSError:
                    break
    pid_file.unlink(missing_ok=True)


def server_command(server, plan, model_path, mmproj_path, cfg, port, gpu):
    cmd = [str(server), "-m", str(model_path), "--alias", cfg["alias"],
           "--host", "127.0.0.1", "--port", str(port),
           "--parallel", "1", "--metrics", "--cache-ram", "2048",
           "-lv", "4", "--log-colors", "off"]
    if gpu:
        cmd += ["-fa", "on", "-ctk", cfg["kv_type"], "-ctv", cfg["kv_type"],
                "--fit", "on", "--fit-target", str(int(plan["margin"]))]
    if plan["ctx"]:
        cmd += ["-c", str(int(plan["ctx"]))]
    else:  # llama.cpp tự lấy context lớn nhất còn vừa VRAM, không thấp hơn min_ctx
        cmd += ["--fit-ctx", str(int(cfg["min_ctx"]))]
    if plan["vision"] and mmproj_path:
        cmd += ["--mmproj", str(mmproj_path)]
    if plan["mtp"]:
        cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]
    if cfg["reasoning"] == "off":
        cmd += ["--reasoning", "off"] + SAMPLING["instruct"]
    else:
        cmd += ["--reasoning-effort", cfg["reasoning"]] + SAMPLING["thinking"]
    if cfg["password"]:
        cmd += ["--api-key", cfg["password"]]
    return cmd


def _fail_reason(log_text):
    pats = r"out of memory|failed to|error|cannot|unable|exception|unknown (?:model )?architecture|not found|invalid"
    lines = [ln for ln in log_text.splitlines() if re.search(pats, ln, re.I) and "API key" not in ln]
    return lines[-6:]


def wait_ready(proc, port, step, gpu, base_used, est_weights_mib, timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not proc.alive():
            return False
        st, _ = _http(f"http://127.0.0.1:{port}/health", timeout=4)
        if st == 200:
            return True
        detail, pct = f"đang nạp model… {_fmt_dur(time.time() - t0)}", None
        if gpu:
            used, _ = gpu_used_mib()
            if used is not None:
                loaded = max(0, used - base_used)
                pct = min(99.0, 100.0 * loaded / max(1, est_weights_mib))
                detail = f"đang nạp vào VRAM: {loaded / 1024:.1f} GB · {_fmt_dur(time.time() - t0)}"
        step(detail=detail, pct=pct)
        time.sleep(1.5)
    return False


def self_test(port, password):
    body = json.dumps({"messages": [{"role": "user", "content": "Chào bạn! Hãy trả lời bằng đúng một câu ngắn."}],
                       "max_tokens": 48, "stream": False}).encode()
    hdr = {"Content-Type": "application/json"}
    if password:
        hdr["Authorization"] = f"Bearer {password}"
    st, raw = _http(f"http://127.0.0.1:{port}/v1/chat/completions", timeout=300, headers=hdr, data=body)
    if st != 200:
        return None, f"HTTP {st}: {raw[:200].decode('utf-8', 'replace')}"
    try:
        d = json.loads(raw)
        tps = float((d.get("timings") or {}).get("predicted_per_second") or 0)
        ntok = int((d.get("usage") or {}).get("completion_tokens") or 0)
        return (tps if tps > 0 else None), (None if ntok > 0 else "model không sinh ra token nào")
    except Exception as e:  # noqa: BLE001
        return None, str(e)


# Tên quick tunnel luôn có dạng tu-tu-tu-tu; loại "api.trycloudflare.com" hay xuất hiện trong dòng báo lỗi
_CF_RE = re.compile(r"https://(?!api\.)[a-z0-9]+(?:-[a-z0-9]+)+\.trycloudflare\.com")
_LHR_RE = re.compile(r"https://[a-z0-9]+\.lhr\.life")


def _start_tunnel(provider, cmd, log_path, url_re, wait_s, step, label):
    """Chạy 1 tiến trình tunnel và chờ nó in ra URL. Trả về (Proc, url, lỗi)."""
    proc = Proc(provider, cmd, log_path).start()
    proc.provider, proc.url_re = provider, url_re
    t0 = time.time()
    while time.time() - t0 < wait_s:
        found = url_re.findall(proc.log_text())
        if found:
            return proc, found[-1], ""
        if not proc.alive():
            break
        step(detail=label)
        time.sleep(1.0)
    errs = [ln.strip() for ln in proc.log_text().splitlines()
            if re.search(r"\bERR\b|error|failed|429|denied|refused|timeout", ln, re.I)]
    proc.stop(grace=3)
    return None, None, (errs[-1] if errs else "hết thời gian chờ")[-180:]


def open_tunnel(cloudflared, port, log_dir, step):
    """Mở link công khai. Ưu tiên Cloudflare quick tunnel; nếu Cloudflare từ chối cấp link
    (hay gặp khi IP dùng chung bị giới hạn) thì chuyển sang localhost.run qua SSH.
    Trả về (Proc, url, danh sách sự cố đã gặp)."""
    problems = []
    target = f"http://127.0.0.1:{port}"
    for attempt in range(3):
        proc, url, err = _start_tunnel(
            "Cloudflare Tunnel", [str(cloudflared), "tunnel", "--no-autoupdate", "--url", target],
            Path(log_dir) / "cloudflared.log", _CF_RE, 45, step,
            f"xin địa chỉ từ Cloudflare… (lần {attempt + 1}/3)")
        if url:
            return proc, url, problems
        problems.append(f"Cloudflare: {err}")
        if attempt < 2:
            step(detail=f"Cloudflare chưa cấp link ({err[:70]}) — thử lại sau {6 * (attempt + 1)} giây")
            time.sleep(6 * (attempt + 1))
    if shutil.which("ssh"):
        ssh = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
               "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3", "-o", "ExitOnForwardFailure=yes",
               "-o", "LogLevel=ERROR", "-T", "-R", f"80:127.0.0.1:{port}", "nokey@localhost.run",
               "--", "--output", "json"]
        for attempt in range(2):
            proc, url, err = _start_tunnel(
                "localhost.run", ssh, Path(log_dir) / "localhost_run.log", _LHR_RE, 40, step,
                "Cloudflare không cấp link — chuyển sang đường hầm dự phòng localhost.run…")
            if url:
                return proc, url, problems
            problems.append(f"localhost.run: {err}")
            time.sleep(4)
    raise LauncherError("Không mở được đường hầm công khai nào (" + " | ".join(problems[-3:]) + ").")


def colab_private_url(port):
    """Phương án chót: link proxy của Colab — chỉ mở được trên trình duyệt đang đăng nhập Colab này."""
    try:
        from google.colab.output import eval_js  # noqa: PLC0415
        url = eval_js(f"google.colab.kernel.proxyPort({int(port)})")
        return str(url).rstrip("/") if url else None
    except Exception:  # noqa: BLE001
        return None


def verify_public(url, timeout=45):
    """Gọi thử link công khai và chắc chắn bên kia đúng là llama-server của mình."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        st, body = _http(url + "/health", timeout=8)
        if st == 200 and b'"status"' in body and b"ok" in body:
            return True
        time.sleep(3)
    return False


def read_metrics(port, password):
    hdr = {"Authorization": f"Bearer {password}"} if password else {}
    st, raw = _http(f"http://127.0.0.1:{port}/metrics", timeout=5, headers=hdr)
    if st != 200:
        return None
    vals = {}
    for m in re.finditer(r"^llamacpp:(\w+)\s+([-+0-9.eE]+)\s*$", raw.decode("utf-8", "replace"), re.M):
        try:
            vals[m.group(1)] = float(m.group(2))
        except ValueError:
            pass
    return vals


def make_qr(url):
    """QR để mở trên điện thoại. Cần gói `segno` (tự cài, ~100 KB); lỗi thì bỏ qua."""
    try:
        try:
            import segno  # noqa: PLC0415
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "segno"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
            import segno  # noqa: PLC0415
        return segno.make(url, error="m").svg_inline(scale=3, border=1, dark="#0f172a", light="#ffffff")
    except Exception:  # noqa: BLE001
        return ""


# --------------------------------------------------------------------------- #
# Điều phối chính
# --------------------------------------------------------------------------- #
def run(**overrides):
    cfg = dict(DEFAULTS)
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    env_over = os.environ.get("QWEN_LAUNCHER_OVERRIDES")
    if env_over:
        cfg.update(json.loads(env_over))
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Tham số không hợp lệ: {sorted(unknown)}")
    cfg["reasoning"] = str(cfg["reasoning"]).strip().lower()
    cfg["vision"] = str(cfg["vision"]).strip().lower()
    cfg["quant"] = str(cfg["quant"]).strip().split(" ")[0] or "auto"
    cfg["context"] = str(cfg["context"]).strip().split(" ")[0].lower() or "auto"
    cfg["password"] = str(cfg["password"] or "").strip()
    cfg["hf_token"] = str(cfg["hf_token"] or "").strip() or os.environ.get("HF_TOKEN", "")
    if cfg["reasoning"] not in ("low", "medium", "xhigh", "off"):
        raise ValueError("reasoning phải là low / medium / xhigh / off")
    if cfg["vision"] not in ("auto", "on", "off"):
        raise ValueError("vision phải là auto / on / off")
    if cfg["context"] != "auto" and not cfg["context"].isdigit():
        raise ValueError("context phải là “auto” hoặc một số nguyên")
    if cfg["kv_type"] not in KV_BITS:
        raise ValueError(f"kv_type phải thuộc {sorted(KV_BITS)}")
    model_name = cfg["repo"].split("/")[-1].replace("-GGUF", "")
    cfg["alias"] = model_name.lower()

    work = Path(cfg["workdir"] or ("/content/qwen38" if os.path.isdir("/content") else Path.home() / ".cache" / "qwen38"))
    bin_dir, model_dir, log_dir, run_dir = work / "bin", work / "models", work / "logs", work / "run"
    for d in (bin_dir, model_dir, log_dir, run_dir):
        d.mkdir(parents=True, exist_ok=True)

    board = Board(f"{model_name} · Colab one-click", f"launcher v{__version__} · llama.cpp + Cloudflare Tunnel")
    for key, label in [("sys", "Kiểm tra máy"), ("pick", "Chọn bản model"), ("llama", "Cài llama.cpp"),
                       ("model", "Tải model"), ("cf", "Cài Cloudflare Tunnel"),
                       ("load", "Nạp model & chạy thử"), ("link", "Mở link công khai")]:
        board.add(key, label)
    procs = {}
    t_start = time.time()

    _CANCEL.clear()

    def cleanup():
        _CANCEL.set()
        for p in list(procs.values()):
            p.stop()
        (run_dir / "pids.json").unlink(missing_ok=True)

    def save_pids():
        (run_dir / "pids.json").write_text(json.dumps([p.p.pid for p in procs.values() if p.alive()]))

    atexit.register(cleanup)
    try:
        return _run(cfg, board, procs, save_pids, t_start, work, bin_dir, model_dir, log_dir, run_dir, model_name)
    except KeyboardInterrupt:
        cleanup()
        for k in board.order:
            if board.steps[k]["state"] == "running":
                board.set(k, "warn", "đã dừng")
        if board.ready:
            board.set("link", "warn", "đã tắt")
        board.ready = None
        board.live = ""
        board.box("info", "⏹ Đã dừng. Model và tunnel đã tắt; link cũ không còn dùng được. "
                          "Chạy lại ô này để bật lại (model đã tải sẽ không phải tải lại trong cùng phiên).", replace=True)
        board.render(force=True)
        return None
    except LauncherError as e:
        cleanup()
        for k in board.order:
            if board.steps[k]["state"] == "running":
                board.set(k, "fail")
        board.ready = None
        board.live = ""
        board.box("err", f"<b>Không hoàn tất được:</b> {html.escape(str(e))}")
        board.render(force=True)
        return None
    except BaseException:
        cleanup()
        raise


def _run(cfg, board, procs, save_pids, t_start, work, bin_dir, model_dir, log_dir, run_dir, model_name):
    # ---- 1. Kiểm tra máy -------------------------------------------------- #
    board.set("sys", "running")
    board.render()
    kill_leftovers(run_dir)
    gpu = detect_gpu()
    ram, disk_free = ram_total_mib(), shutil.disk_usage(work).free
    if not gpu and not cfg["allow_cpu"]:
        board.set("sys", "fail", "không thấy GPU NVIDIA")
        raise LauncherError("Phiên Colab này chưa bật GPU. Vào menu Runtime → Change runtime type → chọn "
                            "T4 GPU → Save, rồi chạy lại ô này.")
    gpu_txt = f"{gpu['name']} · VRAM trống {gpu['free_mib'] / 1024:.1f}/{gpu['total_mib'] / 1024:.1f} GB" if gpu else "CPU (không GPU)"
    board.set("sys", "done", f"{gpu_txt} · RAM {ram / 1024:.1f} GB · đĩa trống {_fmt_size(disk_free)}")

    # ---- 2. Chọn model ---------------------------------------------------- #
    board.set("pick", "running", "đọc danh sách file trên Hugging Face…")
    board.render()
    models, mmproj = split_files(hf_list(cfg["repo"], cfg["hf_token"]))
    if not models:
        raise LauncherError(f"Repo {cfg['repo']} không có file GGUF nào dùng được.")
    plans, prof = make_plans(cfg, models, mmproj, gpu, disk_free)
    plan = plans[0]

    def plan_text(p):
        bits = [f"{p['model']['quant']} · {_fmt_size(p['model']['size'])}", p["note"]]
        if gpu:
            bits.append(f"ước cần {p['est_mib'] / 1024:.1f} GB VRAM")
        bits.append("hình ảnh: " + ("bật" if p["vision"] else "tắt"))
        if p["mtp"]:
            bits.append("MTP: bật")
        return " · ".join(bits)

    board.set("pick", "done", plan_text(plan))
    board.render()

    # ---- 3-5. Cài đặt + tải song song -------------------------------------- #
    results, errors = {}, {}

    def worker(key, fn):
        try:
            results[key] = fn()
        except BaseException as e:  # noqa: BLE001
            errors[key] = e

    def step_fn(key):
        return lambda **kw: board.set(key, **kw)

    def do_llama():
        board.set("llama", "running", "tìm bản dựng sẵn…")
        server, env, desc = install_llama(cfg, gpu, bin_dir / "llama", step_fn("llama"))
        board.set("llama", "done", desc, pct=None)
        return server, env, desc

    def do_cf():
        board.set("cf", "running")
        path, ver = install_cloudflared(bin_dir, step_fn("cf"))
        board.set("cf", "done", f"cloudflared {ver}", pct=None)
        return path

    def do_model(p):
        m = p["model"]
        t0 = time.time()

        def prog(label):
            def cb(done, total, speed):
                eta = f" · còn ~{_fmt_dur((total - done) / speed)}" if speed > MiB and done < total else ""
                spd = f" · {speed / MiB:.0f} MB/s" if speed > 0 else ""
                board.set("model", "running", f"{label}: {_fmt_size(done)} / {_fmt_size(total)}{spd}{eta}",
                          pct=100 * done / max(1, total))
            return cb

        mm_path = None
        if p["vision"] and mmproj:
            mm_path = fetch_model_file(cfg["repo"], mmproj["path"], mmproj["size"], model_dir / "mmproj",
                                       cfg["hf_token"], prog("bộ mã hoá hình ảnh"), log_dir / "download.log")
        path = fetch_model_file(cfg["repo"], m["path"], m["size"], model_dir / m["quant"],
                                cfg["hf_token"], prog(m["quant"]), log_dir / "download.log")
        dt = max(0.001, time.time() - t0)
        board.set("model", "done", f"{m['path']} · {_fmt_size(m['size'])} · xong trong {_fmt_dur(dt)}", pct=None)
        return path, mm_path

    def run_parallel(jobs):
        threads = [threading.Thread(target=worker, args=(k, fn), daemon=True) for k, fn in jobs.items()]
        for t in threads:
            t.start()
        while any(t.is_alive() for t in threads):
            board.render()
            time.sleep(0.35)
        board.render(force=True)
        for k in jobs:
            if k in errors:
                board.set(k, "fail")
                err = errors.pop(k)
                if isinstance(err, LauncherError):
                    raise err
                raise LauncherError(f"{board.steps[k]['label']}: {err}")

    board.set("model", "running", "bắt đầu tải…", pct=0)
    run_parallel({"llama": do_llama, "cf": do_cf, "model": lambda: do_model(plan)})
    server, env, llama_desc = results["llama"]
    cloudflared = results["cf"]
    model_path, mmproj_path = results["model"]

    # ---- 6. Nạp model (có phương án dự phòng) ------------------------------ #
    port = _free_port(int(cfg["port"]))
    info = None
    for i, plan in enumerate(plans):
        if i > 0:
            board.set("pick", "warn", plan_text(plan))
            board.render()
            if plan["model"]["path"] != Path(model_path).name:  # hạ quant: xoá bản cũ, tải bản nhỏ hơn
                shutil.rmtree(Path(model_path).parent, ignore_errors=True)
                board.set("model", "running", "tải bản nhỏ hơn…", pct=0)
                run_parallel({"model": lambda p=plan: do_model(p)})
                model_path, mmproj_path = results["model"]
        board.set("load", "running", "khởi động llama-server…", pct=0)
        board.render()
        base_used = (gpu_used_mib()[0] or 0) if gpu else 0
        cmd = server_command(server, plan, model_path, mmproj_path if plan["vision"] else None, cfg, port, gpu)
        (log_dir / "server.cmd.txt").write_text(" ".join(cmd) + "\n")
        sp = Proc("llama-server", cmd, log_dir / "server.log", env).start()
        procs["server"] = sp
        save_pids()
        est_w = plan["model"]["size"] / MiB - prof["emb_mib"]

        def load_step(**kw):
            board.set("load", **kw)
            board.render()

        ok = wait_ready(sp, port, load_step, gpu, base_used, est_w)
        log_text = sp.log_text()
        off = re.findall(r"offloaded (\d+)/(\d+) layers to GPU", log_text)
        on_gpu, n_layers = (int(off[-1][0]), int(off[-1][1])) if off else (None, None)
        # --fit báo "filling dense layers…/overflow…" khi phải đẩy bớt trọng số xuống RAM
        overflow = re.search(r"filling dense(?:-only)? layers|overflow to next device/system memory"
                             r"|failed to fit params", log_text)
        spilled = bool(gpu and ok and (overflow or (on_gpu is not None and on_gpu < n_layers)))
        has_next = i + 1 < len(plans)
        if ok and not (spilled and has_next):
            info = {"plan": plan, "on_gpu": on_gpu, "n_layers": n_layers, "spilled": spilled}
            break
        sp.stop()
        if ok and spilled:
            msg = f"{plan['model']['quant']} không nằm trọn trong VRAM (một phần phải chạy bằng CPU, sẽ rất chậm)"
        else:
            reason = _fail_reason(log_text)
            msg = f"{plan['model']['quant']} không khởi động được" + (f": {reason[-1][-160:]}" if reason else "")
        if not has_next:
            board.set("load", "fail", msg)
            tail = "\n".join(_tail(log_dir / "server.log", 30))
            board.box("err", f"<b>llama-server không chạy được.</b> {html.escape(msg)}"
                             f"<pre>{html.escape(tail)}</pre>Log đầy đủ: <code>{html.escape(str(log_dir / 'server.log'))}</code>")
            raise LauncherError("Đã thử hết các phương án dự phòng mà model vẫn không nạp được.")
        board.box("warn", f"⚠ {html.escape(msg)} → chuyển sang phương án dự phòng.")
        board.render()

    plan = info["plan"]
    hdr = {"Authorization": f"Bearer {cfg['password']}"} if cfg["password"] else {}
    st, raw = _http(f"http://127.0.0.1:{port}/props", timeout=10, headers=hdr)
    props = json.loads(raw) if st == 200 else {}
    n_ctx = (props.get("default_generation_settings") or {}).get("n_ctx")
    vision_on = bool((props.get("modalities") or {}).get("vision"))

    tps = None
    if cfg["self_test"]:
        board.set("load", detail="chạy thử một câu để đo tốc độ…", pct=None)
        board.render()
        tps, test_err = self_test(port, cfg["password"])
        if test_err:
            board.box("warn", f"⚠ Chạy thử chưa đạt: {html.escape(test_err)}. Link vẫn được mở, bạn thử chat xem sao.")
    layers_txt = f"{info['on_gpu']}/{info['n_layers']} lớp trên GPU" if (gpu and info["on_gpu"] is not None) else ""
    load_bits = [b for b in [f"context {n_ctx:,} token" if n_ctx else "", layers_txt,
                             f"{tps:.1f} token/giây" if tps else ""] if b]
    board.set("load", "warn" if info["spilled"] else "done", " · ".join(load_bits), pct=None)
    if info["spilled"]:
        board.box("warn", "⚠ Một phần model đang chạy trên CPU nên sẽ chậm. Chọn quant nhỏ hơn "
                          "(ví dụ <code>UD-Q3_K_XL</code>) hoặc tắt hình ảnh / MTP rồi chạy lại.")

    # ---- 7. Tunnel ---------------------------------------------------------- #
    board.set("link", "running", "mở Cloudflare Tunnel…")
    board.render()

    def link_step(**kw):
        board.set("link", **kw)
        board.render()

    tunnel_info = {"provider": "", "private": False}

    def bring_up_tunnel():
        procs.pop("tunnel", None)
        link_step(state="running", detail="mở đường hầm…")
        try:
            tp, url, problems = open_tunnel(cloudflared, port, log_dir, link_step)
        except LauncherError as e:
            url = colab_private_url(port)
            if not url:
                raise
            tunnel_info.update(provider="proxy riêng của Colab", private=True)
            link_step(state="warn", detail=f"{url} — {e} Đang dùng link riêng của Colab: chỉ mở được trên "
                                           f"trình duyệt đang đăng nhập Colab này.")
            return url
        procs["tunnel"] = tp
        save_pids()
        tunnel_info.update(provider=tp.provider, private=False)
        link_step(detail=f"kiểm tra {url} …")
        verified = verify_public(url)
        note = "" if verified else " (chưa kiểm tra được từ Colab — đợi 10–20 giây rồi mở)"
        if tp.provider != "Cloudflare Tunnel":
            note += f" · qua {tp.provider} vì Cloudflare đang từ chối cấp link ({problems[-1][:90] if problems else ''})"
        link_step(state="done" if (verified and tp.provider == "Cloudflare Tunnel") else "warn", detail=url + note)
        return url

    url = bring_up_tunnel()
    setup_s = time.time() - t_start

    def facts():
        m = plan["model"]
        think = {"off": "tắt (trả lời ngay)", "low": "ngắn", "medium": "vừa", "xhigh": "sâu nhất"}[cfg["reasoning"]]
        rows = [("Model", f"{model_name} · {m['quant']} ({_fmt_size(m['size'])})"),
                ("GPU", " · ".join(x for x in (gpu["name"], layers_txt) if x) if gpu else "không có (chạy CPU)"),
                ("Context", f"{n_ctx:,} token" if n_ctx else "?"),
                ("Suy luận", think),
                ("Hình ảnh", "bật — đính kèm ảnh được" if vision_on else
                 ("tắt — đặt VISION = on để bật (sẽ dùng quant nhỏ hơn nếu thiếu VRAM)"
                  if (mmproj and cfg["vision"] == "auto") else "tắt")),
                ("Tốc độ đo thử", f"{tps:.1f} token/giây" if tps else "—"),
                ("Thời gian cài đặt", _fmt_dur(setup_s)),
                ("Engine", llama_desc + (" · MTP" if plan["mtp"] else "")),
                ("Đường hầm", tunnel_info["provider"])]
        return rows

    qr = make_qr(url) if cfg["qr"] and board.rich else ""
    board.ready = {"url": url, "facts": facts(), "qr": qr, "password": cfg["password"]}
    board.render(force=True)
    result = {"url": url, "api_base": url + "/v1", "port": port, "quant": plan["model"]["quant"],
              "n_ctx": n_ctx, "vision": vision_on, "tokens_per_second": tps, "setup_seconds": setup_s,
              "server_log": str(log_dir / "server.log")}
    (run_dir / "state.json").write_text(json.dumps(result, indent=2))
    if not cfg["block"]:
        return result

    # ---- 8. Canh gác + thống kê trực tiếp ----------------------------------- #
    t_ready = time.time()
    restarts, last_stats, prev = [], 0.0, None
    recent_tps = None
    while True:
        time.sleep(2)
        now = time.time()
        if cfg["exit_after_s"] and now - t_ready > cfg["exit_after_s"]:
            for p in list(procs.values()):
                p.stop()
            board.ready = None
            board.set_live("")
            board.box("info", "⏹ Đã hết thời gian chạy đặt trước; model và đường hầm đã tắt.", replace=True)
            board.render(force=True)
            return result
        if not procs["server"].alive():
            restarts = [t for t in restarts if now - t < 900] + [now]
            if len(restarts) > 3:
                tail = "\n".join(_tail(log_dir / "server.log", 30))
                board.box("err", f"<b>llama-server chết liên tục.</b><pre>{html.escape(tail)}</pre>")
                raise LauncherError("Model bị sập quá 3 lần trong 15 phút (thường do hết VRAM/RAM). "
                                    "Hãy giảm context hoặc chọn quant nhỏ hơn rồi chạy lại.")
            board.set_live("model vừa bị sập — đang tự khởi động lại, link giữ nguyên…")
            board.render(force=True)
            procs["server"] = Proc("llama-server", procs["server"].cmd, log_dir / "server.log", env).start()
            save_pids()
            wait_ready(procs["server"], port, lambda **kw: None, None, 0, 1)
            prev = None
        tun = procs.get("tunnel")
        new_url = None
        if tun is not None and not tun.alive():
            board.set_live("đường hầm bị rớt — đang xin link mới…")
            board.render(force=True)
            new_url = bring_up_tunnel()
        elif tun is not None:  # nhà cung cấp đổi tên miền giữa chừng (localhost.run hay làm vậy)
            seen = tun.url_re.findall(tun.log_text())
            if seen and seen[-1] != url:
                new_url = seen[-1]
        if new_url and new_url != url:
            url = new_url
            result.update(url=url, api_base=url + "/v1")
            board.ready.update(url=url, facts=facts(), qr=make_qr(url) if cfg["qr"] and board.rich else "")
            board.touch()
            (run_dir / "state.json").write_text(json.dumps(result, indent=2))
        if now - last_stats >= 10:
            last_stats = now
            met = read_metrics(port, cfg["password"]) or {}
            if prev and met:
                d_tok = met.get("tokens_predicted_total", 0) - prev.get("tokens_predicted_total", 0)
                d_sec = met.get("tokens_predicted_seconds_total", 0) - prev.get("tokens_predicted_seconds_total", 0)
                if d_tok > 0 and d_sec > 0:
                    recent_tps = d_tok / d_sec
            prev = met or prev
            parts = [f"đang chạy {_fmt_dur(now - t_ready)}"]
            if met:
                parts.append(f"đã sinh {int(met.get('tokens_predicted_total', 0)):,} token")
                if recent_tps:
                    parts.append(f"{recent_tps:.1f} token/giây")
                if met.get("requests_processing", 0) > 0:
                    parts.append("đang trả lời…")
            if gpu:
                used, util = gpu_used_mib()
                if used is not None:
                    parts.append(f"VRAM {used / 1024:.1f}/{gpu['total_mib'] / 1024:.1f} GB · GPU {util}%")
            board.set_live(" · ".join(parts))
        board.render()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Qwen3.8-27B one-click launcher")
    ap.add_argument("--quant", default=DEFAULTS["quant"])
    ap.add_argument("--reasoning", default=DEFAULTS["reasoning"])
    ap.add_argument("--vision", default=DEFAULTS["vision"])
    ap.add_argument("--context", default=DEFAULTS["context"])
    ap.add_argument("--password", default="")
    ap.add_argument("--mtp", action="store_true")
    ap.add_argument("--llama-build", default=DEFAULTS["llama_build"])
    ap.add_argument("--repo", default=DEFAULTS["repo"])
    ap.add_argument("--allow-cpu", action="store_true")
    a = ap.parse_args()
    sys.exit(0 if run(quant=a.quant, reasoning=a.reasoning, vision=a.vision, context=a.context,
                      password=a.password, mtp=a.mtp, llama_build=a.llama_build, repo=a.repo,
                      allow_cpu=a.allow_cpu) else 1)
