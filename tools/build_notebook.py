#!/usr/bin/env python3
"""Sinh notebook Colab từ launcher.py (nguồn duy nhất của toàn bộ logic).

    python tools/build_notebook.py

Notebook tự chứa toàn bộ mã: mở là chạy được, không tải thêm script nào từ repo.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "Qwen3_8_27B_Colab.ipynb"
REPO = "tiendatmeow-source/qwen38-colab"

INTRO = f"""# 🧠 Qwen3.8-27B trên Google Colab — một lần bấm là có chatbox

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{NOTEBOOK.name})

**Cách dùng:** bấm ▶ ở ô bên dưới rồi đợi. Khi mọi thứ sẵn sàng, ô sẽ hiện nút **💬 Mở chatbox** kèm link công khai (Cloudflare Tunnel) và mã QR để mở trên điện thoại.

Notebook đã đặt sẵn GPU **T4**. Nếu ô báo chưa có GPU: *Runtime → Change runtime type → T4 GPU*.

| Việc notebook tự làm | Chi tiết |
|---|---|
| Chọn bản model | Đo VRAM trống rồi lấy bản GGUF **lớn nhất còn nằm trọn trên GPU**. T4 → `UD-IQ4_XS` (13,4 GB); L4 → `UD-Q6_K`; A100 → `UD-Q8_K_XL` |
| Cài engine | llama.cpp bản dựng sẵn CUDA (vài chục giây). Không dùng được thì tự build từ mã nguồn |
| Tải model | Chạy song song với bước cài, có thanh tiến độ, tự chuyển cách tải nếu bị treo |
| Tối ưu VRAM | Tự lấy context lớn nhất còn vừa GPU, nén KV cache `q8_0`, flash attention |
| Tự kiểm tra | Chạy thử một câu, đo token/giây trước khi đưa link |
| Mở link | Cloudflare Tunnel; nếu Cloudflare từ chối cấp link thì tự chuyển sang localhost.run |
| Canh gác | Model hoặc đường hầm chết thì tự bật lại; thống kê cập nhật 10 giây/lần |
"""

FORM = '''#@title 🚀 Chạy Qwen3.8-27B — bấm ▶ một lần là xong { display-mode: "form" }
#@markdown ### Tuỳ chọn — để nguyên mặc định là chạy tốt trên T4
#@markdown **Chất lượng model.** `auto` = bản lớn nhất còn nằm trọn trong VRAM (T4 → `UD-IQ4_XS`).
QUANT = "auto"  #@param ["auto", "UD-IQ4_XS", "UD-Q3_K_XL", "UD-IQ3_S", "UD-Q2_K_XL", "UD-Q4_K_XL", "UD-Q5_K_XL", "UD-Q6_K_XL", "UD-Q8_K_XL"] {allow-input: true}
#@markdown **Mức suy luận.** Càng cao càng kỹ nhưng chờ càng lâu; `off` = trả lời ngay, không suy nghĩ.
REASONING = "medium"  #@param ["off", "low", "medium", "xhigh"]
#@markdown **Đọc hình ảnh.** `auto` = chỉ bật khi còn dư VRAM; `on` = luôn bật (trên T4 sẽ dùng `UD-Q3_K_XL`).
VISION = "auto"  #@param ["auto", "on", "off"]
#@markdown **Context (độ dài hội thoại nhớ được).** `auto` = lớn nhất còn vừa VRAM.
CONTEXT = "auto"  #@param ["auto", "8192", "16384", "32768", "65536"] {allow-input: true}
#@markdown **Mật khẩu** (không bắt buộc). Để trống thì ai có link cũng chat được.
PASSWORD = ""  #@param {type: "string"}
#@markdown **Tăng tốc MTP** (thử nghiệm): đoán trước nhiều token mỗi bước, tốn thêm VRAM nên trên T4 sẽ dùng quant nhỏ hơn.
MTP = False  #@param {type: "boolean"}
#@markdown ---
#@markdown Nâng cao: bản llama.cpp (`b12110` là bản đã kiểm thử, `stable` là bản ổn định mới nhất) và token Hugging Face nếu cần.
LLAMA_CPP_BUILD = "b12110"  #@param ["b12110", "stable"] {allow-input: true}
HF_TOKEN = ""  #@param {type: "string"}

'''

CALL = '''

_ = run(quant=QUANT, reasoning=REASONING, vision=VISION, context=CONTEXT, password=PASSWORD,
        mtp=MTP, llama_build=LLAMA_CPP_BUILD, hf_token=HF_TOKEN)
'''

OUTRO = """## Cần biết

- **Link chỉ sống khi ô ở trên còn chạy.** Tắt tab, hết phiên Colab hoặc bấm ■ là link mất; chạy lại sẽ có link mới.
- **Tốc độ.** T4 là GPU đời cũ nên model 27B chạy chậm hơn nhiều so với card 24 GB. Tốc độ thật được đo và hiện ngay trên thẻ kết quả. Muốn nhanh hơn: đặt `REASONING = low` hoặc `off`.
- **Gửi ảnh.** Đặt `VISION = on`. Trên T4 notebook sẽ dùng `UD-Q3_K_XL` để dành VRAM cho bộ mã hoá hình ảnh.
- **Colab miễn phí** có thể ngắt các phiên chủ yếu dùng qua giao diện web bên ngoài. Nếu bị ngắt thường xuyên, hãy dùng gói có trả phí.
- **Bảo mật.** Ai có link đều dùng được model của bạn. Đặt `PASSWORD` nếu định chia sẻ link.

## Gọi bằng API (tương thích OpenAI)

```python
from openai import OpenAI

client = OpenAI(base_url="https://<link-của-bạn>/v1", api_key="<PASSWORD hoặc chữ bất kỳ>")
stream = client.chat.completions.create(
    model="qwen3.8-27b",
    messages=[{"role": "user", "content": "Giải thích định lý Bayes bằng một ví dụ đời thường."}],
    stream=True,  # nên bật stream: đường hầm cắt các yêu cầu im lặng quá 100 giây
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

## Gặp lỗi?

| Hiện tượng | Cách xử lý |
|---|---|
| “Phiên Colab này chưa bật GPU” | *Runtime → Change runtime type → T4 GPU*, rồi chạy lại |
| Tải model đứng lâu | Bấm ■ rồi ▶ lại: phần đã tải được giữ nguyên |
| Model sập giữa chừng | Notebook tự bật lại. Nếu sập liên tục: giảm `CONTEXT` hoặc chọn `UD-Q3_K_XL` |
| Mở link báo lỗi 502 / 1033 | Đợi 10–20 giây rồi tải lại trang |
| Muốn xem log | `/content/qwen38/logs/server.log` và `cloudflared.log` |
"""


def launcher_body():
    src = (ROOT / "launcher.py").read_text(encoding="utf-8")
    body = src.split('\nif __name__ == "__main__":')[0].rstrip() + "\n"
    body = re.sub(r"\A# -\*- coding: utf-8 -\*-\n", "", body)
    return body


def cell(kind, source, **meta):
    lines = source.splitlines(keepends=True)
    c = {"cell_type": kind, "metadata": meta, "source": lines}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def main():
    code = FORM + launcher_body() + CALL
    compile(code, "notebook-cell", "exec")  # bắt lỗi cú pháp ngay lúc build
    nb = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4", "name": NOTEBOOK.name},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": [
            cell("markdown", INTRO, id="intro"),
            cell("code", code, id="run", cellView="form"),
            cell("markdown", OUTRO, id="notes"),
        ],
    }
    NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Đã ghi {NOTEBOOK.name}: {NOTEBOOK.stat().st_size / 1024:.0f} KB, ô chạy dài {code.count(chr(10))} dòng")


if __name__ == "__main__":
    main()
