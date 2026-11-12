# 🧠 Qwen3.8-27B trên Google Colab T4 — một lần bấm là có chatbox

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tiendatmeow-source/qwen38-colab/blob/main/Qwen3_8_27B_Colab.ipynb)
[![CI](https://github.com/tiendatmeow-source/qwen38-colab/actions/workflows/ci.yml/badge.svg)](https://github.com/tiendatmeow-source/qwen38-colab/actions/workflows/ci.yml)

Notebook một ô: bấm ▶, đợi cài đặt xong, nhận link chatbox công khai qua Cloudflare Tunnel.

## Cách dùng

1. Bấm nút **Open in Colab** ở trên.
2. Bấm ▶ ở ô duy nhất. Notebook đã đặt sẵn GPU T4; nếu ô báo chưa có GPU thì vào *Runtime → Change runtime type → T4 GPU*.
3. Đợi các bước chạy xong. Ô sẽ hiện nút **💬 Mở chatbox**, link công khai và mã QR.

Link sống chừng nào ô còn chạy. Bấm ■ để tắt; mỗi lần chạy lại sẽ có link mới.

## Notebook tự làm những gì

| Bước | Chi tiết |
|---|---|
| Chọn bản model | Đo VRAM trống rồi lấy bản GGUF lớn nhất còn nằm trọn trên GPU (bảng bên dưới) |
| Cài llama.cpp | Bản dựng sẵn CUDA, tự tìm thư viện CUDA phù hợp trên máy. Không dùng được thì build từ mã nguồn |
| Tải model | Song song với bước cài, qua `huggingface_hub` (dự phòng bằng `curl`), có phát hiện treo |
| Nạp model | llama.cpp tự lấy context lớn nhất còn vừa VRAM (sàn 8.192 token), KV cache nén `q8_0`, flash attention |
| Tự hạ cấp | Nếu bản đã chọn không nằm trọn trên GPU, tự thử lại không kèm vision/MTP, rồi xuống một bậc quant |
| Chạy thử | Sinh thử một đoạn và đo token/giây trước khi đưa link |
| Mở link | Cloudflare quick tunnel (3 lần thử) → localhost.run qua SSH → link riêng của Colab |
| Canh gác | Model hoặc đường hầm chết thì tự bật lại; dòng thống kê cập nhật 10 giây/lần (token, tốc độ, VRAM) |

## Bản model được chọn

Model lấy từ [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF).

| GPU | Mặc định (`auto`) | `VISION = on` | `MTP = True` |
|---|---|---|---|
| T4 (15 GB) | `UD-IQ4_XS` · 13,4 GB | `UD-Q3_K_XL` · 12,2 GB | `UD-Q3_K_XL` |
| L4 (22,5 GB) | `UD-Q6_K` | — | — |
| A100 (40 GB) | `UD-Q8_K_XL` + vision | — | — |

Trên T4, `UD-IQ4_XS` chỉ dư khoảng 200 MB so với ước lượng. Nếu thực tế không vừa, notebook tự chuyển sang `UD-Q3_K_XL` (phải tải thêm 12 GB). Muốn chắc ăn ngay từ đầu, đặt `QUANT = UD-Q3_K_XL`.

## Tuỳ chọn trên form

| Tuỳ chọn | Mặc định | Ý nghĩa |
|---|---|---|
| `QUANT` | `auto` | Tên bản lượng tử hoá, hoặc `auto` |
| `REASONING` | `medium` | `off` / `low` / `medium` / `xhigh`. Mặc định gốc của model là `xhigh`, rất chậm trên T4 |
| `VISION` | `auto` | Đọc hình ảnh. `auto` chỉ bật khi còn dư VRAM |
| `CONTEXT` | `auto` | Số token, hoặc `auto` để lấy mức lớn nhất còn vừa |
| `PASSWORD` | trống | Đặt thì thành API key. Trong chatbox: ⚙ Settings → API Key |
| `MTP` | tắt | Multi-Token Prediction, thử nghiệm, tốn thêm VRAM |
| `LLAMA_CPP_BUILD` | `b12110` | Tag llama.cpp, hoặc `stable` để lấy bản ổn định mới nhất |
| `HF_TOKEN` | trống | Token Hugging Face (không bắt buộc) |

Tham số lấy mẫu theo khuyến nghị của Qwen/Unsloth: chế độ suy luận dùng `temp 1.0, top_p 0.95, top_k 20`; chế độ `off` dùng `temp 0.7, top_p 0.8, top_k 20, presence_penalty 1.5`.

## Gọi bằng API

Server tương thích OpenAI tại `<link>/v1`:

```python
from openai import OpenAI

client = OpenAI(base_url="https://<link-của-bạn>/v1", api_key="<PASSWORD hoặc chữ bất kỳ>")
stream = client.chat.completions.create(
    model="qwen3.8-27b",
    messages=[{"role": "user", "content": "Xin chào!"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

Nên bật `stream=True`: đường hầm cắt các yêu cầu im lặng quá 100 giây.

## Đã kiểm thử đến đâu

| Hạng mục | Trạng thái |
|---|---|
| Toàn bộ luồng (cài, tải, nạp, chạy thử, mở link, canh gác, dừng) | Đã chạy thật trên CPU với model 135M, cả trong kernel Jupyter; CI chạy lại mỗi lần push |
| Tự bật lại khi model hoặc đường hầm chết | Đã thử bằng cách kill tiến trình |
| Đường hầm Cloudflare và localhost.run | Đã thử cả hai, streaming không bị dồn |
| Chọn quant cho T4 / L4 / A100 | Tính trên kích thước file thật của repo Hugging Face |
| **Chạy trên GPU T4 thật** | **Chưa.** Chuỗi cài bản CUDA, ước lượng VRAM và tốc độ chưa được đo trên Colab |

Nếu lần chạy đầu trên T4 có lỗi, hãy mở issue kèm nội dung `/content/qwen38/logs/server.log`.

## Lưu ý

- Ai có link đều dùng được model. Đặt `PASSWORD` trước khi chia sẻ.
- Colab miễn phí có thể ngắt các phiên chủ yếu phục vụ giao diện web từ xa.
- Model đã tải nằm ở `/content/qwen38/models`, mất khi phiên Colab kết thúc.

## Chạy ngoài Colab

```bash
python launcher.py --quant auto --reasoning medium
```

Cần Linux x64 và GPU NVIDIA. `launcher.py` chỉ dùng thư viện chuẩn của Python.

## Phát triển

`launcher.py` là nguồn duy nhất. Sau khi sửa, sinh lại notebook:

```bash
python tools/build_notebook.py
```

## Giấy phép

Mã nguồn: MIT. Model Qwen3.8-27B, llama.cpp và cloudflared theo giấy phép riêng của từng dự án.
