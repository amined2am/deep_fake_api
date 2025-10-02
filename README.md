# Deep-Fake API

This project is a FastAPI backend that runs deep-fake detection on videos.  
You send a short video → it extracts frames → detects faces → runs them through a deep learning model (ResNeXt + BiLSTM) → and returns if the video is FAKE or REAL with a confidence score.

⚠️ **Note:** The model weights (`checkpoint.pth`) are **not in this repo**.  
I keep them in my Drive:  
`IMAP > publication > amine > checkpoint`  
You need to download the checkpoint before using the API.

---

## 🔧 Tech Stack

- **FastAPI + Uvicorn** → REST API
- **PyTorch** → deep learning runtime
- **OpenCV / dlib** → video decoding, frame extraction, face detection
- **Model** → ResNeXt (for spatial features) + BiLSTM (for temporal sequence consistency)

---

## 📂 Repo structure

- `apiForAppNewOne.py` → main FastAPI app  
- `deepfake_model.py`, `model_definition.py` → model loader + forward pass  
- `video_utils.py` → video preprocessing, face detection, normalization  
- `usage_example.py` → client example to call the API  
- `requirements.txt` → dependencies  
- `benchmark_*` + `apiBenchmark.py` → quick perf tests (latency/throughput)  
- `videos/` and `SHORT_VIDEO_HD/` → demo input videos  

---

## 🚀 How to run

### 1) Setup
```bash
python -m venv .venv
source .venv/bin/activate        # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
