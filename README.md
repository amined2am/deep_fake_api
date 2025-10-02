# Deep-Fake API

This project is a FastAPI backend that runs deep-fake detection on videos.  
You send a short video → it extracts frames → detects faces → runs them through a deep learning model (ResNeXt + BiLSTM) → and returns if the video is FAKE or REAL with a confidence score.

⚠️ **Note:** The model weights (`checkpoint.pth`) are **not in this repo**.  
I keep them in the Drive:  
`IMAP > publication > amine > checkpoint` (you can chose the old one or the new one) 
You need to download the checkpoint before using the API. ( if you don't have the checkpoint, it’s useless to launch the api it will not launch without having the model trained )  

---

## 🔧 Tech Stack

- **FastAPI + Uvicorn** → REST API
- **PyTorch** → deep learning runtime
- **OpenCV / MEDIA PIPE /dlib** → video decoding, frame extraction, face detection
- **Model** → ResNeXt (for spatial features) + BiLSTM (for temporal sequence consistency)

PS : for the faces_cropped I have used Media pipe as a priority before using open CV
---

## 📂 Repo structure

- `apiForAppNewOne.py` → main FastAPI app  ( you can rename it at api if you want, I name it like that because it's to the new model) 
- `deepfake_model.py`, `model_definition.py` → model loader + forward pass  
- `video_utils.py` → video preprocessing, face detection, normalization  
- `usage_example.py` → client example to call the API  
- `requirements.txt` → dependencies  
- `benchmark_*` + `apiBenchmark.py` → quick perf tests (latency/throughput)  
- `videos/` and `SHORT_VIDEO_HD/` → demo input videos
- 'static' this folder is used to store split frames and cropped faces  

---

## 🚀 How to run

### 1) Setup
```bash
python -m venv .venv
source .venv/bin/activate        # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt  # probably you have some conflict about dependence but it's fine
pip install -r requirements.lock.txt # 
