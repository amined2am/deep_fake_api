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

### 2) launch API

all the command are in the file cmdAPI, but after the set up you just need to right this commande 
**uvicorn apiForAppNewOne:app --host 0.0.0.0 --port 8080 --reload**

## API endPoint

## 📡 API Endpoints

### 1. `POST /predict/`
This endpoint runs deep-fake detection on a short video.  
It extracts around 30 frames, selects 4 representative ones, and performs inference using the model.  
The response contains the main prediction (FAKE or REAL), the average confidence score, the processing time, the extracted metadata, and links to the frames and cropped faces stored in the static folder.

---

### 2. `POST /predict_csv/`
This endpoint runs detection on about 30 evenly distributed frames of the video.  
Instead of returning only a global prediction, it provides detailed results for each frame, including the predicted label and the confidence score.  
It is useful for generating frame-level analysis and exporting results in CSV-like format.

---

### 3. `POST /integrity/sign`
This endpoint creates an RSA signature for a video file.  
It calculates the SHA256 hash of the video, signs it with a private RSA key, and returns the signature in base64 format along with the hash and the signing timestamp.  
The goal is to prove later that the file has not been modified.

---

### 4. `POST /integrity/verify`
This endpoint checks if a video is authentic by verifying its RSA signature.  
It takes the video and a base64-encoded signature as input, compares the computed SHA256 hash with the signed one, and returns whether the signature is valid or not.  
It also provides additional details such as the hash value and the verification timestamp.
