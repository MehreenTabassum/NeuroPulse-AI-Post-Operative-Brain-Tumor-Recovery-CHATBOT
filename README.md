<div align="center">

# 🧠 NeuroRecovery AI — Level 3 Medical AI Agent

**Post-Operative Brain Tumor Recovery Analysis**
*Powered by LangGraph · Groq (LLaMA 3) · FastAPI · CNN-ViT · BraTS-2024*

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-15-000000?style=for-the-badge&logo=nextdotjs&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2-1C3C3C?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-LLaMA3-F55036?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> ⚠️ **Research prototype only. Not validated for clinical decision-making.**

</div>

---

## 📌 Overview

NeuroRecovery AI is a **Level 3 Agentic Medical AI system** that performs post-operative recovery analysis for Glioblastoma Multiforme (GBM) patients using the **UCSD-PTGBM-BraTS-2024** dataset.

The system accepts:
- Structured **clinical metadata** (KPS score, tumor grade, treatment protocol, IDH/MGMT status)
- Real **3D MRI scans** in NIfTI format (`.nii` / `.nii.gz`) uploaded directly from the frontend

It then:
1. Extracts deep multimodal features using a custom **3D CNN-ViT hybrid model** (17M parameters)
2. Retrieves evidence from a **RAG vector store** of neuro-oncology literature via Chroma + Ollama embeddings
3. Synthesises a structured **7-section clinical report** via **Groq API (LLaMA 3.1)**

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Next.js Frontend  (localhost:3000)              │
│     Clinical Dashboard · MRI Upload · React · Tailwind       │
└──────────────────────┬──────────────────────────────────────┘
                       │ POST /analyze-recovery
┌──────────────────────▼──────────────────────────────────────┐
│              FastAPI Agent API  (localhost:8000)              │
│                                                              │
│   ┌──────────────────────────────────────────────────────┐  │
│   │            LangGraph Orchestrator (agent.py)          │  │
│   │                                                      │  │
│   │  [Node 1] extract_features_tool                      │  │
│   │      └─► POST localhost:8001/extract-features        │  │
│   │                                                      │  │
│   │  [Node 2] retrieve_literature_node                   │  │
│   │      └─► Chroma DB + nomic-embed-text (Ollama)       │  │
│   │                                                      │  │
│   │  [Verification Edge] retry if < 2 docs               │  │
│   │                                                      │  │
│   │  [Node 3] synthesize_report_node                     │  │
│   │      └─► Groq API — LLaMA 3.1 8B Instant            │  │
│   └──────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│           Vision Microservice  (localhost:8001)               │
│                                                              │
│   .nii/.nii.gz Upload                                       │
│       └─► nibabel loader + BraTS preprocessing              │
│       └─► 3D CNN-ViT Feature Extractor (PyTorch)            │
│             ├─ CNN: 3D ResNet (4 stages + SE attention)     │
│             ├─ ViT: Patch embed + 8-layer Transformer       │
│             └─ Fusion: Bidirectional Cross-Attention        │
│       └─► 768-dim feature vector → JSON                     │
└──────────────────────────────────────────────────────────────┘
```

---

## 🧬 CNN-ViT Model Architecture

| Component | Design | Output |
|---|---|---|
| Input | 4-ch MRI (T1, T1CE, T2, FLAIR) | `(B, 4, 128, 128, 128)` |
| CNN Stem | Conv3d 7³ s2 + MaxPool | `(B, 32, 32³)` |
| CNN Stages 0–3 | Pre-act ResBlocks + SE attention | `(B, 512, 4³)` |
| CNN Head | AdaptiveAvgPool3d + Linear + LN | `(B, 384)` |
| Patch Embed | Conv3d(k=16, s=16) + LN | `(B, 512, 384)` |
| Pos Embedding | Factorised 3D learnable (D/H/W) | added in-place |
| ViT Encoder | 8× pre-norm TransformerEncoderLayer | `(B, 384)` |
| Fusion | Bidirectional cross-attention + MLP | `(B, 768)` |
| Classifier | Dropout + Linear (train only) | `(B, 2)` |
| **Total Params** | **~17M trainable** | |

---

## 📁 Project Structure

```
neurorecovery/
├── agent.py              # LangGraph orchestrator (3 nodes + verification edge)
├── main.py               # FastAPI — /analyze-recovery endpoint
├── vision.py             # 3D CNN-ViT PyTorch model
├── vision_service.py     # FastAPI microservice — /extract-features endpoint
├── ingest_literature.py  # One-time Chroma vector store population
├── run_all.py            # Single launcher for all 3 services
├── test_stack.py         # End-to-end smoke test
├── requirements.txt
├── README.md
├── chroma_db/            # Auto-generated vector store (gitignore this)
└── frontend/
    ├── app/
    │   ├── page.tsx      # Clinical dashboard with MRI upload
    │   ├── layout.tsx
    │   └── globals.css
    ├── tailwind.config.js
    ├── postcss.config.js
    └── package.json
```

---

## ⚙️ Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | Conda recommended |
| Node.js | 18+ | For Next.js frontend |
| Ollama | Latest | [ollama.com](https://ollama.com) — for embeddings only |
| Groq API Key | Free | [console.groq.com](https://console.groq.com) — for LLM |
| RAM | 4 GB+ free | 8 GB recommended |

---

## 🚀 Setup & Installation

### 1. Clone
```bash
git clone https://github.com/YOUR_USERNAME/neurorecovery-ai.git
cd neurorecovery-ai
```

### 2. Python environment
```bash
conda create -n neuro_agent python=3.11
conda activate neuro_agent
pip install -r requirements.txt
```

### 3. Frontend
```bash
cd frontend
npm install
cd ..
```

### 4. Ollama (embeddings only)
```bash
ollama pull nomic-embed-text
```

### 5. Groq API key
Get a free key at [console.groq.com](https://console.groq.com) then set it:

**Windows:**
```powershell
[System.Environment]::SetEnvironmentVariable("GROQ_API_KEY","your_key_here","User")
```
**Mac/Linux:**
```bash
export GROQ_API_KEY="your_key_here"
```

### 6. Populate vector store (run once)
```bash
python ingest_literature.py
```

### 7. Launch everything
```bash
python run_all.py
```

| Service | URL |
|---|---|
| 🖥️ Frontend Dashboard | http://localhost:3000 |
| 🤖 Agent API | http://localhost:8000/docs |
| 🔬 Vision API | http://localhost:8001/docs |

---

## 🧪 How to Use

1. Open **http://localhost:3000**
2. *(Optional)* Upload a BraTS-2024 `.nii.gz` MRI scan using the upload panel
3. Fill in patient parameters (KPS score, tumor grade, treatment protocol, etc.)
4. Click **Run Agentic Analysis**
5. The system will:
   - Extract 768-dim CNN-ViT features from the MRI
   - Retrieve relevant neuro-oncology literature via RAG
   - Generate a 7-section clinical report via LLaMA 3.1

---

## 🗂️ BraTS-2024 Dataset

Download from:
- [Synapse Platform](https://www.synapse.org/#!Synapse:syn51156910) (official)
- [Kaggle Mirror](https://www.kaggle.com/datasets/andrewmvd/brain-tumor-segmentation-in-mri-brats-2015)

Each patient folder contains 4 NIfTI files:
```
BraTS-GLI-00000-000/
├── BraTS-GLI-00000-000-t1n.nii.gz    ← T1
├── BraTS-GLI-00000-000-t1c.nii.gz    ← T1CE
├── BraTS-GLI-00000-000-t2w.nii.gz    ← T2
└── BraTS-GLI-00000-000-t2f.nii.gz    ← FLAIR
```
Upload any one of these to the frontend for analysis.

---

## 🧪 End-to-End Test

```bash
python test_stack.py
```

Expected:
```
══════ 1. Vision Service ══════
  ✔  Health probe
  ✔  Feature dim = 768
  ✔  Feature extraction accepted

══════ 2. Agent API ══════
  ✔  Health probe
  ✔  All 3 nodes present
  ✔  Analysis endpoint returned 200
  ✔  Report is non-empty

══════ Result: 13/13 checks passed ══════
```

---

## 🗺️ Roadmap

- [ ] Fine-tune CNN-ViT on labelled BraTS-2024 data
- [ ] Stack all 4 MRI modalities (T1/T1CE/T2/FLAIR) in single forward pass
- [ ] DICOM input support
- [ ] RANO criteria automated scoring
- [ ] Patient history tracking across sessions
- [ ] Docker Compose deployment

---

## 📚 References

- **BraTS 2024** — Brain Tumor Segmentation Challenge 2024
- **UCSD-PTGBM** — Post-Treatment GBM dataset
- Stupp et al. (2005) — *Radiotherapy plus Concomitant and Adjuvant Temozolomide for Glioblastoma*, NEJM
- Wen et al. (2010) — *Updated Response Assessment Criteria (RANO)*, Journal of Clinical Oncology

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with LangGraph · PyTorch · FastAPI · Next.js · Groq**
*For research and educational purposes only*

</div>
