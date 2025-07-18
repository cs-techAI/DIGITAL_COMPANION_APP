# DIGITAL_COMPANION_APP
# 🧠 Advanced RAG Chatbot (“Digital Companion”)

A multi-user, multi-modal Retrieval-Augmented-Generation chatbot built with  
Streamlit, Gemini Flash, FAISS and faster-whisper.  
It supports PDF/TXT ingestion, offline video transcription, and automatic
YouTube processing—all with strict document grounding and role-based UI
(Student / Teacher / Parent).

---

## ✨ Key Features
* **Strict Document Grounding** – no hallucinations; confidence score shown.  
* **Multi-Role Login** – separate themes, quotas and dashboards.  
* **Unlimited File Size** – up to 10 uploads for students (higher for teachers).  
* **Video & YouTube Support** – FFmpeg-free audio extraction + Whisper STT.  
* **Persistent Vector Store** – FAISS in-memory index with metadata.  
* **Full Audit Trail** – exportable chat & grounding metrics.

---

## 📁 Project Structure

```
DIGITAL_COMPANION_APP/
│
├── DIGITAL_COMPANION_APP.py
├── .streamlit/
    ├── secrets.toml


## 🖥️ Local Installation

### 1. Clone repository  
git clone <your-repo>
cd <your-repo>

### 2. Create virtual environment  
python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate

### 3. Install dependencies  
pip install -r requirements.txt

### 4. Obtain Gemini API key  
1. Visit https://aistudio.google.com/  
2. Create / copy **API key**.  
3. Either:  
   * Add a `.streamlit/secrets.toml` file:  
     ```
     GEMINI_API_KEY = "YOUR_KEY"
     ```  
   * _or_ enter the key in the UI at first launch.

### 5. Run the app  
streamlit run DIGITAL_COMPANION_APP.py

_The browser opens automatically at http://localhost:8501._

---

## 🚀 User Guide

### Login
| Role    | Username | Password   |
|---------|----------|------------|
| Student | student1 | student123 |
| Teacher | teacher1 | teacher123 |
| Parent  | parent1  | parent123  |

### Upload Limits  
* **Student** – 10 files (any size)  
* **Teacher** – 20 files • 50 MB suggestion  
* **Parent**  – 10 files • 25 MB suggestion  

### Typical Workflow
1. **Log in** with a demo or your own account.  
2. **Upload** PDFs / TXTs under “📄 Documents” _or_ videos under “🎥 Videos”.  
3. Optionally paste a **YouTube URL** and click “Process YouTube”.  
4. Watch “Creating embeddings…” spinner; chunks count appears.  
5. Ask questions in the chat panel – answers are strictly grounded.  
6. Check confidence indicator (🟢 ≥ 0.8, 🟡 0.6-0.79, 🔴 < 0.6).  
7. View sources and grounding metrics via 🔍 expander.  
8. Use **Export Chat** to download a JSON audit.  
9. **Logout** anytime via red 🚪 button in the sidebar.

---

## 🛠️ Troubleshooting

| Issue | Fix |
|-------|-----|
| *yt-dlp not found* | `pip install yt-dlp` (no version pin). |
| *Whisper model slow* | Replace `"base"` with `"tiny"` in `WhisperModel(...)`. |
| *CUDA available* | Install `faster-whisper[cuda]` and set `device="cuda"`. |
| *Memory errors on huge PDFs* | Split PDF before upload or increase `chunk_size`. |

---

## ⚖️ Licensing & Credits
This template wraps the following OSS projects: Streamlit, sentence-transformers,
FAISS, faster-whisper, yt-dlp, youtube-transcript-api, streamlit-authenticator.
All licenses remain with their respective authors.
