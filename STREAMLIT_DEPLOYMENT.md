# Deploying ASL Vision on Streamlit Community Cloud

This guide provides step-by-step instructions to deploy the ASL Vision application on **[Streamlit Community Cloud](https://share.streamlit.io)** (100% Free).

---

## 1-Click Streamlit Cloud Deployment

1. Visit **[share.streamlit.io](https://share.streamlit.io/)** and log in with your GitHub account (**`HarshitaSmriti`**).
2. Click **New app** (top right).
3. Fill in the repository details:
   - **Repository**: `HarshitaSmriti/asl-vision`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
   - **App URL**: `asl-vision` *(or your custom subdomain)*
4. Under **Advanced settings**:
   - **Python version**: Select `3.11`
5. Click **Deploy!**

Streamlit Cloud will install the dependencies from `requirements.txt` and `packages.txt` and launch the app in ~1-2 minutes at:
**`https://asl-vision.streamlit.app`** (or your chosen URL).

---

## What is Included in the Streamlit App (`streamlit_app.py`)

- **📹 Live Camera Mode (Primary)**:
  - In-browser 60 FPS MediaPipe hand and pose skeleton tracking overlay with glowing cyber aesthetics.
  - Snapshot camera classification using the trained `ASLTransformer` model.
- **🎬 Video Upload Mode**:
  - Upload MP4, WebM, MOV, or AVI video clips.
  - Full MediaPipe temporal landmark extraction with live progress bar.
  - Resampling to 64 frames, 696-dim velocity feature calculation, and Top-5 prediction breakdown.
- **🖼️ Image Upload Mode**:
  - Upload single photos with landmark extraction and clear temporal disclaimer.
- **📖 95-Sign Dictionary Explorer**:
  - Searchable interactive gallery of all 95 recognizable ASL vocabulary words.

---

## Running Streamlit Locally

To run the Streamlit application on your local machine:

```bash
streamlit run streamlit_app.py
```

It will automatically open in your browser at `http://localhost:8501`.
