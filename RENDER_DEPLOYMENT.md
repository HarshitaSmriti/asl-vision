# Deploying ASL Vision on Render

This guide provides step-by-step instructions to deploy the ASL Vision backend (and full-stack frontend) to [Render](https://render.com).

---

## Method 1: One-Click Render Blueprint (Recommended)

1. Log in to your [Render Dashboard](https://dashboard.render.com).
2. Click **New +** in the top right and select **Blueprint**.
3. Connect your GitHub account and select your repository: **`HarshitaSmriti/asl-vision`**.
4. Render will automatically read [`render.yaml`](./render.yaml) and configure the service:
   - **Service Name**: `asl-vision-app`
   - **Runtime**: `Python`
   - **Build Command**: `pip install -r requirements.txt && cd frontend && npm install && npm run build && cd ..`
   - **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
   - **Health Check Path**: `/health`
5. Click **Apply**. Render will build and deploy the application with live URL (e.g. `https://asl-vision-app.onrender.com`).

---

## Method 2: Manual Web Service Setup on Render

If you prefer to configure the Web Service manually:

1. In the [Render Dashboard](https://dashboard.render.com), click **New +** $\rightarrow$ **Web Service**.
2. Select your repository: **`HarshitaSmriti/asl-vision`**.
3. Fill in the following settings:
   - **Name**: `asl-vision-backend`
   - **Language / Runtime**: `Python 3`
   - **Region**: Any (e.g. `Oregon (US West)` or `Frankfurt (EU)`)
   - **Branch**: `main`
   - **Root Directory**: *(leave blank / default)*
   - **Build Command**:
     ```bash
     pip install -r requirements.txt && cd frontend && npm install && npm run build && cd ..
     ```
   - **Start Command**:
     ```bash
     uvicorn backend.main:app --host 0.0.0.0 --port $PORT
     ```
   - **Instance Type**: `Free`
4. Under **Advanced Settings**:
   - **Health Check Path**: `/health`
   - **Environment Variables**:
     - `PYTHON_VERSION` = `3.11.9`
     - `NODE_VERSION` = `20.10.0`
5. Click **Create Web Service**.

---

## Verifying the Deployment

Once Render finishes building:
- **API Health**: Navigate to `https://<your-render-url>/health` $\rightarrow$ `{"status": "ok"}`
- **Model Info**: Navigate to `https://<your-render-url>/api/info` $\rightarrow$ Model metadata & 95 ASL classes.
- **Web App**: Navigate to `https://<your-render-url>/` $\rightarrow$ Full interactive Live ASL Vision app with live webcam inference & visual landmark overlay.
- **WebSocket Streaming**: Live webcam streams directly to `wss://<your-render-url>/ws/live`.
