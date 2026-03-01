# 🚢 Naval Navigator — DDQN Ship Navigation AI

A deployed web application where users can select any start and destination point
on the Indian Ocean Region grid and watch a **Double DQN trained neural network**
compute the optimal ship route in real time.

---

## 📁 Project Structure

```
naval_deploy/
├── app.py                  # FastAPI backend (inference server)
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container config for Render/Railway
├── ddqn_checkpoint.pth     # ← YOU MUST ADD THIS FILE (trained model)
├── static/
│   └── index.html          # Frontend (grid visualisation + UI)
└── README.md               # This file
```

---

## 🧠 How to Update the Model

When your **full 10,000-episode training** completes on Google Colab:

1. The final cell in your notebook saves `ddqn_checkpoint.pth`
2. Download it from Colab (Files panel → right-click → Download)
3. Replace the existing `ddqn_checkpoint.pth` in this folder
4. Push to GitHub — Render will **auto-redeploy** within 2 minutes

---

## 🚀 Deployment on Render.com (Free — Recommended)

### Step 1 — Create GitHub Repository

1. Go to [github.com](https://github.com) → New repository
2. Name it `naval-navigator` (or anything)
3. Set to **Public**

### Step 2 — Upload Files

Upload all files in this folder to the repo:
```
app.py
requirements.txt
Dockerfile
ddqn_checkpoint.pth     ← your trained model
static/index.html
README.md
```

> **Important:** `ddqn_checkpoint.pth` must be in the repo root alongside `app.py`.
> GitHub has a 100MB file limit. If your `.pth` exceeds this, use
> [Git LFS](https://git-lfs.github.com/) — it's free.

### Step 3 — Deploy on Render

1. Go to [render.com](https://render.com) → Sign up (free)
2. Dashboard → **New** → **Web Service**
3. Connect your GitHub account → Select `naval-navigator` repo
4. Configure:
   - **Name:** `naval-navigator`
   - **Environment:** `Docker`
   - **Branch:** `main`
   - **Instance Type:** `Free`
5. Click **Create Web Service**
6. Wait ~3–5 minutes for the first build
7. Your app is live at: `https://naval-navigator.onrender.com`

> **Note:** Free Render instances spin down after 15 minutes of inactivity.
> The first request after sleep takes ~20–30 seconds to wake up. This is normal.

---

## 🚀 Alternative: Railway.app

1. Go to [railway.app](https://railway.app) → Sign up with GitHub
2. New Project → Deploy from GitHub Repo → Select your repo
3. Railway auto-detects the Dockerfile and deploys
4. Live URL appears in the dashboard within 3–4 minutes

---

## 🏃 Running Locally (for testing)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place your model in the same directory as app.py
cp path/to/ddqn_checkpoint.pth .

# 3. Run the server
python app.py

# 4. Open browser
# http://localhost:8000
```

---

## 🔌 API Endpoints

| Method | Endpoint    | Description |
|--------|-------------|-------------|
| GET    | `/`         | Serves the frontend |
| GET    | `/health`   | Liveness check + model status |
| GET    | `/grid`     | Returns 100×100 grid + port coords |
| POST   | `/navigate` | Runs DDQN inference, returns route |

### `/navigate` Request Body

```json
{
  "start_row": 40,
  "start_col": 31,
  "goal_row":  43,
  "goal_col":  58
}
```

### `/navigate` Response

```json
{
  "route":        [[40,31], [39,32], ...],
  "steps":        284,
  "reached_goal": true,
  "total_reward": 312.4,
  "bumps":        3,
  "model_loaded": true
}
```

---

## ⚙️ Architecture

```
User Browser
    │
    │  GET /grid       → 100×100 grid + ports
    │  POST /navigate  → route JSON
    │
FastAPI Server (app.py)
    │
    ├── NavDQN (PyTorch)          ← loads ddqn_checkpoint.pth
    ├── greedy_episode()          ← ε=0 inference
    ├── build_india_grid()        ← same grid as training
    └── StaticFiles(static/)      ← serves index.html
```

---

## 🔁 Swapping to Full Model

The app works with any checkpoint saved by the training notebook.
When full training completes, simply replace `ddqn_checkpoint.pth` and redeploy.
The server will automatically load the new weights on startup.

---

*Naval Ship Navigation · Deep RL Project · Indian Naval Academy*
