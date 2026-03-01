"""
Naval Ship Navigation — FastAPI Backend
=======================================
Serves the DDQN agent for inference.

Endpoint:
    POST /navigate
    Body : { "start_row": int, "start_col": int, "goal_row": int, "goal_col": int }
    Returns: { "route": [[r,c],...], "steps": int, "reached_goal": bool,
               "total_reward": float, "bumps": int }

    GET /health       — liveness check
    GET /grid         — returns the 100x100 grid + port locations
"""

import os, math, random, json
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ─────────────────────────────────────────────
#  CONFIG  (must match training notebook exactly)
# ─────────────────────────────────────────────
CFG = {
    "ROWS": 100, "COLS": 100, "PAD": 3, "WINDOW": 7, "N_ACTIONS": 8,
    "CNN_FILTERS": [16, 32], "CNN_KERNEL": 3,
    "CNN_FC": 64, "COORD_FC": 32, "DECISION_FC": 128,
    # Rewards (needed for greedy episode)
    "R_GOAL": 500.0, "R_BUMP": -50.0, "R_STEP": -1.0,
    "R_CLOSER": 2.0,  "R_FARTHER": -1.5,
    "MAX_FUEL": 3000,   # Generous limit for inference
}

ACTION_DELTAS = [
    (-1,0),(-1,+1),(0,+1),(+1,+1),(+1,0),(+1,-1),(0,-1),(-1,-1)
]
ACTION_NAMES = ['N','NE','E','SE','S','SW','W','NW']

device = torch.device("cpu")   # Inference on CPU is fast enough

# ─────────────────────────────────────────────
#  GRID
# ─────────────────────────────────────────────
def build_india_grid():
    ROWS, COLS = 100, 100
    grid = np.zeros((ROWS, COLS), dtype=np.int32)

    def fill_polygon(points):
        min_r = max(0, min(p[0] for p in points))
        max_r = min(ROWS-1, max(p[0] for p in points))
        for r in range(min_r, max_r+1):
            intersections = []
            n = len(points)
            for i in range(n):
                r1,c1 = points[i]; r2,c2 = points[(i+1)%n]
                if (r1<=r<r2) or (r2<=r<r1):
                    c_int = c1 + (r-r1)*(c2-c1)/(r2-r1)
                    intersections.append(c_int)
            intersections.sort()
            for i in range(0, len(intersections)-1, 2):
                c_s = max(0, int(intersections[i]))
                c_e = min(COLS-1, int(intersections[i+1]))
                grid[r, c_s:c_e+1] = 1

    def paint_circle(cr, cc, radius, val=1):
        for r in range(max(0,cr-radius), min(ROWS,cr+radius+1)):
            for c in range(max(0,cc-radius), min(COLS,cc+radius+1)):
                if (r-cr)**2+(c-cc)**2 <= radius**2:
                    grid[r,c] = val

    fill_polygon([(5,20),(5,35),(10,35),(12,37),(17,30),(17,55),(15,60),(22,80),(28,78),(32,72),(33,68),(35,67),(40,62),(43,57),(47,52),(52,48),(57,45),(62,50),(67,42),(63,37),(57,37),(50,35),(43,33),(40,32),(35,30),(30,25),(22,22),(17,20),(10,18),(5,20)])
    fill_polygon([(5,20),(3,18),(0,15),(0,20),(5,28),(10,25),(17,20),(10,18),(5,20)])
    fill_polygon([(22,22),(25,18),(30,18),(33,20),(35,25),(35,30),(30,25),(25,22),(22,22)])
    fill_polygon([(28,72),(25,72),(22,78),(22,82),(28,80),(32,75),(28,72)])
    fill_polygon([(22,82),(20,85),(25,88),(30,90),(35,92),(40,93),(45,92),(50,90),(55,88),(55,85),(45,85),(38,87),(30,88),(25,86),(22,82)])
    fill_polygon([(62,50),(63,52),(65,54),(68,53),(70,51),(69,49),(67,48),(64,48),(62,50)])
    fill_polygon([(0,0),(0,35),(10,35),(15,30),(20,25),(25,20),(30,18),(35,15),(40,12),(45,10),(50,8),(55,5),(60,0),(0,0)])
    fill_polygon([(20,25),(18,28),(17,33),(17,40),(18,45),(20,50),(20,55),(17,58),(15,58),(12,55),(10,50),(10,40),(12,35),(15,30),(20,25)])
    fill_polygon([(45,0),(40,0),(38,2),(35,5),(38,8),(42,10),(48,8),(52,5),(55,2),(55,0),(45,0)])
    for r,c in [(55,80),(57,80),(59,81),(61,81),(63,81),(65,82)]: paint_circle(r,c,1)
    for r,c in [(48,30),(50,29),(52,28)]: paint_circle(r,c,1)
    for r,c in [(70,42),(73,41),(76,40),(79,39),(82,38)]: paint_circle(r,c,1)
    grid[63:67, 47:50] = 0
    grid[67:71, 46:50] = 0

    ports = {"Mumbai":(40,31),"Visakhapatnam":(43,58),"Chennai":(55,51),
             "Kochi":(63,40),"Colombo":(70,50),"Karachi":(25,17),"Goa":(49,34)}
    for name,(r,c) in ports.items():
        for dr in range(-1,2):
            for dc in range(-1,2):
                rr,cc = r+dr, c+dc
                if 0<=rr<ROWS and 0<=cc<COLS: grid[rr,cc]=0
    return grid, ports

GRID, PORTS = build_india_grid()

def make_padded_grid(grid, pad=3):
    rows, cols = grid.shape
    padded = np.ones((rows+2*pad, cols+2*pad), dtype=np.int32)
    padded[pad:pad+rows, pad:pad+cols] = grid
    return padded

PADDED_GRID = make_padded_grid(GRID, CFG["PAD"])
WATER_CELLS = list(zip(*np.where(GRID == 0)))

# ─────────────────────────────────────────────
#  NETWORK  (identical to notebook)
# ─────────────────────────────────────────────
class NavDQN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        f1,f2 = cfg["CNN_FILTERS"]
        k     = cfg["CNN_KERNEL"]
        self.vision = nn.Sequential(
            nn.Conv2d(1,f1,kernel_size=k,padding=0), nn.ReLU(),
            nn.Conv2d(f1,f2,kernel_size=k,padding=0), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(f2*3*3, cfg["CNN_FC"]), nn.ReLU(),
        )
        self.coord = nn.Sequential(
            nn.Linear(4, cfg["COORD_FC"]), nn.ReLU(),
        )
        merged = cfg["CNN_FC"] + cfg["COORD_FC"]
        self.decision = nn.Sequential(
            nn.Linear(merged, cfg["DECISION_FC"]), nn.ReLU(),
            nn.Linear(cfg["DECISION_FC"], cfg["N_ACTIONS"]),
        )

    def forward(self, window, coord):
        return self.decision(torch.cat([self.vision(window), self.coord(coord)], dim=1))

# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
MODEL_PATH = Path("ddqn_checkpoint.pth")
online_net = NavDQN(CFG).to(device)

if MODEL_PATH.exists():
    try:
        ckpt = torch.load(MODEL_PATH, map_location=device)
        online_net.load_state_dict(ckpt["online_state_dict"])
        print(f"✅ Model loaded from {MODEL_PATH}")
        MODEL_LOADED = True
    except Exception as e:
        print(f"⚠️  Could not load checkpoint: {e}. Using random weights.")
        MODEL_LOADED = False
else:
    print("⚠️  ddqn_checkpoint.pth not found. Using random weights.")
    MODEL_LOADED = False

online_net.eval()

# ─────────────────────────────────────────────
#  INFERENCE HELPERS
# ─────────────────────────────────────────────
def get_state(r, c, goal_r, goal_c):
    ROWS, COLS = CFG["ROWS"], CFG["COLS"]
    W = CFG["WINDOW"]
    window = PADDED_GRID[r:r+W, c:c+W].astype(np.float32)
    dx = (goal_r - r) / (ROWS-1)
    dy = (goal_c - c) / (COLS-1)
    coord = np.array([r/(ROWS-1), c/(COLS-1), dx, dy], dtype=np.float32)
    return window, coord

def greedy_episode(start_r, start_c, goal_r, goal_c, max_steps=3000):
    r, c = start_r, start_c
    route        = [(r, c)]
    total_reward = 0.0
    bumps        = 0
    prev_dist    = math.sqrt((r-goal_r)**2 + (c-goal_c)**2)

    with torch.no_grad():
        for _ in range(max_steps):
            window, coord = get_state(r, c, goal_r, goal_c)
            w_t = torch.tensor(window).unsqueeze(0).unsqueeze(0).to(device)
            c_t = torch.tensor(coord).unsqueeze(0).to(device)
            q   = online_net(w_t, c_t).squeeze(0).cpu().numpy()
            action = int(np.argmax(q))

            dr, dc = ACTION_DELTAS[action]
            nr, nc = r+dr, c+dc

            # Bump check
            oob  = not (0 <= nr < CFG["ROWS"] and 0 <= nc < CFG["COLS"])
            land = (not oob) and (GRID[nr, nc] == 1)

            if oob or land:
                total_reward += CFG["R_BUMP"]
                bumps        += 1
            else:
                r, c = nr, nc
                curr_dist = math.sqrt((r-goal_r)**2+(c-goal_c)**2)
                delta = prev_dist - curr_dist
                if delta > 0:
                    total_reward += CFG["R_STEP"] + CFG["R_CLOSER"] * delta
                else:
                    total_reward += CFG["R_STEP"] + CFG["R_FARTHER"] * abs(delta)
                prev_dist = curr_dist
                route.append((r, c))

            if r == goal_r and c == goal_c:
                total_reward += CFG["R_GOAL"]
                break

            # Anti-loop: if route is very long with no progress, stop
            if len(route) > 2000:
                break

    reached = (r == goal_r and c == goal_c)
    return route, len(route)-1, reached, total_reward, bumps

# ─────────────────────────────────────────────
#  FASTAPI APP
# ─────────────────────────────────────────────
app = FastAPI(
    title="Naval Ship Navigation API",
    description="DDQN-powered ship navigation on Indian Ocean Region grid",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class NavigateRequest(BaseModel):
    start_row: int
    start_col: int
    goal_row:  int
    goal_col:  int

class NavigateResponse(BaseModel):
    route:        List[List[int]]
    steps:        int
    reached_goal: bool
    total_reward: float
    bumps:        int
    model_loaded: bool

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_LOADED,
            "grid_shape": [100, 100], "water_cells": len(WATER_CELLS)}

@app.get("/grid")
def get_grid():
    return {
        "grid":  GRID.tolist(),
        "ports": {k: list(v) for k,v in PORTS.items()},
        "rows":  100, "cols": 100
    }

@app.post("/navigate", response_model=NavigateResponse)
def navigate(req: NavigateRequest):
    ROWS, COLS = CFG["ROWS"], CFG["COLS"]

    # Validate inputs
    for val, name in [(req.start_row,"start_row"),(req.start_col,"start_col"),
                      (req.goal_row,"goal_row"),  (req.goal_col,"goal_col")]:
        if not (0 <= val < 100):
            raise HTTPException(400, f"{name}={val} out of range [0,99]")

    if GRID[req.start_row, req.start_col] == 1:
        raise HTTPException(400, f"Start ({req.start_row},{req.start_col}) is land. Choose a water cell.")
    if GRID[req.goal_row, req.goal_col] == 1:
        raise HTTPException(400, f"Goal ({req.goal_row},{req.goal_col}) is land. Choose a water cell.")
    if req.start_row==req.goal_row and req.start_col==req.goal_col:
        raise HTTPException(400, "Start and goal must be different cells.")

    route, steps, reached, reward, bumps = greedy_episode(
        req.start_row, req.start_col, req.goal_row, req.goal_col)

    return NavigateResponse(
        route        = [[r,c] for r,c in route],
        steps        = steps,
        reached_goal = reached,
        total_reward = round(reward, 2),
        bumps        = bumps,
        model_loaded = MODEL_LOADED,
    )

# Serve frontend (index.html + static files)
@app.get("/")
def root():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
