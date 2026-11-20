# Self-Driving Car DQN Web Application

Full-stack web implementation of the Self-Driving Car Deep Q-Learning project using **FastAPI + WebSockets** on the backend and **HTML5 Canvas + Chart.js** on the frontend.

## Project Structure
- `ai.py`: Modernized PyTorch Deep Q-Network (`Network`, `ReplayMemory`, `Dqn`).
- `simulation.py`: Headless physics and sensor simulation decoupled from Kivy.
- `server.py`: FastAPI WebSocket server streaming simulation frames at 60 FPS with per-user session isolation.
- `static/index.html`: Interactive web UI with dual-layer HTML5 Canvas, sand drawing, speed controls, and live score chart.
- `Dockerfile`: Ready for deployment on Hugging Face Spaces, Render, or Railway.

## Run Locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the server:
   ```bash
   python server.py
   ```
3. Open `http://127.0.0.1:8000` in your browser.
