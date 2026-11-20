import asyncio
import json
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from simulation import Simulation

app = FastAPI(title="Self-Driving Car DQN Simulator")

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # Each connected browser tab gets an isolated simulation & DQN brain
    sim = Simulation(width=800, height=600)
    paused = False
    steps_per_tick = 1

    async def receive_commands():
        nonlocal paused, steps_per_tick
        try:
            while True:
                raw = await websocket.receive_text()
                msg = json.loads(raw)
                action = msg.get("type")

                if action == "draw_sand":
                    points = msg.get("points", [])
                    for pt in points:
                        sim.add_sand(pt["x"], pt["y"], radius=10)
                elif action == "clear":
                    sim.clear_sand()
                elif action == "reset_car":
                    sim.reset_car()
                elif action == "save":
                    sim.brain.save("last_brain.pth")
                    await websocket.send_text(json.dumps({
                        "type": "notification",
                        "message": "Saved model weights to last_brain.pth"
                    }))
                elif action == "load":
                    loaded = sim.brain.load("last_brain.pth")
                    await websocket.send_text(json.dumps({
                        "type": "notification",
                        "message": "Loaded last_brain.pth!" if loaded else "No saved last_brain.pth found."
                    }))
                elif action == "toggle_pause":
                    paused = not paused
                elif action == "set_speed":
                    steps_per_tick = max(1, min(10, int(msg.get("value", 1))))
        except WebSocketDisconnect:
            pass

    receiver_task = asyncio.create_task(receive_commands())

    try:
        while True:
            if not paused:
                state = None
                for _ in range(steps_per_tick):
                    state = sim.step()
                state["type"] = "state"
                state["paused"] = paused
                await websocket.send_text(json.dumps(state))
            await asyncio.sleep(1.0 / 60.0)
    except WebSocketDisconnect:
        pass
    finally:
        receiver_task.cancel()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
