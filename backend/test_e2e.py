import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
import websockets
import numpy as np
import cv2
import urllib.request
import io

async def test_websocket():
    uri = "ws://localhost:8000/ws/live"
    print("Testing WebSocket endpoint:", uri)
    async with websockets.connect(uri) as ws:
        # Send synthetic landmark frame
        dummy_landmarks = {
            "pose": [[0.5, 0.5, 0.0] for _ in range(25)],
            "face": [[0.5, 0.5, 0.0] for _ in range(7)],
            "left_hand": [[0.4, 0.6, 0.0] for _ in range(21)],
            "right_hand": [[0.6, 0.6, 0.0] for _ in range(21)]
        }

        for i in range(20):
            payload = {"type": "landmarks", "landmarks": dummy_landmarks}
            await ws.send(json.dumps(payload))
            msg = await ws.recv()
            data = json.loads(msg)
            if i == 19:
                print(f"[PASS] Received live prediction from WebSocket: {data.get('prediction')} (Conf: {data.get('confidence')})")
                assert "top_predictions" in data
                assert len(data["top_predictions"]) <= 5
                assert data.get("hand_detected") is True

def test_image_predict():
    print("Testing POST /predict/image endpoint...")
    # Create a synthetic image with OpenCV
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(img, (320, 240), 50, (255, 255, 255), -1)
    _, img_encoded = cv2.imencode('.jpg', img)
    
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = io.BytesIO()
    body.write(f"--{boundary}\r\n".encode())
    body.write(b'Content-Disposition: form-data; name="file"; filename="test.jpg"\r\n')
    body.write(b'Content-Type: image/jpeg\r\n\r\n')
    body.write(img_encoded.tobytes())
    body.write(f"\r\n--{boundary}--\r\n".encode())

    req = urllib.request.Request(
        "http://localhost:8000/predict/image",
        data=body.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )

    with urllib.request.urlopen(req) as res:
        data = json.loads(res.read())
        print(f"[PASS] Image predict response: {data.get('prediction')} (Disclaimer present: {'disclaimer' in data})")
        assert data.get("success") is True
        assert "disclaimer" in data

def test_video_predict():
    print("Testing POST /predict/video endpoint...")
    # Create a synthetic 1-second video
    tmp_vid_path = "test_synthetic.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(tmp_vid_path, fourcc, 30.0, (320, 240))
    for _ in range(30):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        out.write(frame)
    out.release()

    with open(tmp_vid_path, "rb") as f:
        vid_bytes = f.read()

    if os.path.exists(tmp_vid_path):
        os.remove(tmp_vid_path)

    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = io.BytesIO()
    body.write(f"--{boundary}\r\n".encode())
    body.write(b'Content-Disposition: form-data; name="file"; filename="test.mp4"\r\n')
    body.write(b'Content-Type: video/mp4\r\n\r\n')
    body.write(vid_bytes)
    body.write(f"\r\n--{boundary}--\r\n".encode())

    req = urllib.request.Request(
        "http://localhost:8000/predict/video",
        data=body.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )

    with urllib.request.urlopen(req) as res:
        data = json.loads(res.read())
        print(f"[PASS] Video predict response: {data.get('prediction')} (Top-5: {len(data.get('top_predictions', []))})")
        assert data.get("success") is True
        assert len(data.get("top_predictions", [])) == 5

if __name__ == "__main__":
    print("=== RUNNING ASL VISION END-TO-END TESTS ===")
    asyncio.run(test_websocket())
    test_image_predict()
    test_video_predict()
    print(">>> ALL END-TO-END TESTS PASSED SUCCESSFULLY! <<<")
