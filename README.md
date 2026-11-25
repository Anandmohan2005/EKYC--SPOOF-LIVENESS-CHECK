---
title: eKYC Spoof and Liveness Detection
emoji: 🪪
colorFrom: teal
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# eKYC Spoof and Liveness Detection

Identity verification with random-challenge liveness and spoof detection
(MobileNetV2 trained on NUAA + own recordings), bound to a single face.

Punjab Engineering College, Application of Machine Learning and Data Analysis Lab.

**Demo only: do not upload a real ID document.** Nothing is saved on the server.

## Run locally

    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    python server.py

Then open http://localhost:7860 in Chrome or Edge.

## Known limits

- Genuine users can be rejected in dim light or with a bright window behind them.
- A second person standing far behind the user can be ignored.
