# KnightSight EdgeVision Challenge - Deepsight Team

This repository contains the source code for the KnightSight EdgeVision Challenge. Our goal is to build a deployable, lightweight vehicle intelligence system that runs efficiently on edge devices.

## High-Level Pipeline Architecture

Our pipeline uses a cascaded approach to maximize accuracy while remaining extremely lightweight for edge devices.

1. **Preprocessing & Low-Light Enhancement**: Applying CLAHE and unsharp masking.
2. **Stage 1 (Vehicle Detection)**: YOLOv8n (Nano) detecting vehicles.
3. **Stage 2 (Plate Detection)**: YOLOv8n (Nano) fine-tuned on Indian Plates to crop plate regions.
4. **Stage 3 (OCR / Text Recognition)**: PaddleOCR mobile model extracting `ABC-1234` text.
5. **Output**: Structured JSON/CSV, displayed via a Streamlit UI (FastAPI backend).

## Tech Stack
* **Detection Models**: Ultralytics YOLOv8n
* **OCR**: PaddleOCR (en-PP-OCRv3-mobile)
* **Optimization**: ONNX Runtime (with Quantization)
* **API / UI**: FastAPI & Streamlit

## Setup Instructions

*Environment setup and usage instructions will be updated as the pipeline is developed.*
