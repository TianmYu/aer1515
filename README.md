# Multimodal intent prediction prototype

This small prototype demonstrates a path to combine the Shutter (Yale) pose CSVs and simple trajectory features into a multimodal transformer for intent prediction.

Files added:
- `transformer_model.py` - multimodal transformer prototype (pose + traj)
- `data_utils.py` - dataset that creates sliding windows from the provided CSVs
- `train.py` - small smoke-training script (one epoch, CPU-friendly)
- `requirements.txt` - minimal dependencies

How to run (PowerShell):

```powershell
python .\train.py --data datasets --seq_len 30 --batch 8
```

Notes:
- The dataset loader extracts all columns that end with `_x` or `_y` as pose keypoints and uses `pelvis_x`/`pelvis_y` as trajectory. It labels a window as positive if any frame in the window has `interacting`=True.
- This is a prototype. Next steps: add pretraining for modality encoders, more robust timestamp-based alignment, handling of missing modalities across datasets, and additional modalities (gaze, face embeddings).
