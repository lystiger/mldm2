# mldm2

## Project Layout

- `src/`: main Python code (`collect_data.py`, feature and model scripts)
- `notebooks/`: Jupyter notebooks (`media.ipynb`)
- `firmware/`: embedded code (`code_ESP.cpp`)
- `scripts/`: standalone experiment scripts
- `data/raw/`: raw captured datasets (glove, camera landmarks, landmark folders)
- `data/processed/`: fused/processed datasets
- `data/models/`: model artifacts (`.joblib`, `.task`)

## Common Commands

Create/activate env and install:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

Collect glove data:

```bash
python src/collect_data.py collect --port /dev/ttyACM0 --baud 115200 --samples 100 --out data/raw/glove_dataset.csv
```

Collect CV landmarks:

```bash
python src/collect_data.py collect-cv --out data/raw/hand_landmarks.csv --camera-id 0
```

Collect glove + CV together (single toggle key):

```bash
python src/collect_data.py collect-sync --port /dev/ttyACM0 --baud 115200 --camera-id 0 --glove-out data/raw/glove_dataset.csv --camera-out data/raw/hand_landmarks.csv
```

Fuse glove + CV:

```bash
python src/collect_data.py fuse --glove data/raw/glove_dataset.csv --camera data/raw/hand_landmarks.csv --out data/processed/fused_dataset.csv --tol-ms 30
```
