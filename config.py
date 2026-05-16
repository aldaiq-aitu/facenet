# Model selection
BACKBONE = "facenet"
CHECKPOINT_PATH = None  # e.g. "experiments/mobilefacenet/best.pt"

# Camera and detection
FRAME_SKIP = 15
RECOG_INTERVAL = 15
MIN_FACE_SIZE = 120

# Recognition
THRESHOLD = 0.7
USE_CHECKPOINT_THRESHOLD = True
DB_FILE = "faces_database.pkl"
