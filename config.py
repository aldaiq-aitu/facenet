# ─── Model Selection ─────────────────────────────────────────────
# Change BACKBONE to switch the recognition model.
# Options: 'facenet', 'mobilefacenet', 'efficientnet_lite0'
BACKBONE       = 'facenet'

# ─── Camera & Detection ─────────────────────────────────────────
FRAME_SKIP     = 15   # run face detection every N frames
RECOG_INTERVAL = 15   # re-recognize every N frames
MIN_FACE_SIZE  = 120  # minimum face size in pixels

# ── Recognition ────────────────────────────────
THRESHOLD      = 0.7  # cosine similarity threshold
DB_FILE        = 'faces_database.pkl'

# ── Model-Specific (used only for FaceNet) ──────────────────────
MODEL_NAME     = 'vggface2'