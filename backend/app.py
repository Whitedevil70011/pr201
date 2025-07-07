import os
import tempfile
import logging

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

# ─── Setup ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "model.h5"))  # ← updated

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("deepfake-api")

app = Flask(__name__)
CORS(app)

# ─── Load Model & Determine Expected Input Dimension ──────────────────────────
try:
    logger.info(f"Loading model from {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Get input shape and handle different model types
    input_shape = model.input_shape
    logger.info(f"Model input shape: {input_shape}")
    
    if len(input_shape) == 4:  # CNN model: (batch, height, width, channels)
        _, height, width, channels = input_shape
        EXPECTED_HEIGHT = height
        EXPECTED_WIDTH = width
        EXPECTED_CHANNELS = channels
        MODEL_TYPE = "CNN"
        logger.info(f"CNN model detected: {height}x{width}x{channels}")
    elif len(input_shape) == 2:  # Dense model: (batch, features)
        _, features = input_shape
        MODEL_TYPE = "DENSE"
        # Try to infer image dimensions
        import math
        side = int(math.sqrt(features))
        if side * side == features:
            EXPECTED_HEIGHT = EXPECTED_WIDTH = side
            EXPECTED_CHANNELS = 1  # Grayscale
            logger.info(f"Dense model detected: {features} features (likely {side}x{side} grayscale)")
        else:
            rgb_side = int(math.sqrt(features / 3))
            if rgb_side * rgb_side * 3 == features:
                EXPECTED_HEIGHT = EXPECTED_WIDTH = rgb_side
                EXPECTED_CHANNELS = 3
                logger.info(f"Dense model detected: {features} features (likely {rgb_side}x{rgb_side} RGB)")
            else:
                logger.warning(f"Cannot infer image dimensions from {features} features")
                EXPECTED_HEIGHT = EXPECTED_WIDTH = 224  # Default
                EXPECTED_CHANNELS = 3
    else:
        raise ValueError(f"Unsupported model input shape: {input_shape}")
        
except Exception as e:
    logger.critical(
        f"Cannot load model at {MODEL_PATH}. Exception: {type(e).__name__}: {e}",
        exc_info=True
    )
    raise SystemExit(1)

# ─── Helpers ─────────────────────────────────────────────────────────────────
def preprocess_image(bgr_img: np.ndarray) -> np.ndarray:
    """
    Preprocess image to match model expectations.
    """
    try:
        # Convert BGR to RGB (if needed)
        if EXPECTED_CHANNELS == 3:
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        else:
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        
        # Resize to expected dimensions
        resized = cv2.resize(rgb_img, (EXPECTED_WIDTH, EXPECTED_HEIGHT))
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        if MODEL_TYPE == "CNN":
            # Keep spatial structure for CNN
            if EXPECTED_CHANNELS == 1:
                # Add channel dimension for grayscale
                processed = np.expand_dims(normalized, axis=-1)
            else:
                processed = normalized
            # Add batch dimension
            processed = np.expand_dims(processed, axis=0)
        else:  # DENSE
            # Flatten for dense model
            processed = normalized.flatten()
            processed = np.expand_dims(processed, axis=0)
        
        logger.info(f"Preprocessed image shape: {processed.shape}")
        return processed
        
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        raise

def extract_first_frame(stream) -> np.ndarray:
    """
    Save video stream to temp file and grab the first frame.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(stream.read())
        tmp.flush()
        path = tmp.name

    cap = cv2.VideoCapture(path)
    ret, frame = cap.read()
    cap.release()
    os.unlink(path)

    if not ret or frame is None:
        raise ValueError("Could not read first frame from video")
    return frame

# ─── Routes ──────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health():
    return "🚀 Deepfake API is up!", 200

@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    try:
        logger.info(f"Processing file: {file.filename} ({file.mimetype})")
        
        if file.mimetype.startswith("image/"):
            data = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Cannot decode image")
        elif file.mimetype.startswith("video/"):
            img = extract_first_frame(file.stream)
        else:
            return jsonify({"error": f"Unsupported type: {file.mimetype}"}), 400

        # Preprocess image
        processed_img = preprocess_image(img)
        
        # Make prediction
        pred = model.predict(processed_img, verbose=0)
        
        # Handle different output formats
        if len(pred.shape) == 2 and pred.shape[1] == 1:
            # Binary classification: (1, 1)
            confidence_score = float(pred[0][0])
        elif len(pred.shape) == 2 and pred.shape[1] > 1:
            # Multi-class: (1, n_classes)
            confidence_score = float(np.max(pred[0]))
        else:
            # Single value
            confidence_score = float(pred[0])
        
        confidence = round(confidence_score * 100, 2)
        is_real = bool(confidence_score > 0.5)

        logger.info(f"Prediction: {confidence}% confidence, is_real: {is_real}")
        
        return jsonify({
            "confidence": confidence, 
            "is_real": is_real,
            "model_type": MODEL_TYPE,
            "processed_shape": processed_img.shape
        }), 200

    except Exception as e:
        logger.error(f"Analysis error: {type(e).__name__}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Starting Flask application...")
    for rule in app.url_map.iter_rules():
        logger.info(f"Route {rule} Methods {rule.methods}")
    app.run(host="0.0.0.0", port=5000, debug=True)
