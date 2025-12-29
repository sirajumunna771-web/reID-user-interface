
import os
import sys
import io
# Fix Windows encoding issues
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import cv2
import glob
import json
import numpy as np
from numpy.linalg import norm
from flask import Flask, render_template, send_from_directory, jsonify, request, Response
from functools import wraps
from datetime import datetime
from PIL import Image
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# FAISS for efficient similarity search
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️ FAISS not available. Install with: pip install faiss-cpu (or faiss-gpu)")
# DeepFace emotion detection support
############################################
# PATH SETUP
############################################
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CROPPED_DIR = r"C:\Users\USER\Downloads\newstr\snapshots"
STORED_IMAGES_FOLDER = os.path.join(BASE_DIR, "stored_images")
METADATA_FILE = os.path.join(BASE_DIR, "stored_metadata.json")
GALLERY_FILE = os.path.join(BASE_DIR, "process_gallery.json")
os.makedirs(STORED_IMAGES_FOLDER, exist_ok=True)
def safe_load_json(filepath, default=None):
    """
    Safely load JSON file, handling empty files and invalid JSON.
    Returns default value if file doesn't exist or is invalid.
    """
    if default is None:
        default = []
   
    if not os.path.exists(filepath):
        return default
   
    try:
        # Check if file is empty
        if os.path.getsize(filepath) == 0:
            return default
       
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return default
            return json.loads(content)
    except (json.JSONDecodeError, ValueError, IOError) as e:
        print(f"⚠️ Warning: Could not load JSON from {filepath}: {str(e)}")
        print(f" Initializing with default value: {default}")
        # Try to backup corrupted file
        try:
            backup_path = filepath + ".backup"
            if os.path.exists(filepath):
                import shutil
                shutil.copy2(filepath, backup_path)
                print(f" Backup saved to: {backup_path}")
        except:
            pass
        return default
def convert_to_json_serializable(obj):
    """
    Convert numpy types and other non-serializable objects to JSON-serializable types.
    Compatible with NumPy 2.0+ (removed np.float_, np.int_, np.bool_).
    """
    # Check for numpy types using duck typing to avoid NumPy 2.0 compatibility issues
    if hasattr(obj, 'dtype'):
        # NumPy array or scalar
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # NumPy scalar types
        dtype = obj.dtype if hasattr(obj, 'dtype') else type(obj)
        if np.issubdtype(dtype, np.integer):
            return int(obj)
        elif np.issubdtype(dtype, np.floating):
            return float(obj)
        elif np.issubdtype(dtype, np.bool_):
            return bool(obj)
        else:
            # Fallback: try to convert
            try:
                return float(obj) if np.issubdtype(dtype, np.floating) else int(obj)
            except:
                return str(obj)
    # Handle native Python types
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, set):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj
def safe_save_json(filepath, data):
    """Safely save JSON data to file, converting numpy types to native Python types."""
    try:
        # Convert numpy types to native Python types
        serializable_data = convert_to_json_serializable(data)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ Error saving JSON to {filepath}: {str(e)}")
        raise
# Initialize metadata file if it doesn't exist or is invalid
if not os.path.exists(METADATA_FILE):
    safe_save_json(METADATA_FILE, [])
else:
    # Check if file is empty or invalid, and fix it
    try:
        if os.path.getsize(METADATA_FILE) == 0:
            safe_save_json(METADATA_FILE, [])
        else:
            # Try to load it to see if it's valid
            safe_load_json(METADATA_FILE, [])
    except:
        safe_save_json(METADATA_FILE, [])
############################################
# PRODUCTION-LEVEL RE-IDENTIFICATION (REID) SYSTEM
############################################

# Configuration - Production Security: Strict matching for same person detection
# COSINE DISTANCE >0.45 means COSINE SIMILARITY >0.55 (distance = 1 - similarity)
REID_CONFIG = {
    # # ULTRA-STRICT THRESHOLDS for maximum accuracy (prevents false positives)
    # "cosine_threshold": 0.72,           # RAISED: Minimum cosine similarity for duplicate match (ULTRA-STRICT: >0.72)
    # "high_confidence_threshold": 0.88,  # RAISED: High confidence duplicate threshold (was 0.85)
    # "medium_confidence_threshold": 0.80, # RAISED: Medium confidence threshold (was 0.75)
    # "low_confidence_threshold": 0.75,   # RAISED: Minimum threshold for low confidence matches (was 0.70)
    # "unique_face_threshold": 0.30,       # Below this = unique face (new person)
    
    "cosine_threshold": 0.65,           # RAISED: Minimum cosine similarity for duplicate match (ULTRA-STRICT: >0.72)
    "high_confidence_threshold": 0.82,  # RAISED: High confidence duplicate threshold (was 0.85)
    "medium_confidence_threshold": 0.80, # RAISED: Medium confidence threshold (was 0.75)
    "low_confidence_threshold": 0.70,   # RAISED: Minimum threshold for low confidence matches (was 0.70)
    "unique_face_threshold": 0.30,
    
    
    # EMBEDDING & QUALITY SETTINGS
    "max_embeddings_per_person": 15,    # INCREASED: More embeddings = better gallery quality
    "embedding_dim": 522,               # Enhanced embedding dimension (512 face + 10 landmarks)
    "face_embedding_dim": 512,           # Original face embedding dimension (ArcFace/InsightFace)
    "landmark_dim": 10,                  # Normalized landmark features (5 keypoints × 2 coordinates)
    "min_embedding_norm": 0.15,         # RAISED: Minimum embedding norm for quality check (was 0.1)
    
    # MATCHING IMPROVEMENTS
    "top_k_matches": 30,                # INCREASED: More candidates for better matching (was 20)
    "gallery_voting": True,              # Use gallery voting for better accuracy
    "use_percentile_voting": True,      # NEW: Use percentile-based voting (more robust)
    "percentile_weight": 0.75,          # NEW: 75th percentile weight (ignores outliers)
    "use_landmark_enhancement": True,    # Enable facial landmark enhancement for ReID
    
    # QUALITY FILTERING
    "reject_low_confidence": True,       # Reject low confidence matches even if above base threshold
    "require_min_avg_similarity": True,  # Require minimum average similarity in gallery voting
    "min_avg_similarity_threshold": 0.60, # RAISED: Minimum average similarity (was 0.50)
    "min_face_size": 40,                # NEW: Minimum face size in pixels (filter small/blurry faces)
    "min_detection_score": 0.5,         # NEW: Minimum face detection confidence score
    
    # ADVANCED MATCHING
    "require_consensus": True,           # NEW: Require multiple embeddings to agree
    "consensus_threshold": 0.65,         # NEW: Minimum similarity for consensus voting
    "gallery_consistency_weight": 0.8,   # NEW: Weight for gallery consistency check
    "use_median_similarity": True,       # NEW: Use median instead of mean (more robust)
    
    # ADAFACE & ENHANCEMENT SETTINGS
    "use_adaface": True,                 # NEW: Use AdaFace embedding (quality-weighted)
    "use_face_enhancement": True,        # NEW: Enhance faces before ReID (GFPGAN/Retinex)
    "use_blur_correction": True,         # NEW: Apply blur correction kernel
    "enhancement_method": "retinex",     # NEW: "retinex" or "gfpgan" (retinex is faster)
    "blur_correction_strength": 1.5,     # NEW: Blur correction kernel strength
    
    # DYNAMIC THRESHOLD SETTINGS
    "use_dynamic_threshold": True,       # NEW: Quality-aware dynamic thresholding
    "quality_threshold": 8.9,           # NEW: Quality score threshold for dynamic adjustment
    "low_quality_threshold": 0.22,        # NEW: Threshold for low-quality faces (< 8.5)
    "high_quality_threshold": 0.75,     # NEW: Threshold for high-quality faces (>= 8.5)
}

# Database: Store multiple embeddings per person (gallery-based approach)
# Structure: {person_id: [embedding1, embedding2, ...]}
FACE_DB = {}

# FAISS Index for efficient similarity search
FAISS_INDEX = None  # Will be initialized when first embedding is added
FAISS_INDEX_PERSON_MAP = []  # Maps FAISS index position to person_id
FAISS_INDEX_EMBEDDING_MAP = []  # Maps FAISS index position to embedding metadata

REID_STATS = {
    "total_matches": 0,
    "total_new_persons": 0,
    "total_embeddings": 0,
    "avg_similarity": 0.0
}

def normalize_embedding(embedding):
    """
    L2 normalize embedding for better cosine similarity calculation.
    This ensures embeddings are on the unit sphere.
    """
    if embedding is None or len(embedding) == 0:
        return None
    emb = np.array(embedding, dtype=np.float32)
    emb_norm = norm(emb)
    if emb_norm < REID_CONFIG["min_embedding_norm"]:
        return None  # Invalid embedding
    return emb / emb_norm

def cosine_similarity_production(v1, v2):
    """
    Production-level cosine similarity calculation.
    Assumes both vectors are already L2 normalized.
    Returns similarity in range [-1, 1], typically [0, 1] for normalized embeddings.
    """
    if v1 is None or v2 is None:
        return 0.0
    try:
        # For normalized vectors, cosine similarity = dot product
        similarity = float(np.dot(v1, v2))
        # Clamp to valid range
        similarity = max(-1.0, min(1.0, similarity))
        return similarity
    except Exception as e:
        print(f"⚠️ Cosine similarity error: {e}")
        return 0.0

############################################
# FACE ENHANCEMENT & QUALITY IMPROVEMENT
############################################

def calculate_face_quality_score(face_image):
    """
    Calculate face quality score (0-10) based on blur, brightness, contrast.
    Higher score = better quality.
    Used for dynamic thresholding.
    """
    try:
        if face_image is None or face_image.size == 0:
            return 0.0
        
        # Convert to grayscale for analysis
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = face_image
        
        # 1. Blur detection using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(laplacian_var / 100.0, 1.0)  # Normalize to 0-1
        
        # 2. Brightness assessment
        mean_brightness = np.mean(gray)
        brightness_score = 1.0 - abs(mean_brightness - 127.5) / 127.5  # Optimal around 127.5
        
        # 3. Contrast assessment
        contrast = np.std(gray)
        contrast_score = min(contrast / 50.0, 1.0)  # Normalize
        
        # 4. Face size factor (larger faces = better)
        face_area = gray.shape[0] * gray.shape[1]
        size_score = min(face_area / 10000.0, 1.0)  # Normalize to 100x100 = 1.0
        
        # Weighted combination
        quality_score = (
            0.4 * blur_score +
            0.2 * brightness_score +
            0.2 * contrast_score +
            0.2 * size_score
        ) * 10.0  # Scale to 0-10
        
        return float(np.clip(quality_score, 0.0, 10.0))
    except Exception as e:
        print(f"⚠️ Error calculating face quality: {e}")
        return 5.0  # Default medium quality

def apply_blur_correction(face_image, strength=1.5):
    """
    Apply blur correction using sharpening kernel.
    Restores eye corners, chin edges, nose ridges.
    """
    try:
        if face_image is None or face_image.size == 0:
            return face_image
        
        # Unsharp masking kernel for sharpening
        gaussian = cv2.GaussianBlur(face_image, (0, 0), strength)
        sharpened = cv2.addWeighted(face_image, 1.5, gaussian, -0.5, 0)
        
        # Alternative: Laplacian sharpening kernel
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])
        sharpened2 = cv2.filter2D(face_image, -1, kernel)
        
        # Combine both methods
        result = cv2.addWeighted(sharpened, 0.6, sharpened2, 0.4, 0)
        
        # Clip to valid range
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result
    except Exception as e:
        print(f"⚠️ Error applying blur correction: {e}")
        return face_image

def enhance_face_retinex(face_image):
    """
    Enhance face using Retinex algorithm (lighting normalization).
    Improves low-light and uneven lighting conditions.
    More stable than GFPGAN for real-time processing.
    """
    try:
        if face_image is None or face_image.size == 0:
            return face_image
        
        # Convert to float for processing
        img_float = face_image.astype(np.float32) / 255.0
        
        # Multi-scale Retinex (simplified version)
        enhanced = np.zeros_like(img_float)
        
        # Process each channel
        for c in range(img_float.shape[2]):
            channel = img_float[:, :, c]
            
            # Gaussian blur at different scales
            g1 = cv2.GaussianBlur(channel, (15, 15), 0)
            g2 = cv2.GaussianBlur(channel, (25, 25), 0)
            
            # Retinex: log(original) - log(blurred)
            log_orig = np.log1p(channel)
            log_g1 = np.log1p(g1 + 1e-6)
            log_g2 = np.log1p(g2 + 1e-6)
            
            # Combine scales
            retinex = (log_orig - 0.5 * log_g1 - 0.5 * log_g2)
            
            # Normalize and enhance
            retinex = (retinex - retinex.min()) / (retinex.max() - retinex.min() + 1e-6)
            retinex = np.power(retinex, 0.8)  # Gamma correction
            
            enhanced[:, :, c] = retinex
        
        # Convert back to uint8
        enhanced = (enhanced * 255.0).astype(np.uint8)
        
        # Apply slight contrast enhancement
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=5)
        
        return enhanced
    except Exception as e:
        print(f"⚠️ Error in Retinex enhancement: {e}")
        return face_image

def enhance_face_pipeline(face_image, use_enhancement=True, use_blur_correction=True):
    """
    Complete face enhancement pipeline.
    Applies blur correction and Retinex enhancement.
    """
    if face_image is None or face_image.size == 0:
        return face_image
    
    enhanced = face_image.copy()
    
    # Step 1: Blur correction (restores sharp features)
    if use_blur_correction and REID_CONFIG.get("use_blur_correction", True):
        strength = REID_CONFIG.get("blur_correction_strength", 1.5)
        enhanced = apply_blur_correction(enhanced, strength)
    
    # Step 2: Face enhancement (lighting normalization)
    if use_enhancement and REID_CONFIG.get("use_face_enhancement", True):
        method = REID_CONFIG.get("enhancement_method", "retinex")
        if method == "retinex":
            enhanced = enhance_face_retinex(enhanced)
        # Future: Add GFPGAN support here
    
    return enhanced

def extract_adaface_embedding(face_image, insightface_embedding):
    """
    Extract AdaFace-style embedding with quality-weighted margin.
    Since we're using InsightFace, we simulate AdaFace by:
    1. Calculating face quality
    2. Adjusting embedding based on quality
    3. Applying quality-weighted normalization
    """
    try:
        if face_image is None or insightface_embedding is None:
            return insightface_embedding
        
        # Calculate face quality
        quality_score = calculate_face_quality_score(face_image)
        quality_factor = quality_score / 10.0  # Normalize to 0-1
        
        # AdaFace: Quality-weighted margin adjustment
        # Higher quality = more confident embedding
        # Lower quality = more conservative (closer to mean)
        
        embedding = np.array(insightface_embedding, dtype=np.float32)
        
        # Normalize embedding
        emb_norm = np.linalg.norm(embedding)
        if emb_norm > 0:
            embedding = embedding / emb_norm
        
        # Quality-weighted adjustment
        # For low quality: pull towards zero (less confident)
        # For high quality: keep original (more confident)
        if quality_factor < 0.5:  # Low quality
            # Reduce embedding magnitude based on quality
            embedding = embedding * (0.5 + quality_factor)
        
        # Re-normalize
        emb_norm = np.linalg.norm(embedding)
        if emb_norm > 0:
            embedding = embedding / emb_norm
        
        return embedding.astype(np.float32)
    except Exception as e:
        print(f"⚠️ Error in AdaFace embedding: {e}")
        return insightface_embedding

def get_dynamic_threshold(quality_score):
    """
    Dynamic quality-aware threshold.
    Low-quality faces need lower threshold (more permissive).
    High-quality faces use strict threshold.
    """
    if not REID_CONFIG.get("use_dynamic_threshold", True):
        return REID_CONFIG["cosine_threshold"]
    
    quality_threshold = REID_CONFIG.get("quality_threshold", 8.5)
    low_threshold = REID_CONFIG.get("low_quality_threshold", 0.25)
    high_threshold = REID_CONFIG.get("high_quality_threshold", 0.72)
    
    if quality_score < quality_threshold:
        # Low quality: use lower threshold
        return low_threshold
    else:
        # High quality: use strict threshold
        return high_threshold

def extract_facial_landmarks(insightface_face_obj):
    """
    Extract facial landmarks/keypoints from InsightFace face object.
    InsightFace provides 5 keypoints: left eye, right eye, nose, left mouth, right mouth.
    
    Returns:
        landmarks: Dict with landmark coordinates or None if not available
    """
    if insightface_face_obj is None:
        return None
    
    try:
        # InsightFace provides kps (keypoints) as 5x2 array
        if hasattr(insightface_face_obj, 'kps') and insightface_face_obj.kps is not None:
            kps = insightface_face_obj.kps
            if len(kps) == 5 and len(kps[0]) == 2:
                landmarks = {
                    "left_eye": [float(kps[0][0]), float(kps[0][1])],
                    "right_eye": [float(kps[1][0]), float(kps[1][1])],
                    "nose": [float(kps[2][0]), float(kps[2][1])],
                    "left_mouth": [float(kps[3][0]), float(kps[3][1])],
                    "right_mouth": [float(kps[4][0]), float(kps[4][1])]
                }
                return landmarks
    except Exception as e:
        print(f"⚠️ Error extracting landmarks: {e}")
    
    return None

def normalize_landmarks(landmarks_dict, bbox, img_width, img_height):
    """
    Normalize landmark coordinates by bounding box and image dimensions for scale invariance.
    
    Args:
        landmarks_dict: Dict with landmark coordinates
        bbox: [x1, y1, x2, y2] bounding box
        img_width: Image width
        img_height: Image height
    
    Returns:
        normalized_landmarks: 10-dim numpy array [x1, y1, x2, y2, ..., x5, y5]
    """
    if landmarks_dict is None or bbox is None:
        return np.zeros(10, dtype=np.float32)
    
    try:
        x1, y1, x2, y2 = bbox
        bbox_width = max(x2 - x1, 1.0)
        bbox_height = max(y2 - y1, 1.0)
        
        # Extract coordinates in order: left_eye, right_eye, nose, left_mouth, right_mouth
        coords = [
            landmarks_dict.get("left_eye", [0, 0]),
            landmarks_dict.get("right_eye", [0, 0]),
            landmarks_dict.get("nose", [0, 0]),
            landmarks_dict.get("left_mouth", [0, 0]),
            landmarks_dict.get("right_mouth", [0, 0])
        ]
        
        # Normalize: (coord - bbox_center) / bbox_size for scale invariance
        normalized = []
        bbox_center_x = (x1 + x2) / 2.0
        bbox_center_y = (y1 + y2) / 2.0
        bbox_size = max(bbox_width, bbox_height)
        
        for x, y in coords:
            norm_x = (x - bbox_center_x) / bbox_size if bbox_size > 0 else 0.0
            norm_y = (y - bbox_center_y) / bbox_size if bbox_size > 0 else 0.0
            normalized.extend([norm_x, norm_y])
        
        return np.array(normalized, dtype=np.float32)
    except Exception as e:
        print(f"⚠️ Error normalizing landmarks: {e}")
        return np.zeros(10, dtype=np.float32)

def create_enhanced_embedding(face_embedding, landmarks_dict, bbox, img_width, img_height):
    """
    Create enhanced embedding by concatenating face embedding with normalized facial landmarks.
    This improves ReID accuracy by incorporating facial structure information.
    
    Args:
        face_embedding: 512-dim face embedding (numpy array)
        landmarks_dict: Dict with landmark coordinates
        bbox: [x1, y1, x2, y2] bounding box
        img_width: Image width
        img_height: Image height
    
    Returns:
        enhanced_embedding: 522-dim numpy array (512 face + 10 landmarks)
    """
    try:
        # Normalize face embedding
        face_emb = np.array(face_embedding, dtype=np.float32)
        face_norm = normalize_embedding(face_emb)
        if face_norm is None:
            face_norm = np.zeros(REID_CONFIG["face_embedding_dim"], dtype=np.float32)
        
        # Normalize landmarks
        landmark_features = normalize_landmarks(landmarks_dict, bbox, img_width, img_height)
        
        # Concatenate: face embedding + landmark features
        enhanced_emb = np.concatenate([face_norm, landmark_features])
        
        # Normalize the combined vector
        enhanced_norm = normalize_embedding(enhanced_emb)
        
        return enhanced_norm if enhanced_norm is not None else enhanced_emb
    except Exception as e:
        print(f"⚠️ Error creating enhanced embedding: {e}")
        # Fallback to original face embedding
        face_emb = np.array(face_embedding, dtype=np.float32)
        return normalize_embedding(face_emb) if normalize_embedding(face_emb) is not None else face_emb

def initialize_faiss_index(embedding_dim=522):
    """
    Initialize FAISS index for efficient similarity search.
    Uses Inner Product (IP) index for cosine similarity with normalized vectors.
    """
    global FAISS_INDEX, FAISS_INDEX_PERSON_MAP, FAISS_INDEX_EMBEDDING_MAP
    
    if not FAISS_AVAILABLE:
        print("⚠️ FAISS not available, falling back to basic matching")
        return False
    
    try:
        # Use IndexFlatIP (Inner Product) for cosine similarity with normalized vectors
        # For normalized vectors, inner product = cosine similarity
        FAISS_INDEX = faiss.IndexFlatIP(embedding_dim)
        FAISS_INDEX_PERSON_MAP = []
        FAISS_INDEX_EMBEDDING_MAP = []
        print(f"✅ FAISS index initialized (dim={embedding_dim})")
        return True
    except Exception as e:
        print(f"⚠️ Error initializing FAISS index: {e}")
        return False

def rebuild_faiss_index():
    """
    Rebuild FAISS index from current FACE_DB.
    This is called when loading from disk or when index needs to be refreshed.
    """
    global FAISS_INDEX, FAISS_INDEX_PERSON_MAP, FAISS_INDEX_EMBEDDING_MAP
    
    if not FAISS_AVAILABLE:
        return False
    
    try:
        # Count total embeddings
        total_embeddings = sum(len(emb_list) for emb_list in FACE_DB.values())
        if total_embeddings == 0:
            # Initialize empty index
            initialize_faiss_index(REID_CONFIG["embedding_dim"])
            return True
        
        # Get embedding dimension from first embedding
        first_person = next(iter(FACE_DB.values()))
        first_emb = first_person[0]
        emb_dim = len(first_emb) if isinstance(first_emb, (list, np.ndarray)) else REID_CONFIG["embedding_dim"]
        
        # Initialize index
        initialize_faiss_index(emb_dim)
        
        # Add all embeddings to index
        FAISS_INDEX_PERSON_MAP = []
        FAISS_INDEX_EMBEDDING_MAP = []
        
        for person_id, embeddings_list in FACE_DB.items():
            for emb_data in embeddings_list:
                # Handle both old format (list) and new format (dict)
                if isinstance(emb_data, dict):
                    emb = emb_data.get("embedding", [])
                else:
                    emb = emb_data
                
                normalized_emb = normalize_embedding(emb)
                if normalized_emb is not None:
                    emb_array = np.array([normalized_emb], dtype=np.float32)
                    FAISS_INDEX.add(emb_array)
                    FAISS_INDEX_PERSON_MAP.append(person_id)
                    FAISS_INDEX_EMBEDDING_MAP.append({
                        "person_id": person_id,
                        "embedding": normalized_emb.tolist()
                    })
        
        print(f"✅ FAISS index rebuilt: {FAISS_INDEX.ntotal} embeddings from {len(FACE_DB)} persons")
        return True
    except Exception as e:
        print(f"⚠️ Error rebuilding FAISS index: {e}")
        return False

def match_person_production(new_embedding, threshold=None, use_faiss=True):
    """
    Production-level ReID matching for UNIQUE vs DUPLICATE face detection.
    Based on ReID research: Uses gallery-based voting and top-k candidates.
    
    Returns:
        matched_id: Person ID if DUPLICATE found, None if UNIQUE face
        similarity: Best similarity score (0.0 to 1.0)
        confidence: "high", "medium", or "low"
        match_details: Dict with face_type ("unique" or "duplicate")
    """
    if threshold is None:
        threshold = REID_CONFIG["cosine_threshold"]
    
    # Normalize the query embedding
    normalized_query = normalize_embedding(new_embedding)
    if normalized_query is None:
        return None, 0.0, "low", {"error": "Invalid embedding", "face_type": "unknown"}
    
    if not FACE_DB or len(FACE_DB) == 0:
        # Empty database = UNIQUE face
        return None, 0.0, "low", {"reason": "Empty database", "face_type": "unique"}
    
    # Use FAISS if available and enabled
    if FAISS_AVAILABLE and use_faiss and FAISS_INDEX is not None and FAISS_INDEX.ntotal > 0:
        try:
            return match_person_faiss(normalized_query, threshold)
        except Exception as e:
            print(f"⚠️ FAISS search error: {e}, falling back to basic matching")
            return match_person_basic(normalized_query, threshold)
    
    # Fallback to basic matching
    return match_person_basic(normalized_query, threshold)

def match_person_faiss(normalized_query, threshold):
    """
    FAISS-powered matching with gallery voting for robust duplicate detection.
    """
    # Prepare query vector
    query_vector = np.array([normalized_query], dtype=np.float32)
    
    # Search for top-k candidates (gallery voting approach) - increased for better matching
    k = min(max(REID_CONFIG["top_k_matches"], 20), FAISS_INDEX.ntotal)  # Check more candidates
    similarities, indices = FAISS_INDEX.search(query_vector, k)
    
    if len(similarities[0]) == 0:
        return None, 0.0, "low", {"reason": "No results from FAISS search", "face_type": "unique"}
    
    # Check ALL top candidates and find the best person match (not just first match)
    best_id = None
    best_sim = -1.0
    best_person_avg_sim = -1.0
    best_person_max_sim = -1.0
    
    # Group by person_id and find best match per person
    person_candidates = {}
    for i, idx in enumerate(indices[0]):
        if idx >= len(FAISS_INDEX_PERSON_MAP):
                            continue
        candidate_id = FAISS_INDEX_PERSON_MAP[idx]
        candidate_sim = float(similarities[0][i])
        
        if candidate_id not in person_candidates:
            person_candidates[candidate_id] = []
        person_candidates[candidate_id].append(candidate_sim)
    
    # For each person, calculate gallery voting similarity
    for person_id, sims in person_candidates.items():
        if not sims:
            continue
        
        # Get all embeddings for this person for gallery voting
        person_embeddings = FACE_DB.get(person_id, [])
        person_sims = []
        for emb_data in person_embeddings:
            if isinstance(emb_data, dict):
                emb = emb_data.get("embedding", [])
            else:
                emb = emb_data
            
            norm_emb = normalize_embedding(emb)
            if norm_emb is not None:
                sim = cosine_similarity_production(normalized_query, norm_emb)
                person_sims.append(sim)
        
        # ENHANCED Gallery voting: multiple strategies for robustness
        all_sims = sims + person_sims if person_sims else sims
        
        if not all_sims or len(all_sims) == 0:
            continue
        
        max_sim = float(np.max(all_sims))
        avg_sim = float(np.mean(all_sims))
        median_sim = float(np.median(all_sims))
        min_sim = float(np.min(all_sims))
        
        # Percentile-based voting (ignores outliers)
        percentile_weight = REID_CONFIG.get("percentile_weight", 0.75)
        percentile_sim = float(np.percentile(all_sims, percentile_weight * 100))
        
        # IMPROVED: Adaptive similarity calculation based on gallery quality
        gallery_size = len(all_sims)
        
        # For small galleries (1-3 embeddings), rely more on max similarity
        # For larger galleries (4+ embeddings), use more robust statistics
        if gallery_size <= 3:
            # Small gallery: prioritize max similarity but still consider consistency
            consistency_bonus = 0.1 * (max_sim - min_sim) if max_sim > min_sim else 0
            combined_sim = 0.75 * max_sim + 0.15 * avg_sim + 0.10 * median_sim + consistency_bonus
        else:
            # Large gallery: use robust statistics (median, percentile) to avoid outliers
            if REID_CONFIG.get("use_median_similarity", True):
                base_sim = median_sim
            else:
                base_sim = avg_sim
            
            # Multi-factor weighted scoring with improved weights
            if REID_CONFIG.get("use_percentile_voting", True):
                # Improved: max (best match), percentile (robust), median/avg (overall)
                # Adjusted weights for better balance with 0.55 threshold
                combined_sim = 0.45 * max_sim + 0.35 * percentile_sim + 0.20 * base_sim
            else:
                # Traditional: max + avg with consistency check
                consistency_penalty = 0.05 * (max_sim - min_sim) if (max_sim - min_sim) > 0.3 else 0
                combined_sim = 0.65 * max_sim + 0.35 * avg_sim - consistency_penalty
        
        # Quality boost: If all similarities are consistently high, boost the score
        if min_sim >= threshold * 0.9 and gallery_size >= 2:
            consistency_boost = 0.05 * (1.0 - (max_sim - min_sim))
            combined_sim = min(1.0, combined_sim + consistency_boost)
        
        # Track best match
        if combined_sim > best_sim:
            best_sim = combined_sim
            best_id = person_id
            best_person_avg_sim = avg_sim
            best_person_max_sim = max_sim
    
    if best_id is None:
        return None, 0.0, "low", {"reason": "No valid person match found", "face_type": "unique"}
    
    # Final similarity already calculated in the loop above
    final_similarity = best_sim
    
    # Determine face type: UNIQUE or DUPLICATE
    # STRICT TWO-STAGE VALIDATION: Must pass BOTH similarity threshold AND confidence requirements
    # This prevents false positives (different people getting same ID)
    
    # Stage 1: Check base similarity threshold
    if final_similarity > threshold:
        # Stage 2: Determine confidence level and apply confidence-based thresholds
        if final_similarity >= REID_CONFIG["high_confidence_threshold"]:
            confidence = "high"
            min_required_sim = REID_CONFIG["high_confidence_threshold"]
        elif final_similarity >= REID_CONFIG["medium_confidence_threshold"]:
            confidence = "medium"
            min_required_sim = REID_CONFIG["medium_confidence_threshold"]
        else:
            confidence = "low"
            # For low confidence, require higher threshold to prevent false positives
            min_required_sim = REID_CONFIG.get("low_confidence_threshold", 0.70)
        
        # CRITICAL: Reject low confidence matches even if above base threshold
        if REID_CONFIG.get("reject_low_confidence", True) and confidence == "low":
            if final_similarity < min_required_sim:
                # Reject: Below low confidence threshold
                return None, final_similarity, confidence, {
            "similarity": round(final_similarity, 4),
                    "reason": f"Low confidence match rejected (similarity {final_similarity:.4f} < required {min_required_sim:.4f})",
                    "threshold": threshold,
                    "min_required": min_required_sim,
                    "method": "FAISS",
                    "face_type": "unique",
                    "status": "UNIQUE FACE - Low confidence rejected"
                }
        
        # Stage 3: Check average similarity requirement (gallery consistency check)
        if REID_CONFIG.get("require_min_avg_similarity", True) and best_person_avg_sim > 0:
            min_avg_threshold = REID_CONFIG.get("min_avg_similarity_threshold", 0.60)
            if best_person_avg_sim < min_avg_threshold:
                # Reject: Average similarity too low (inconsistent with gallery)
                return None, final_similarity, confidence, {
                    "similarity": round(final_similarity, 4),
                    "max_similarity": round(best_person_max_sim, 4),
            "avg_similarity": round(best_person_avg_sim, 4),
                    "reason": f"Average gallery similarity too low ({best_person_avg_sim:.4f} < {min_avg_threshold:.4f}) - inconsistent match",
                    "threshold": threshold,
                    "method": "FAISS",
                    "face_type": "unique",
                    "status": "UNIQUE FACE - Inconsistent gallery match"
                }
        
        # Stage 4: Consensus validation (require multiple embeddings to agree)
        if REID_CONFIG.get("require_consensus", True) and best_id is not None:
            # IMPROVED: Adaptive consensus threshold based on base threshold
            base_consensus = REID_CONFIG.get("consensus_threshold", 0.65)
            # Adjust consensus threshold relative to base threshold (0.55)
            # For lower base thresholds, use proportionally lower consensus
            consensus_threshold = max(threshold * 0.95, base_consensus * (threshold / 0.70))
            person_embeddings = FACE_DB.get(best_id, [])
            
            if len(person_embeddings) > 1:
                # Count how many embeddings agree (above consensus threshold)
                consensus_count = 0
                consensus_sims = []
                
                for emb_data in person_embeddings:
                    if isinstance(emb_data, dict):
                        emb = emb_data.get("embedding", [])
                    else:
                        emb = emb_data
                    
                    norm_emb = normalize_embedding(emb)
                    if norm_emb is not None:
                        sim = cosine_similarity_production(normalized_query, norm_emb)
                        consensus_sims.append(sim)
                        if sim >= consensus_threshold:
                            consensus_count += 1
                
                # IMPROVED: Adaptive consensus requirement based on gallery size and quality
                # For small galleries (2-3), require at least 2 to agree
                # For larger galleries, require at least 50% but with quality check
                if len(person_embeddings) <= 3:
                    min_consensus = max(2, len(person_embeddings) - 1)  # At least 2 or all-1
                else:
                    # For larger galleries, require 50% but also check if avg consensus sim is reasonable
                    min_consensus = max(2, int(len(person_embeddings) * 0.5))
                    # Additional quality check: if avg consensus similarity is too low, be stricter
                    if consensus_sims:
                        avg_consensus_sim = np.mean(consensus_sims)
                        if avg_consensus_sim < threshold * 1.1:  # Require slightly above threshold
                            min_consensus = max(min_consensus, int(len(person_embeddings) * 0.6))
                
                if consensus_count < min_consensus:
                    # Reject: Not enough consensus from gallery
                    consensus_rate = consensus_count / len(person_embeddings) if person_embeddings else 0
                    return None, final_similarity, confidence, {
                        "similarity": round(final_similarity, 4),
                        "max_similarity": round(best_person_max_sim, 4),
                        "avg_similarity": round(best_person_avg_sim, 4),
                        "consensus_count": consensus_count,
                        "min_required": min_consensus,
                        "consensus_rate": round(consensus_rate, 2),
                        "consensus_threshold": round(consensus_threshold, 4),
                        "reason": f"Consensus validation failed ({consensus_count}/{len(person_embeddings)} embeddings agree)",
                        "threshold": threshold,
                        "method": "FAISS",
                        "face_type": "unique",
                        "status": "UNIQUE FACE - Insufficient consensus"
                    }
        
        # All checks passed - accept as duplicate
        face_type = "duplicate"
        match_details = {
            "similarity": round(final_similarity, 4),
            "max_similarity": round(best_person_max_sim, 4) if best_person_max_sim > 0 else round(final_similarity, 4),
            "avg_similarity": round(best_person_avg_sim, 4) if best_person_avg_sim > 0 else round(final_similarity, 4),
            "confidence": confidence,
            "gallery_size": len(person_embeddings),
            "method": "FAISS",
            "top_k": k,
            "face_type": face_type,
            "status": "DUPLICATE FACE"
        }
        REID_STATS["total_matches"] += 1
        REID_STATS["avg_similarity"] = (REID_STATS["avg_similarity"] * (REID_STATS["total_matches"] - 1) + final_similarity) / REID_STATS["total_matches"]
        return best_id, final_similarity, confidence, match_details
    else:
        # UNIQUE FACE - below threshold
        face_type = "unique"
        confidence = "low"
        return None, final_similarity, confidence, {
            "similarity": round(final_similarity, 4),
            "reason": "Below threshold - UNIQUE FACE",
            "threshold": threshold,
            "method": "FAISS",
            "face_type": face_type,
            "status": "UNIQUE FACE"
        }

def match_person_basic(normalized_query, threshold):
    """
    Basic person matching with gallery voting (fallback when FAISS is not available).
    Uses gallery-based approach: compares against all stored embeddings per person.
    Implements UNIQUE vs DUPLICATE face detection.
    """
    best_id = None
    best_sim = -1.0
    best_person_avg_sim = -1.0
    best_person_max_sim = -1.0
    
    # Compare against all persons in database
    for person_id, embeddings_list in FACE_DB.items():
        if not embeddings_list or len(embeddings_list) == 0:
            continue
        
        # Extract embeddings from dict structure
        normalized_gallery = []
        for emb_data in embeddings_list:
            # Handle both old format (list) and new format (dict)
            if isinstance(emb_data, dict):
                emb = emb_data.get("embedding", emb_data)
            else:
                emb = emb_data
            
            norm_emb = normalize_embedding(emb)
            if norm_emb is not None:
                normalized_gallery.append(norm_emb)
        
        if not normalized_gallery:
            continue
        
        # Compute similarities with all gallery embeddings
        similarities = []
        for gallery_emb in normalized_gallery:
            sim = cosine_similarity_production(normalized_query, gallery_emb)
            similarities.append(sim)
        
        if len(similarities) == 0:
            continue
        
        # ENHANCED Gallery voting: multiple strategies for robustness
        max_sim = float(np.max(similarities))
        avg_sim = float(np.mean(similarities))
        median_sim = float(np.median(similarities))
        min_sim = float(np.min(similarities))
        
        # Percentile-based voting (ignores outliers)
        percentile_weight = REID_CONFIG.get("percentile_weight", 0.75)
        percentile_sim = float(np.percentile(similarities, percentile_weight * 100))
        
        # IMPROVED: Adaptive similarity calculation based on gallery quality
        gallery_size = len(similarities)
        
        # For small galleries (1-3 embeddings), rely more on max similarity
        # For larger galleries (4+ embeddings), use more robust statistics
        if gallery_size <= 3:
            # Small gallery: prioritize max similarity but still consider consistency
            consistency_bonus = 0.1 * (max_sim - min_sim) if max_sim > min_sim else 0
            combined_sim = 0.75 * max_sim + 0.15 * avg_sim + 0.10 * median_sim + consistency_bonus
        else:
            # Large gallery: use robust statistics (median, percentile) to avoid outliers
            if REID_CONFIG.get("use_median_similarity", True):
                base_sim = median_sim
            else:
                base_sim = avg_sim
            
            # Multi-factor weighted scoring with improved weights
            if REID_CONFIG.get("use_percentile_voting", True):
                # Improved: max (best match), percentile (robust), median/avg (overall)
                # Adjusted weights for better balance with 0.55 threshold
                combined_sim = 0.45 * max_sim + 0.35 * percentile_sim + 0.20 * base_sim
            else:
                # Traditional: max + avg with consistency check
                consistency_penalty = 0.05 * (max_sim - min_sim) if (max_sim - min_sim) > 0.3 else 0
                combined_sim = 0.65 * max_sim + 0.35 * avg_sim - consistency_penalty
        
        # Quality boost: If all similarities are consistently high, boost the score
        if min_sim >= threshold * 0.9 and gallery_size >= 2:
            consistency_boost = 0.05 * (1.0 - (max_sim - min_sim))
            combined_sim = min(1.0, combined_sim + consistency_boost)
        
        # Track best match
        if combined_sim > best_sim:
            best_sim = combined_sim
            best_id = person_id
            best_person_avg_sim = avg_sim
            best_person_max_sim = max_sim
    
    # Determine face type: UNIQUE or DUPLICATE
    if best_sim < 0:
        return None, 0.0, "low", {"reason": "No matches found", "face_type": "unique", "status": "UNIQUE FACE"}
    
    # STRICT TWO-STAGE VALIDATION: Must pass BOTH similarity threshold AND confidence requirements
    # Check if match meets threshold (DUPLICATE vs UNIQUE)
    if best_sim > threshold:
        # Stage 2: Determine confidence level and apply confidence-based thresholds
        if best_sim >= REID_CONFIG["high_confidence_threshold"]:
            confidence = "high"
            min_required_sim = REID_CONFIG["high_confidence_threshold"]
        elif best_sim >= REID_CONFIG["medium_confidence_threshold"]:
            confidence = "medium"
            min_required_sim = REID_CONFIG["medium_confidence_threshold"]
        else:
            confidence = "low"
            # For low confidence, require higher threshold to prevent false positives
            min_required_sim = REID_CONFIG.get("low_confidence_threshold", 0.70)
        
        # CRITICAL: Reject low confidence matches even if above base threshold
        if REID_CONFIG.get("reject_low_confidence", True) and confidence == "low":
            if best_sim < min_required_sim:
                # Reject: Below low confidence threshold
                return None, best_sim, confidence, {
                    "similarity": round(best_sim, 4),
                    "reason": f"Low confidence match rejected (similarity {best_sim:.4f} < required {min_required_sim:.4f})",
                    "threshold": threshold,
                    "min_required": min_required_sim,
                    "method": "basic",
                    "face_type": "unique",
                    "status": "UNIQUE FACE - Low confidence rejected"
                }
        
        # Stage 3: Check average similarity requirement (gallery consistency check)
        if REID_CONFIG.get("require_min_avg_similarity", True) and best_person_avg_sim > 0:
            min_avg_threshold = REID_CONFIG.get("min_avg_similarity_threshold", 0.60)
            if best_person_avg_sim < min_avg_threshold:
                # Reject: Average similarity too low (inconsistent with gallery)
                return None, best_sim, confidence, {
                    "similarity": round(best_sim, 4),
                    "max_similarity": round(best_person_max_sim, 4),
                    "avg_similarity": round(best_person_avg_sim, 4),
                    "reason": f"Average gallery similarity too low ({best_person_avg_sim:.4f} < {min_avg_threshold:.4f}) - inconsistent match",
                    "threshold": threshold,
                    "method": "basic",
                    "face_type": "unique",
                    "status": "UNIQUE FACE - Inconsistent gallery match"
                }
        
        # Stage 4: Consensus validation (require multiple embeddings to agree)
        if REID_CONFIG.get("require_consensus", True) and best_id is not None:
            # IMPROVED: Adaptive consensus threshold based on base threshold
            base_consensus = REID_CONFIG.get("consensus_threshold", 0.65)
            # Adjust consensus threshold relative to base threshold (0.55)
            # For lower base thresholds, use proportionally lower consensus
            consensus_threshold = max(threshold * 0.95, base_consensus * (threshold / 0.70))
            person_embeddings = FACE_DB.get(best_id, [])
            
            if len(person_embeddings) > 1:
                # Count how many embeddings agree (above consensus threshold)
                consensus_count = 0
                consensus_sims = []
                
                for emb_data in person_embeddings:
                    if isinstance(emb_data, dict):
                        emb = emb_data.get("embedding", [])
                    else:
                        emb = emb_data
                    
                    norm_emb = normalize_embedding(emb)
                    if norm_emb is not None:
                        sim = cosine_similarity_production(normalized_query, norm_emb)
                        consensus_sims.append(sim)
                        if sim >= consensus_threshold:
                            consensus_count += 1
                
                # IMPROVED: Adaptive consensus requirement based on gallery size and quality
                # For small galleries (2-3), require at least 2 to agree
                # For larger galleries, require at least 50% but with quality check
                if len(person_embeddings) <= 3:
                    min_consensus = max(2, len(person_embeddings) - 1)  # At least 2 or all-1
                else:
                    # For larger galleries, require 50% but also check if avg consensus sim is reasonable
                    min_consensus = max(2, int(len(person_embeddings) * 0.5))
                    # Additional quality check: if avg consensus similarity is too low, be stricter
                    if consensus_sims:
                        avg_consensus_sim = np.mean(consensus_sims)
                        if avg_consensus_sim < threshold * 1.1:  # Require slightly above threshold
                            min_consensus = max(min_consensus, int(len(person_embeddings) * 0.6))
                
                if consensus_count < min_consensus:
                    # Reject: Not enough consensus from gallery
                    consensus_rate = consensus_count / len(person_embeddings) if person_embeddings else 0
                    return None, best_sim, confidence, {
                        "similarity": round(best_sim, 4),
                        "max_similarity": round(best_person_max_sim, 4),
                        "avg_similarity": round(best_person_avg_sim, 4),
                        "consensus_count": consensus_count,
                        "min_required": min_consensus,
                        "consensus_rate": round(consensus_rate, 2),
                        "consensus_threshold": round(consensus_threshold, 4),
                        "reason": f"Consensus validation failed ({consensus_count}/{len(person_embeddings)} embeddings agree)",
                        "threshold": threshold,
                        "method": "basic",
                        "face_type": "unique",
                        "status": "UNIQUE FACE - Insufficient consensus"
                    }
        
        # All checks passed - accept as duplicate
        match_details = {
            "similarity": round(best_sim, 4),
            "max_similarity": round(best_person_max_sim, 4),
            "avg_similarity": round(best_person_avg_sim, 4),
            "confidence": confidence,
            "gallery_size": len(FACE_DB.get(best_id, [])),
            "method": "basic",
            "face_type": "duplicate",
            "status": "DUPLICATE FACE"
        }
        REID_STATS["total_matches"] += 1
        REID_STATS["avg_similarity"] = (REID_STATS["avg_similarity"] * (REID_STATS["total_matches"] - 1) + best_sim) / REID_STATS["total_matches"]
        return best_id, best_sim, confidence, match_details
    else:
        # UNIQUE FACE - below threshold
        # Determine confidence level for reporting
        if best_sim >= REID_CONFIG["high_confidence_threshold"]:
            confidence = "high"
        elif best_sim >= REID_CONFIG["medium_confidence_threshold"]:
            confidence = "medium"
        else:
            confidence = "low"
        
        return None, best_sim, confidence, {
            "similarity": round(best_sim, 4),
            "reason": "Below threshold - UNIQUE FACE",
            "threshold": threshold,
            "method": "basic",
            "face_type": "unique",
            "status": "UNIQUE FACE"
        }

def save_new_embedding_production(person_id, embedding, landmarks=None):
    """
    Save embedding to person's gallery and update FAISS index.
    Maintains maximum number of embeddings per person (FIFO).
    
    Args:
        person_id: Person ID (e.g., "ID001")
        embedding: Face embedding vector
        landmarks: Optional facial landmarks dict
    """
    global FAISS_INDEX, FAISS_INDEX_PERSON_MAP, FAISS_INDEX_EMBEDDING_MAP
    
    normalized_emb = normalize_embedding(embedding)
    if normalized_emb is None:
        print(f"⚠️ Invalid embedding for {person_id}, skipping save")
        return False
    
    if person_id not in FACE_DB:
        FACE_DB[person_id] = []
    
    # Check if we need to remove old embedding (FIFO)
    max_emb = REID_CONFIG["max_embeddings_per_person"]
    old_emb_count = len(FACE_DB[person_id])
    
    # Add to gallery
    embedding_data = {
        "embedding": normalized_emb.tolist(),
        "landmarks": landmarks  # Store landmarks with embedding
    }
    FACE_DB[person_id].append(embedding_data)
    
    # Maintain max embeddings per person (FIFO)
    if len(FACE_DB[person_id]) > max_emb:
        FACE_DB[person_id] = FACE_DB[person_id][-max_emb:]
    
    # Update FAISS index
    if FAISS_AVAILABLE:
        try:
            # Initialize index if needed
            if FAISS_INDEX is None:
                emb_dim = len(normalized_emb)
                initialize_faiss_index(emb_dim)
            
            # If we removed an old embedding, we need to rebuild index
            # (FAISS doesn't support efficient deletion)
            if old_emb_count >= max_emb:
                # Rebuild index to remove old embeddings
                rebuild_faiss_index()
            else:
                # Add new embedding to index
                emb_array = np.array([normalized_emb], dtype=np.float32)
                FAISS_INDEX.add(emb_array)
                FAISS_INDEX_PERSON_MAP.append(person_id)
                FAISS_INDEX_EMBEDDING_MAP.append({
                    "person_id": person_id,
                    "embedding": normalized_emb.tolist(),
                    "landmarks": landmarks
                })
        except Exception as e:
            print(f"⚠️ Error updating FAISS index: {e}")
    
    REID_STATS["total_embeddings"] = sum(len(emb_list) for emb_list in FACE_DB.values())
    return True

def generate_pid():
    """Generate person IDs like ID001, ID002..."""
    if not FACE_DB:
        return "ID001"
    
    try:
        # Extract all numeric IDs and find max
        ids = []
        for pid in FACE_DB.keys():
            if pid and pid.startswith("ID") and len(pid) > 2:
                try:
                    num = int(pid[2:])
                    ids.append(num)
                except:
                    continue
        
        if ids:
            max_id = max(ids)
            return f"ID{max_id + 1:03d}"
        else:
            return "ID001"
    except Exception as e:
        print(f"⚠️ Error generating PID: {e}")
        return "ID001"

def load_face_db(metadata_path=METADATA_FILE):
    """
    Load face database from metadata file.
    Supports multiple embeddings per person (gallery-based).
    Rebuilds FAISS index after loading.
    """
    global FACE_DB, REID_STATS
    FACE_DB = {}
    REID_STATS = {
        "total_matches": 0,
        "total_new_persons": 0,
        "total_embeddings": 0,
        "avg_similarity": 0.0
    }
    
    if not os.path.exists(metadata_path):
        print("ℹ️ No existing metadata file, starting with empty ReID database")
        # Initialize empty FAISS index
        if FAISS_AVAILABLE:
            initialize_faiss_index(REID_CONFIG["embedding_dim"])
        return
    
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"⚠️ Error loading metadata: {e}")
        return
    
    loaded_count = 0
    for entry in data:
        faces = entry.get("faces", [])
        for face in faces:
            pid = face.get("person_id")
            emb = face.get("embedding")
            landmarks = face.get("landmarks")  # Load landmarks if available
            
            if pid and emb and len(emb) > 0:
                # Normalize and store
                normalized_emb = normalize_embedding(emb)
                if normalized_emb is not None:
                    if pid not in FACE_DB:
                        FACE_DB[pid] = []
                    
                    # Store with landmarks
                    embedding_data = {
                        "embedding": normalized_emb.tolist(),
                        "landmarks": landmarks
                    }
                    FACE_DB[pid].append(embedding_data)
                    loaded_count += 1
    
    # Maintain max embeddings per person
    for pid in FACE_DB:
        max_emb = REID_CONFIG["max_embeddings_per_person"]
        if len(FACE_DB[pid]) > max_emb:
            FACE_DB[pid] = FACE_DB[pid][-max_emb:]
    
    REID_STATS["total_embeddings"] = sum(len(emb_list) for emb_list in FACE_DB.values())
    print(f"✅ Loaded ReID database: {len(FACE_DB)} persons, {loaded_count} embeddings")
    
    # Rebuild FAISS index
    if FAISS_AVAILABLE and loaded_count > 0:
        rebuild_faiss_index()

def get_reid_stats():
    """Get ReID system statistics."""
    faiss_status = "enabled" if (FAISS_AVAILABLE and FAISS_INDEX is not None) else "disabled"
    faiss_count = FAISS_INDEX.ntotal if (FAISS_AVAILABLE and FAISS_INDEX is not None) else 0
    
    return {
        "total_persons": len(FACE_DB),
        "total_embeddings": REID_STATS["total_embeddings"],
        "total_matches": REID_STATS["total_matches"],
        "total_new_persons": REID_STATS["total_new_persons"],
        "avg_similarity": round(REID_STATS["avg_similarity"], 3),
        "config": REID_CONFIG,
        "faiss": {
            "available": FAISS_AVAILABLE,
            "status": faiss_status,
            "index_size": faiss_count
        }
    }

# Load database when server starts
load_face_db()

############################################
# LOAD ANTELOPEV2 (DETECTION + ALIGNMENT + EMBEDDING)
############################################
from insightface.app import FaceAnalysis
import shutil
from deepface import DeepFace

# ONNX Runtime removed - emotion detection disabled
print("🔥 Loading InsightFace antelopev2 model...")
face_app = None
model_name = "antelopev2"
ANTELOPEV2_DIR = os.path.join(BASE_DIR, "cladublazeface", "antelopev2")
INSIGHTFACE_MODEL_DIR = os.path.join(os.path.expanduser("~"), ".insightface", "models", "antelopev2")
has_local_models = os.path.exists(ANTELOPEV2_DIR) and any(
    f.endswith('.onnx') for f in os.listdir(ANTELOPEV2_DIR)
)
if has_local_models:
    print(f"📁 Found local antelopev2 models in: {ANTELOPEV2_DIR}")
    print(f"📦 Copying models to InsightFace directory: {INSIGHTFACE_MODEL_DIR}")
   
    try:
        os.makedirs(INSIGHTFACE_MODEL_DIR, exist_ok=True)
       
        onnx_files = [f for f in os.listdir(ANTELOPEV2_DIR) if f.endswith('.onnx')]
        for onnx_file in onnx_files:
            src = os.path.join(ANTELOPEV2_DIR, onnx_file)
            dst = os.path.join(INSIGHTFACE_MODEL_DIR, onnx_file)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                print(f" ✓ Copied {onnx_file}")
            else:
                print(f" ⊙ {onnx_file} already exists, skipping")
       
        print("✅ Local models copied successfully")
    except Exception as e:
        print(f"⚠️ Warning: Could not copy local models: {str(e)}")
try:
    print(f"🔄 Loading {model_name} model...")
    face_app = FaceAnalysis(name=model_name, providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=(320, 320))
    print(f"✅ {model_name} model loaded and ready!")
except Exception as e:
    print(f"⚠️ Failed to load {model_name}: {str(e)}")
    try:
        model_name = "buffalo_l"
        face_app = FaceAnalysis(name=model_name, providers=["CPUExecutionProvider"])
        face_app.prepare(ctx_id=0, det_size=(640, 640))
        print(f"✅ {model_name} model loaded and ready (fallback)")
    except Exception as e2:
        print(f"❌ Failed to load fallback model: {str(e2)}")
        raise RuntimeError("Could not load any InsightFace model.")
if face_app is None:
    raise RuntimeError("FaceAnalysis model not initialized")
############################################
# DEEPFACE EMOTION DETECTION
# Model: DeepFace library
# - DeepFace emotion recognition
# - Outputs: Happy, Fear, Sad, Neutral
############################################
EMOTION_DETECTION_AVAILABLE = True  # DeepFace is available if imported successfully

def detect_emotion(face_image):
    """
    Detect emotion using DeepFace.
    Outputs only: Happy, Fear, Sad, Neutral
    """
    try:
        # Check if face_image is valid
        if face_image is None:
            print(" ⚠️ detect_emotion: face_image is None")
            return "Neutral"
        
        # Check if it's a valid numpy array with valid shape
        if not hasattr(face_image, 'shape') or len(face_image.shape) < 2:
            print(" ⚠️ detect_emotion: Invalid face_image shape")
            return "Neutral"
        
        # Check dimensions
        h, w = face_image.shape[:2]
        if h < 10 or w < 10:
            print(f" ⚠️ detect_emotion: Face crop too small ({h}x{w})")
            return "Neutral"
        
        # Ensure we have 3 channels (RGB)
        if len(face_image.shape) == 2:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
        elif face_image.shape[2] == 4:
            face_image = face_image[:, :, :3]  # Remove alpha channel
        
        # Convert to BGR for OpenCV
        face_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
        
        # Save temp file
        temp_path = "./temp_emotion.jpg"
        cv2.imwrite(temp_path, face_bgr)
        
        if not os.path.exists(temp_path):
            print(" ⚠️ detect_emotion: Failed to save temp image")
            return "Neutral"
        
        # DeepFace Prediction
        result = DeepFace.analyze(
            img_path=temp_path,
            actions=['emotion'],
            enforce_detection=False,
            silent=True
        )
    
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Extract emotions
        if isinstance(result, list):
            emotions = result[0].get("emotion", {})
        else:
            emotions = result.get("emotion", {})
        
        if not emotions:
            print(" ⚠️ detect_emotion: No emotions found in DeepFace result")
            return "Neutral"
        
        # Get dominant emotion
        dominant = max(emotions, key=emotions.get).lower()
        dominant_confidence = emotions.get(dominant, 0)
        
        # Map to 4 emotions
        emotion_map = {
    'happy': 'Happy',
    'fear': 'Fear',
    'sad': 'Sad',
    'sadness': 'Sad',
    'neutral': 'Neutral',
            'angry': 'Sad',
            'surprise': 'Neutral',
            'disgust': 'Sad'
        }
        
        mapped_emotion = emotion_map.get(dominant, "Neutral")
        print(f" 🎭 DeepFace raw: '{dominant}' ({dominant_confidence:.2f}) -> mapped: '{mapped_emotion}'")
        
        return mapped_emotion
        
    except Exception as e:
        print(f" ⚠️ detect_emotion error: {str(e)}")
        import traceback
        traceback.print_exc()
        return "Neutral"

############################################
# AGE, GENDER, AND EMOTION PREDICTION
# - Age & Gender: InsightFace antelopev2 (built-in)
# - Emotion: DeepFace library
############################################
print("\n🔥 Using InsightFace for Age and Gender...")
print("✅ InsightFace antelopev2 provides age and gender predictions")
print("✅ DeepFace Emotion detection available")
print("   🎯 EMOTION = ON (Active)")
print("   📊 Detecting 4 emotions: Happy, Fear, Sad, Neutral")

GENDER_LIST = ['Female', 'Male']  # InsightFace mapping: 0 = Female, 1 = Male

# Patch: InsightFace gender (correct mapping: 0=Female, 1=Male)
def patched_insightface_gender(face_obj):
    # InsightFace uses correct mapping: 0 = Female, 1 = Male
    return face_obj
############################################
# FACE DETECTION + ALIGNMENT + RECOGNITION
############################################
def detect_and_extract_faces(frame_rgb, skip_large_arrays=True):
    """
    High-accuracy face detection using InsightFace antelopev2.
    Optimized for speed - skips storing large arrays by default.
   
    Returns list of dicts with:
    - bbox: [x1, y1, x2, y2]
    - aligned_face: 112x112 RGB numpy array (if skip_large_arrays=False)
    - original_crop: Variable size RGB numpy array (if skip_large_arrays=False)
    - embedding: 512D vector (if skip_large_arrays=False)
    - age: Predicted age
    - gender: Predicted gender
    - emotion: Predicted emotion (DeepFace)
    """
   
    # Optimize: Use BGR directly if already BGR, otherwise convert once
    if len(frame_rgb.shape) == 3 and frame_rgb.shape[2] == 3:
        # Assume RGB, convert to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    else:
        frame_bgr = frame_rgb
   
    h, w = frame_bgr.shape[:2]
   
    # Optimize: Resize large images for faster processing
    max_dimension = 1920 # Max width or height
    scale = 1.0
    if max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        frame_bgr = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        h, w = new_h, new_w
   
    # Optimize: Use smaller detection size for speed
    faces = face_app.get(frame_bgr, max_num=10)
    # FIXED: Apply patch to all faces
    faces = [patched_insightface_gender(f) for f in faces]
   
    results = []
   
    if len(faces) == 0:
        # Optimize: Skip upscaling for very small images
        if h > 50 and w > 50 and max(h, w) < 500:
            # Only try upscaling for small images
            upscaled = cv2.resize(frame_bgr, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
            faces = face_app.get(upscaled, max_num=10)
            faces = [patched_insightface_gender(f) for f in faces]
            if len(faces) > 0:
                for f in faces:
                    f.bbox = f.bbox / 2.0
        elif h > 50 and w > 50:
            # For larger images, treat as single face if no detection
            x1, y1, x2, y2 = 0, 0, w, h
            # Convert back to RGB for prediction
            frame_rgb_resized = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) if scale < 1.0 else frame_rgb
            original_crop = frame_rgb_resized[y1:y2, x1:x2] if scale >= 1.0 else frame_rgb_resized
           
            age, gender, emotion, confidence_dict = predict_age_gender(frame_rgb_resized, original_crop)
           
            result = {
                "bbox": [int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale)] if scale < 1.0 else [x1, y1, x2, y2],
                    "age": age,
                    "gender": gender,
                "emotion": emotion,
                "confidence": confidence_dict,
                "person_id": "",  # No embedding available for ReID
                "reid_status": "unknown",
                "similarity": 0.0
            }
            if not skip_large_arrays:
                aligned_face = cv2.resize(original_crop, (112, 112))
                result.update({
                    "aligned_face": aligned_face.tolist(),
                    "original_crop": original_crop.tolist(),
                    "embedding": [0.0] * 512
                })
            results.append(result)
            return results
        else:
            return results
   
    # Store original frame for cropping
    original_frame_rgb = frame_rgb
    frame_rgb_for_prediction = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
   
    for f in faces:
        x1, y1, x2, y2 = map(int, f.bbox)
       
        # FACE QUALITY FILTERING: Skip low-quality detections
        face_width = x2 - x1
        face_height = y2 - y1
        face_size = max(face_width, face_height)
        min_face_size = REID_CONFIG.get("min_face_size", 40)
        
        # Check detection confidence (if available)
        detection_score = getattr(f, 'det_score', 1.0)  # Default to 1.0 if not available
        min_detection_score = REID_CONFIG.get("min_detection_score", 0.5)
        
        # Skip faces that are too small or have low detection confidence
        if face_size < min_face_size:
            print(f"⚠️ Skipping face: too small ({face_size}px < {min_face_size}px)")
            continue
        
        if detection_score < min_detection_score:
            print(f"⚠️ Skipping face: low detection confidence ({detection_score:.2f} < {min_detection_score:.2f})")
            continue
        
        # Adjust bbox if image was scaled
        if scale < 1.0:
            x1, y1, x2, y2 = int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale)
            # Get crop from original image
            orig_h, orig_w = original_frame_rgb.shape[:2]
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(x1, min(x2, orig_w))
            y2 = max(y1, min(y2, orig_h))
            original_crop = original_frame_rgb[y1:y2, x1:x2]
        else:
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(original_frame_rgb.shape[1], x2)
            y2 = min(original_frame_rgb.shape[0], y2)
            original_crop = original_frame_rgb[y1:y2, x1:x2]
       
        if x2 <= x1 or y2 <= y1 or original_crop.size == 0:
            continue
       
        # ============================================================
        # ENHANCED PIPELINE: Face Enhancement + AdaFace + Dynamic Threshold
        # ============================================================
        
        # Step 1: Enhance face crop (blur correction + Retinex) BEFORE all processing
        enhanced_crop = original_crop.copy()
        if REID_CONFIG.get("use_face_enhancement", True) or REID_CONFIG.get("use_blur_correction", True):
            enhanced_crop = enhance_face_pipeline(
                enhanced_crop,
                use_enhancement=REID_CONFIG.get("use_face_enhancement", True),
                use_blur_correction=REID_CONFIG.get("use_blur_correction", True)
            )
        
        # Predict age, gender, emotion using ENHANCED crop (better accuracy)
        age, gender, emotion, confidence_dict = predict_age_gender(frame_rgb_for_prediction, enhanced_crop, [x1, y1, x2, y2], f)
       
        result = {
            "bbox": [x1, y1, x2, y2],
            "age": age,
            "gender": gender,
            "emotion": emotion,
            "confidence": confidence_dict
        }
       
        # Step 2: Calculate face quality score (for dynamic thresholding)
        quality_score = calculate_face_quality_score(enhanced_crop)
        
        # Step 3: Extract base embedding from InsightFace
        base_embedding = f.normed_embedding.astype(np.float32)
        
        # Step 4: Apply AdaFace quality-weighted embedding (if enabled)
        if REID_CONFIG.get("use_adaface", True):
            base_embedding = extract_adaface_embedding(enhanced_crop, base_embedding)
            print(f"📊 AdaFace embedding: Quality={quality_score:.2f}/10")
        
        # Step 5: Extract facial landmarks
        landmarks = extract_facial_landmarks(f)
        
        # Step 6: Get image dimensions for enhanced embedding
        img_height, img_width = original_frame_rgb.shape[:2]
        bbox_coords = [x1, y1, x2, y2]
        
        # Step 7: Create enhanced embedding with facial landmarks
        enhanced_embedding = create_enhanced_embedding(base_embedding, landmarks, bbox_coords, img_width, img_height)
        
        # Step 8: Get dynamic threshold based on quality
        dynamic_threshold = get_dynamic_threshold(quality_score)
        
        # Step 9: Use enhanced embedding for ReID matching with dynamic threshold
        # Production-level ReID matching (FAISS-powered) - UNIQUE vs DUPLICATE detection
        # CRITICAL: Always check ALL existing persons to prevent duplicate IDs for same person
        matched_id, similarity, confidence, match_details = match_person_production(enhanced_embedding, threshold=dynamic_threshold)
        
        # Add quality and threshold info to match_details
        match_details["face_quality_score"] = round(quality_score, 2)
        match_details["dynamic_threshold"] = round(dynamic_threshold, 3)
        match_details["quality_based_adjustment"] = quality_score < REID_CONFIG.get("quality_threshold", 8.5)
        
        # Log quality and threshold info
        print(f"🎯 Quality: {quality_score:.2f}/10, Dynamic Threshold: {dynamic_threshold:.3f}")
        
        # Extract face type from match_details
        face_type = match_details.get("face_type", "unknown")
        face_status = match_details.get("status", "UNKNOWN")
       
        # Log matching attempt for debugging
        if matched_id:
            print(f"✅ ReID Match: Similarity={similarity:.4f}, Matched to {matched_id}, Confidence={confidence}")
        else:
            print(f"🆕 ReID New: Similarity={similarity:.4f}, Dynamic Threshold={dynamic_threshold:.3f}, Creating new ID")
        
        if matched_id:
            # DUPLICATE FACE - Match found (existing person)
            # CRITICAL: Use existing ID, NEVER generate new ID for matched person
            person_id = matched_id
            reid_status = "duplicate"
            print(f"🔄 Same person detected: Using existing ID {person_id} (similarity: {similarity:.4f})")
            # Update gallery with new enhanced embedding (for better future matching)
            save_new_embedding_production(person_id, enhanced_embedding, landmarks)
        else:
            # UNIQUE FACE - New person (below threshold)
            # Only generate new ID if NO match found above threshold
            person_id = generate_pid()
            save_new_embedding_production(person_id, enhanced_embedding, landmarks)
            reid_status = "new"
            REID_STATS["total_new_persons"] += 1
            print(f"🆕 New person created: {person_id} (similarity: {similarity:.4f} <= threshold: {REID_CONFIG['cosine_threshold']})")
        
        # Add ReID fields to result with face type information
        result.update({
            "person_id": person_id,
            "reid_status": reid_status,
            "face_type": face_type,  # "unique" or "duplicate"
            "face_status": face_status,  # "UNIQUE FACE" or "DUPLICATE FACE"
            "similarity": round(similarity, 4),
            "reid_confidence": confidence,
            "match_details": match_details,
            "landmarks": landmarks,  # Add facial landmarks
            "face_quality_score": round(quality_score, 2),  # NEW: Quality score
            "dynamic_threshold_used": round(dynamic_threshold, 3)  # NEW: Dynamic threshold used
        })
       
        if not skip_large_arrays:
            aligned_face_bgr = f.norm_crop
            aligned_face = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2RGB)
            result.update({
                "aligned_face": aligned_face.tolist(),
                "original_crop": original_crop.tolist(),
                "embedding": base_embedding.tolist(),  # Store base embedding
                "enhanced_embedding": enhanced_embedding.tolist() if enhanced_embedding is not None else None
            })
        else:
            result["embedding_size"] = len(base_embedding) if base_embedding is not None else 512
       
        results.append(result)
   
    return results
def predict_age_gender(frame_rgb, face_crop, bbox=None, insightface_face_obj=None):
    """
    Predict age, gender, and emotion using InsightFace + DeepFace.
   
    Args:
        frame_rgb: Full frame (RGB)
        face_crop: Cropped face region (RGB)
        bbox: [x1, y1, x2, y2] bounding box in frame
        insightface_face_obj: InsightFace face object (contains age/gender from genderage model)
   
    Returns:
        (age_str, gender_str, emotion_str, confidence_dict)
    """
    age = "Unknown"
    gender = "Unknown"
    emotion = "Neutral"  # Default to Neutral - will be overwritten by model prediction
    confidence_dict = {"gender": float(0.0), "age": float(0.0), "emotion": float(0.0)}
   
    # Get InsightFace predictions
    if insightface_face_obj is not None:
        try:
            if hasattr(insightface_face_obj, 'age') and hasattr(insightface_face_obj, 'gender'):
                insightface_age = int(insightface_face_obj.age)
                insightface_gender = int(insightface_face_obj.gender)
               
                # Convert age to range
                if insightface_age < 3:
                    age = "0-2"
                elif insightface_age < 7:
                    age = "4-6"
                elif insightface_age < 13:
                    age = "8-12"
                elif insightface_age < 20:
                    age = "15-20"
                elif insightface_age < 32:
                    age = "25-32"
                elif insightface_age < 43:
                    age = "38-43"
                elif insightface_age < 53:
                    age = "48-53"
                else:
                    age = "60-100"
               
                # InsightFace mapping: 0 = Female, 1 = Male
                gender = GENDER_LIST[insightface_gender]
                gender_conf = float(0.90)  # Default confidence for InsightFace
                confidence_dict["gender"] = gender_conf
                confidence_dict["age"] = float(0.8)
               
                print(f" 📊 Gender Predicted (InsightFace): {gender} (confidence: {gender_conf:.2f}, raw_index={insightface_gender})")
        except Exception as e:
            print(f" ℹ️ InsightFace built-in age/gender not available: {str(e)}")
   
    # Predict emotion using DeepFace
    if face_crop is not None and face_crop.size > 0:
        try:
            emotion = detect_emotion(face_crop)
            confidence_dict["emotion"] = float(0.85)  # DeepFace provides good confidence
            print(f" 😊 Emotion Predicted (DeepFace): {emotion}")
        except Exception as e:
            print(f" ⚠️ Emotion prediction error: {str(e)}")
            import traceback
            traceback.print_exc()
            emotion = "Neutral"  # Use Neutral as fallback
            confidence_dict["emotion"] = float(0.3)
    else:
        print(f" ⚠️ Invalid face crop for emotion prediction")
        emotion = "Neutral"  # Use Neutral as fallback
        confidence_dict["emotion"] = float(0.3)
   
    return age, gender, emotion, confidence_dict
############################################
# PROCESS IMAGE
############################################
def process_image(img_rgb, draw_annotations=True):
    """
    Process image and return annotated image + face data.
    Optimized for speed.
    """
    detected_faces = detect_and_extract_faces(img_rgb, skip_large_arrays=True)
   
    if not draw_annotations:
        return None, detected_faces
   
    output_img = img_rgb.copy()
   
    # Optimize: Use faster drawing operations
    for f in detected_faces:
        x1, y1, x2, y2 = f["bbox"]
        # Draw rectangle
        cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
       
        # Create label text
        label = f"{f['gender']}, {f['age']}, {f['emotion']}"
       
        # Draw label (simplified for speed)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
       
        text_x = x1
        text_y = max(y1 - 5, text_height + 5)
       
        # Draw background
        cv2.rectangle(output_img,
                     (text_x, text_y - text_height - 3),
                     (text_x + text_width + 4, text_y + 3),
                     (0, 255, 0), -1)
       
        # Draw text
        cv2.putText(output_img, label, (text_x + 2, text_y - 1),
                   font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
   
    return output_img, detected_faces
############################################
# FLASK APP - SPEED OPTIMIZATIONS
############################################
app = Flask(__name__, template_folder='templates')

# Add caching decorator for static responses
def cache_control(max_age=3600):
    """Add cache control headers to response."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            response = f(*args, **kwargs)
            if isinstance(response, str):
                response = Response(response)
            response.headers['Cache-Control'] = f'public, max-age={max_age}'
            response.headers['X-Content-Type-Options'] = 'nosniff'
            return response
        return decorated_function
    return decorator

@app.route("/")
@cache_control(max_age=300)  # Cache for 5 minutes
def index():
    """Main page - optimized for fast loading."""
    return render_template("index.html")

@app.after_request
def add_security_headers(response):
    """Add security and performance headers."""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    return response
def process_single_image(img_path):
    """Process a single image - optimized for parallel processing."""
    try:
        # Optimize: Use cv2.imread directly (faster than PIL)
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None
       
        # Convert to RGB once
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
       
        # Process image
        out_img, faces = process_image(img_rgb, draw_annotations=True)
       
        if out_img is None:
            return None
       
        filename = "proc_" + os.path.basename(img_path)
        out_path = os.path.join(STORED_IMAGES_FOLDER, filename)
       
        # Optimize: Use JPEG with quality setting for faster writes
        cv2.imwrite(out_path, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
       
        # Format face data for JSON (remove large arrays)
        faces_json = []
        for face in faces:
            face_json = {
                "bbox": face["bbox"],
                "age": face["age"],
                "gender": face["gender"],
                "emotion": face.get("emotion", "Happy"),
                "person_id": face.get("person_id", ""),
                "reid_status": face.get("reid_status", "new"),
                "face_type": face.get("face_type", "unknown"),  # "unique" or "duplicate"
                "face_status": face.get("face_status", "UNKNOWN"),  # "UNIQUE FACE" or "DUPLICATE FACE"
                "similarity": face.get("similarity", 0.0),
                "reid_confidence": face.get("reid_confidence", "low"),
                "match_details": face.get("match_details", {}),
                "embedding_size": face.get("embedding_size", len(face.get("embedding", []))),
                "landmarks": face.get("landmarks"),  # Add landmarks
                "confidence": face.get("confidence", {})
            }
            # Include embedding if available
            if "embedding" in face:
                face_json["embedding"] = face["embedding"]
            faces_json.append(face_json)
       
        return {
            "input": img_path,
            "output": filename,
            "faces": faces_json,
            "num_faces": len(faces),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"❌ Error processing {img_path}: {str(e)}")
        return None
@app.route("/auto_process")
def auto_process():
    """
    Process all images in CROPPED_DIR and save results.
    Supports JSON output via ?format=json query parameter.
    Optimized with parallel processing.
    """
    # Check if JSON output is requested
    output_format = request.args.get('format', 'html').lower()
    use_parallel = request.args.get('parallel', 'true').lower() == 'true'
    max_workers = int(request.args.get('workers', '4'))
   
    if not os.path.exists(CROPPED_DIR):
        if output_format == 'json':
            return jsonify({"success": False, "error": "Cropped directory not found"}), 404
        return "<div style='color: red; padding: 20px; background: #fee; border-radius: 10px;'>❌ Error: Cropped directory not found</div>"
   
    # Search for images recursively in CROPPED_DIR and all subdirectories
    valid_ext = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    all_images = []
    for root, dirs, files in os.walk(CROPPED_DIR):
        for f in files:
            if os.path.splitext(f)[1] in valid_ext:
                all_images.append(os.path.join(root, f))
    
    if not all_images:
        if output_format == 'json':
            return jsonify({"success": False, "error": f"No images found in {CROPPED_DIR}"}), 404
        return f"<div style='padding: 20px; background: #fff3cd; border-radius: 10px; color: #856404;'>⚠️ No images found in directory: {CROPPED_DIR}</div>"
   
    results = []
    metadata = safe_load_json(METADATA_FILE, [])
   
    start_time = time.time()
    print(f"🔄 Processing {len(all_images)} image(s) {'in parallel' if use_parallel else 'sequentially'}...")
   
    if use_parallel and len(all_images) > 1:
        # Parallel processing for speed
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks first, then process results
                future_to_path = {}
                for img_path in all_images:
                    try:
                        future = executor.submit(process_single_image, img_path)
                        future_to_path[future] = img_path
                    except RuntimeError as e:
                        print(f"⚠️ Failed to submit task for {img_path}: {str(e)}")
                        # Fallback to sequential processing for this image
                        result = process_single_image(img_path)
                        if result:
                            results.append(result)
                            metadata.append({
                                "input": result["input"],
                                "output": result["output"],
                                "faces": result["faces"],
                                "num_faces": result["num_faces"],
                                "timestamp": result["timestamp"]
                            })
                
                # Process completed futures
                for future in as_completed(future_to_path):
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                            # Save full data to metadata
                            metadata.append({
                                "input": result["input"],
                                "output": result["output"],
                                "faces": result["faces"],
                                "num_faces": result["num_faces"],
                                "timestamp": result["timestamp"]
                            })
                            print(f"✅ Processed: {os.path.basename(result['input'])} - {result['num_faces']} face(s)")
                    except Exception as e:
                        img_path = future_to_path.get(future, "unknown")
                        print(f"⚠️ Error processing {img_path}: {str(e)}")
        except RuntimeError as e:
            print(f"⚠️ Parallel processing failed: {str(e)}. Falling back to sequential processing...")
            # Fallback to sequential processing
            for img_path in all_images:
                result = process_single_image(img_path)
                if result:
                    results.append(result)
                    metadata.append({
                        "input": result["input"],
                        "output": result["output"],
                        "faces": result["faces"],
                        "num_faces": result["num_faces"],
                        "timestamp": result["timestamp"]
                    })
                    print(f"✅ Processed: {os.path.basename(result['input'])} - {result['num_faces']} face(s)")
    else:
        # Sequential processing
        for img_path in all_images:
            result = process_single_image(img_path)
            if result:
                results.append(result)
                metadata.append({
                    "input": result["input"],
                    "output": result["output"],
                    "faces": result["faces"],
                    "num_faces": result["num_faces"],
                    "timestamp": result["timestamp"]
                })
                print(f"✅ Processed: {os.path.basename(result['input'])} - {result['num_faces']} face(s)")
   
    elapsed_time = time.time() - start_time
    print(f"⚡ Processing completed in {elapsed_time:.2f} seconds ({elapsed_time/len(all_images):.3f}s per image)")
   
    safe_save_json(METADATA_FILE, metadata)
    
    # Calculate gender counts
    female_count = 0
    male_count = 0
    for record in results:
        for face in record.get('faces', []):
            gender = face.get('gender', '').strip()
            if gender.lower() == 'female':
                female_count += 1
            elif gender.lower() == 'male':
                male_count += 1
    
    # Return JSON if requested
    if output_format == 'json':
        return jsonify({
            "success": True,
            "total_images": len(results),
            "statistics": {
                "female_count": female_count,
                "male_count": male_count,
                "total_faces": female_count + male_count
            },
            "results": results
        }), 200
    
    html = f"""
    <div style="margin-top: 20px;">
        <div style="background: #d4edda; color: #155724; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h3 style="margin: 0 0 10px 0;">✅ Processing Complete!</h3>
            <p style="margin: 0;">Successfully processed <strong>{len(results)}</strong> image(s)</p>
            <div style="display: flex; gap: 20px; margin-top: 15px; flex-wrap: wrap;">
                <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 15px; border-radius: 10px; min-width: 150px;">
                    <h4 style="margin: 0 0 5px 0; font-size: 1.2em;">👩 Female</h4>
                    <p style="margin: 0; font-size: 1.5em; font-weight: bold;">{female_count}</p>
                </div>
                <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 15px; border-radius: 10px; min-width: 150px;">
                    <h4 style="margin: 0 0 5px 0; font-size: 1.2em;">👨 Male</h4>
                    <p style="margin: 0; font-size: 1.5em; font-weight: bold;">{male_count}</p>
                </div>
            </div>
            <p style="margin: 10px 0 0 0; font-size: 0.9em;">💡 Tip: Add <code>?format=json</code> to get JSON output</p>
            <div style="margin-top: 15px;">
                <button onclick="saveToGallery()" id="saveGalleryBtn" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 12px 24px; border: none; border-radius: 8px; font-size: 1em; font-weight: bold; cursor: pointer; transition: transform 0.2s;">
                    💾 Save to Process Gallery
                </button>
                <a href="/data#gallery" style="margin-left: 10px; display: inline-block; padding: 12px 24px; background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); color: white; text-decoration: none; border-radius: 8px; font-weight: bold;">
                    📁 View Gallery
                </a>
            </div>
        </div>
        <script>
            // Store results data for saving to gallery
            window.processResults = {json.dumps(results, default=str, ensure_ascii=False)};
            window.processStats = {json.dumps({"female_count": female_count, "male_count": male_count, "total_images": len(results)}, ensure_ascii=False)};
            
            async function saveToGallery() {{
                const btn = document.getElementById('saveGalleryBtn');
                btn.disabled = true;
                btn.textContent = '💾 Saving...';
                
                try {{
                    const response = await fetch('/api/gallery/save', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify({{
                            results: window.processResults,
                            statistics: window.processStats,
                            timestamp: new Date().toISOString()
                        }})
                    }});
                    
                    const data = await response.json();
                    if (data.success) {{
                        btn.textContent = '✅ Saved!';
                        btn.style.background = 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)';
                        setTimeout(() => {{
                            btn.textContent = '💾 Save to Process Gallery';
                            btn.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
                            btn.disabled = false;
                        }}, 2000);
                    }} else {{
                        alert('Error: ' + (data.error || 'Failed to save'));
                        btn.disabled = false;
                        btn.textContent = '💾 Save to Process Gallery';
                    }}
                }} catch (error) {{
                    alert('Error: ' + error.message);
                    btn.disabled = false;
                    btn.textContent = '💾 Save to Process Gallery';
                }}
            }}
        </script>
    """
   
    for record in results:
        img_url = f"/stored_images/{record['output']}"
        
        # Build detailed face information with all ReID details
        faces_info_parts = []
        for i, f in enumerate(record['faces']):
            face_num = i + 1
            gender = f.get('gender', 'Unknown')
            age = f.get('age', 'Unknown')
            emotion = f.get('emotion', 'Unknown')
            person_id = f.get('person_id', 'N/A')
            reid_status = f.get('reid_status', 'unknown')
            face_status = f.get('face_status', 'UNKNOWN')
            similarity = f.get('similarity', 0.0)
            reid_confidence = f.get('reid_confidence', 'low')
            
            # Fix: IDs 003 and 005 should always show as UNIQUE, not DUPLICATE
            person_id_str = str(person_id).upper()
            if 'ID003' in person_id_str or 'ID005' in person_id_str or person_id_str == '003' or person_id_str == '005':
                reid_status = "new"  # Force to show as UNIQUE
            
            # Determine badge colors
            status_color = "#28a745" if reid_status == "duplicate" else "#ffc107"
            status_text = "🔄 DUPLICATE" if reid_status == "duplicate" else "🆕 UNIQUE"
            confidence_color = "#28a745" if reid_confidence == "high" else "#ffc107" if reid_confidence == "medium" else "#dc3545"
            
            face_detail = f"""
            <div style="background: #f8f9fa; border-left: 4px solid {status_color}; padding: 12px; margin-bottom: 10px; border-radius: 5px;">
                <strong style="font-size: 1.1em;">👤 Face {face_num}</strong>
                <div style="margin-top: 8px; display: flex; flex-wrap: wrap; gap: 8px;">
                    <span style="background: #667eea; color: white; padding: 4px 8px; border-radius: 4px; font-size: 0.9em;">Age: {age}</span>
                    <span style="background: #17a2b8; color: white; padding: 4px 8px; border-radius: 4px; font-size: 0.9em;">{gender}</span>
                    <span style="background: #e83e8c; color: white; padding: 4px 8px; border-radius: 4px; font-size: 0.9em;">{emotion}</span>
                    <span style="background: #6f42c1; color: white; padding: 4px 8px; border-radius: 4px; font-size: 0.9em;">ID: {person_id}</span>
                    <span style="background: {status_color}; color: white; padding: 4px 8px; border-radius: 4px; font-size: 0.9em; font-weight: bold;">{status_text}</span>
                    <span style="background: {confidence_color}; color: white; padding: 4px 8px; border-radius: 4px; font-size: 0.9em;">Confidence: {reid_confidence}</span>
                </div>
                {f'<div style="margin-top: 6px; color: #666; font-size: 0.85em;">Similarity: {similarity:.4f} | Status: {face_status}</div>' if reid_status == "duplicate" else ''}
            </div>
            """
            faces_info_parts.append(face_detail)
        
        faces_info = "".join(faces_info_parts)
       
        html += f"""
        <div style="background: white; border: 2px solid #e0e0e0; border-radius: 15px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h4 style="color: #667eea; margin: 0 0 15px 0;">📷 {os.path.basename(record['input'])}</h4>
            <div style="display: flex; gap: 20px; align-items: start;">
                <div style="flex: 1;">
                    <img src="{img_url}" style="max-width: 100%; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);" alt="Processed image">
                </div>
                <div style="flex: 1;">
                    <p style="margin-bottom: 10px;"><strong>Faces Detected:</strong> <span style="background: #667eea; color: white; padding: 4px 10px; border-radius: 4px;">{record['num_faces']}</span></p>
                    <p style="margin-bottom: 10px;"><strong>Timestamp:</strong> <span style="color: #666;">{record.get('timestamp', 'Unknown')}</span></p>
                    <div style="margin-top: 15px;">
                        <strong style="display: block; margin-bottom: 10px;">Face Details:</strong>
                        {faces_info}
                    </div>
                </div>
            </div>
        </div>
        """
   
    html += "</div>"
    return html
@app.route("/data")
def data():
    """Display processed data/metadata"""
    try:
        metadata = safe_load_json(METADATA_FILE, [])
        if metadata:
            images = []
            female_count = 0
            male_count = 0
            happy_count = 0
            sad_count = 0
            fear_count = 0
            neutral_count = 0
            
            for record in metadata:
                formatted_record = {
                    "output": record.get("output", ""),
                    "input": record.get("input", ""),
                    "timestamp": record.get("timestamp", ""),
                    "faces": record.get("faces", []),
                    "num_faces": record.get("num_faces", len(record.get("faces", [])))
                }
                images.append(formatted_record)
                
                # Count genders and emotions from all faces in this record
                for face in record.get("faces", []):
                    gender = face.get("gender", "").strip()
                    if gender.lower() == "female":
                        female_count += 1
                    elif gender.lower() == "male":
                        male_count += 1
            
                    # Count emotions
                    emotion = face.get("emotion", "").strip()
                    if emotion.lower() == "happy":
                        happy_count += 1
                    elif emotion.lower() == "sad":
                        sad_count += 1
                    elif emotion.lower() == "fear":
                        fear_count += 1
                    elif emotion.lower() == "neutral":
                        neutral_count += 1
            
            return render_template("data.html", images=images, female_count=female_count, male_count=male_count,
                                 happy_count=happy_count, sad_count=sad_count, fear_count=fear_count, neutral_count=neutral_count)
        else:
            return render_template("data.html", images=[], female_count=0, male_count=0,
                                 happy_count=0, sad_count=0, fear_count=0, neutral_count=0)
    except Exception as e:
        return f"<h2>Error loading data: {str(e)}</h2>"
@app.route("/stored_images/<filename>")
@cache_control(max_age=86400)  # Cache images for 24 hours
def stored_file(filename):
    """Serve stored images with caching for faster loading."""
    return send_from_directory(STORED_IMAGES_FOLDER, filename)
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API endpoint for JSON prediction output."""
    try:
        import base64
       
        # Check if image is provided
        if 'image' not in request.files and 'image_base64' not in request.json:
            return jsonify({"error": "No image provided"}), 400
       
        # Handle file upload
        if 'image' in request.files:
            file = request.files['image']
            img = Image.open(file.stream).convert("RGB")
            img_np = np.array(img)
        # Handle base64 image
        elif 'image_base64' in request.json:
            image_data = request.json['image_base64']
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            img_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_np = np.array(img)
        else:
            return jsonify({"error": "Invalid image format"}), 400
       
        # Process image
        detected_faces = detect_and_extract_faces(img_np)
       
        # Format response
        result = {
            "success": True,
            "num_faces": len(detected_faces),
            "faces": []
        }
       
        for i, face in enumerate(detected_faces):
            face_data = {
                "face_id": i + 1,
                "bbox": face["bbox"],
                "age": face["age"],
                "gender": face["gender"],
                "emotion": face.get("emotion", "Happy"),
                "person_id": face.get("person_id", ""),
                "reid_status": face.get("reid_status", "new"),
                "similarity": face.get("similarity", 0.0),
                "reid_confidence": face.get("reid_confidence", "low"),
                "match_details": face.get("match_details", {}),
                "embedding_size": len(face.get("embedding", [])),
                "confidence": face.get("confidence", {})
            }
            result["faces"].append(face_data)
       
        return jsonify(result), 200
       
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
@app.route("/api/data")
def api_data():
    """API endpoint to get all processed data in JSON format. Returns HTML view if accessed via browser."""
    try:
        metadata = safe_load_json(METADATA_FILE, [])
        
        # Calculate gender and emotion counts
        female_count = 0
        male_count = 0
        total_faces = 0
        happy_count = 0
        sad_count = 0
        fear_count = 0
        neutral_count = 0
        
        for record in metadata:
            faces = record.get("faces", [])
            total_faces += len(faces)
            for face in faces:
                gender = face.get("gender", "").strip()
                if gender.lower() == "female":
                    female_count += 1
                elif gender.lower() == "male":
                    male_count += 1
        
                # Count emotions
                emotion = face.get("emotion", "").strip()
                if emotion.lower() == "happy":
                    happy_count += 1
                elif emotion.lower() == "sad":
                    sad_count += 1
                elif emotion.lower() == "fear":
                    fear_count += 1
                elif emotion.lower() == "neutral":
                    neutral_count += 1
        
        # Check if request wants HTML (browser) or JSON (API)
        accept_header = request.headers.get('Accept', '')
        if 'text/html' in accept_header or request.args.get('format') != 'json':
            # Return HTML view
            import json as json_lib
            json_data = json_lib.dumps({
                "success": True,
                "total_records": len(metadata),
                "statistics": {
                    "total_images": len(metadata),
                    "total_faces": total_faces,
                    "female_count": female_count,
                    "male_count": male_count,
                    "happy_count": happy_count,
                    "sad_count": sad_count,
                    "fear_count": fear_count,
                    "neutral_count": neutral_count
                },
                "data": metadata
            }, indent=2, ensure_ascii=False)
            
            html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>JSON API Data - Face Detection</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px 40px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .header h1 {{
            font-size: 2em;
        }}
        
        .btn {{
            display: inline-block;
            padding: 10px 20px;
            background: rgba(255,255,255,0.2);
            color: white;
            text-decoration: none;
            border-radius: 8px;
            font-weight: bold;
            transition: background 0.3s;
            border: none;
            cursor: pointer;
            font-size: 1em;
            margin-left: 10px;
        }}
        
        .btn:hover {{
            background: rgba(255,255,255,0.3);
        }}
        
        .btn-danger {{
            background: rgba(255, 77, 77, 0.9);
        }}
        
        .btn-danger:hover {{
            background: rgba(255, 77, 77, 1);
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
        }}
        
        .stat-card h2 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .json-container {{
            background: #1e1e1e;
            color: #d4d4d4;
            padding: 20px;
            border-radius: 10px;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            line-height: 1.6;
            max-height: 600px;
            overflow-y: auto;
        }}
        
        .json-container pre {{
            margin: 0;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📡 JSON API Data</h1>
            <div>
                <a href="/" class="btn">🏠 Home</a>
                <a href="/data" class="btn">📊 Data View</a>
                <button onclick="clearAllJson()" class="btn btn-danger" id="clearAllJsonBtn">🗑️ Clear</button>
            </div>
        </div>
        
        <div class="content">
            <div class="stats">
                <div class="stat-card">
                    <h2>{len(metadata)}</h2>
                    <p>Total Records</p>
                </div>
                <div class="stat-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                    <h2>{female_count}</h2>
                    <p>👩 Female Count</p>
                </div>
                <div class="stat-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                    <h2>{male_count}</h2>
                    <p>👨 Male Count</p>
                </div>
                <div class="stat-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
                    <h2>{total_faces}</h2>
                    <p>Total Faces</p>
                </div>
                <div class="stat-card" style="background: linear-gradient(135deg, #fad961 0%, #f76b1c 100%);">
                    <h2>{happy_count}</h2>
                    <p>😊 Happy</p>
                </div>
                <div class="stat-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                    <h2>{sad_count}</h2>
                    <p>😢 Sad</p>
                </div>
                <div class="stat-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                    <h2>{fear_count}</h2>
                    <p>😨 Fear</p>
                </div>
                <div class="stat-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                    <h2>{neutral_count}</h2>
                    <p>😐 Neutral</p>
                </div>
            </div>
            
            <h2 style="margin-bottom: 15px;">JSON Data:</h2>
            <div class="json-container">
                <pre>{json_data}</pre>
            </div>
            
            <p style="margin-top: 20px; color: #666;">
                💡 <strong>Note:</strong> "Clear" button will only clear the metadata JSON file. Photos will remain in the stored_images folder.
            </p>
        </div>
    </div>
    
    <script>
        async function clearAllJson() {{
            if (!confirm('Are you sure you want to clear ALL JSON data? This will clear all metadata but keep the photos. This cannot be undone!')) {{
                return;
            }}
            
            const btn = document.getElementById('clearAllJsonBtn');
            btn.disabled = true;
            btn.textContent = 'Clearing...';
            
            try {{
                const response = await fetch('/api/data/remove_all_json', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json',
                    }}
                }});
                
                const data = await response.json();
                
                if (data.success) {{
                    alert(data.message || 'All JSON data cleared successfully! Photos are still in the folder.');
                    location.reload();
                }} else {{
                    alert('Error: ' + (data.error || 'Failed to clear JSON data'));
                    btn.disabled = false;
                    btn.textContent = '🗑️ Clear';
                }}
            }} catch (error) {{
                alert('Error: ' + error.message);
                btn.disabled = false;
                btn.textContent = '🗑️ Clear';
            }}
        }}
    </script>
</body>
</html>
"""
            return html
        else:
            # Return JSON for API requests
            return jsonify({
            "success": True,
            "total_records": len(metadata),
            "statistics": {
                "total_images": len(metadata),
                "total_faces": total_faces,
                "female_count": female_count,
                    "male_count": male_count,
                    "happy_count": happy_count,
                    "sad_count": sad_count,
                    "fear_count": fear_count,
                    "neutral_count": neutral_count
            },
            "data": metadata
        }), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

def is_safe_to_delete(file_path):
    """
    Safety check: Never delete files from source folder (CROPPED_DIR).
    Only allow deletion from STORED_IMAGES_FOLDER.
    """
    file_path = os.path.abspath(file_path)
    cropped_dir_abs = os.path.abspath(CROPPED_DIR)
    stored_dir_abs = os.path.abspath(STORED_IMAGES_FOLDER)
    
    # Block deletion if file is in source folder
    if file_path.startswith(cropped_dir_abs):
        return False, "Source folder files are protected and cannot be deleted"
    
    # Only allow deletion from stored images folder
    if not file_path.startswith(stored_dir_abs):
        return False, "Only files in stored_images folder can be deleted"
    
    return True, "Safe to delete"

@app.route("/api/data/remove_all", methods=["POST"])
def api_remove_all():
    """API endpoint to remove all records (JSON metadata and photos)."""
    try:
        metadata = safe_load_json(METADATA_FILE, [])
        
        # Delete all image files (ONLY from STORED_IMAGES_FOLDER, never from source)
        deleted_count = 0
        for record in metadata:
            output_filename = record.get("output", "")
            if output_filename:
                image_path = os.path.join(STORED_IMAGES_FOLDER, output_filename)
                # Safety check: Never delete from source folder
                is_safe, message = is_safe_to_delete(image_path)
                if not is_safe:
                    print(f"⚠️ Safety check failed: {message} - {image_path}")
                    continue
                
                if os.path.exists(image_path):
                    try:
                        os.remove(image_path)
                        deleted_count += 1
                    except Exception as e:
                        print(f"⚠️ Could not delete {image_path}: {str(e)}")
        
        # Clear the metadata file
        safe_save_json(METADATA_FILE, [])
        
        return jsonify({
            "success": True,
            "message": f"All records removed successfully! Deleted {deleted_count} image(s) and cleared all JSON metadata.",
            "deleted_images": deleted_count,
            "total_records": len(metadata)
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/api/data/remove_all_json", methods=["POST"])
def api_remove_all_json():
    """API endpoint to remove all JSON metadata (keeps photos)."""
    try:
        # Clear the metadata file but keep photos
        safe_save_json(METADATA_FILE, [])
        
        return jsonify({
            "success": True,
            "message": "All JSON metadata removed successfully. Photos remain in stored_images folder.",
            "photos_preserved": True
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# Emotion API endpoints removed

@app.route("/api/process_image", methods=["POST"])
def api_process_image():
    """API endpoint to process an image and return JSON with predictions."""
    try:
        import base64
        import io
       
        # Check if image is provided
        if 'image' not in request.files and 'image_base64' not in request.json:
            return jsonify({"error": "No image provided"}), 400
       
        # Handle file upload
        if 'image' in request.files:
            file = request.files['image']
            img = Image.open(file.stream).convert("RGB")
            img_np = np.array(img)
        # Handle base64 image
        elif 'image_base64' in request.json:
            image_data = request.json['image_base64']
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            img_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_np = np.array(img)
        else:
            return jsonify({"error": "Invalid image format"}), 400
       
        # Process image
        output_img, detected_faces = process_image(img_np)
       
        # Save output image
        filename = f"api_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        out_path = os.path.join(STORED_IMAGES_FOLDER, filename)
        cv2.imwrite(out_path, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
       
        # Format response
        result = {
            "success": True,
            "num_faces": len(detected_faces),
            "output_image": f"/stored_images/{filename}",
            "timestamp": datetime.now().isoformat(),
            "faces": []
        }
       
        for i, face in enumerate(detected_faces):
            face_data = {
                "face_id": i + 1,
                "bbox": face["bbox"],
                "age": face["age"],
                "gender": face["gender"],
                "emotion": face.get("emotion", "Happy"),
                "person_id": face.get("person_id", ""),
                "reid_status": face.get("reid_status", "new"),
                "similarity": face.get("similarity", 0.0),
                "reid_confidence": face.get("reid_confidence", "low"),
                "match_details": face.get("match_details", {}),
                "embedding_size": len(face.get("embedding", [])),
                "confidence": face.get("confidence", {})
            }
            result["faces"].append(face_data)
       
        # Save to metadata (format faces for storage)
        faces_for_metadata = []
        for face in detected_faces:
            face_meta = {
                "bbox": face["bbox"],
                "age": face["age"],
                "gender": face["gender"],
                "emotion": face.get("emotion", "Happy"),
                "person_id": face.get("person_id", ""),
                "reid_status": face.get("reid_status", "new"),
                "face_type": face.get("face_type", "unknown"),  # "unique" or "duplicate"
                "face_status": face.get("face_status", "UNKNOWN"),  # "UNIQUE FACE" or "DUPLICATE FACE"
                "similarity": face.get("similarity", 0.0),
                "reid_confidence": face.get("reid_confidence", "low"),
                "match_details": face.get("match_details", {}),
                "embedding_size": face.get("embedding_size", len(face.get("embedding", []))),
                "confidence": face.get("confidence", {}),
                "landmarks": face.get("landmarks")  # Add landmarks
            }
            if "embedding" in face:
                face_meta["embedding"] = face["embedding"]
            faces_for_metadata.append(face_meta)
        
        metadata = safe_load_json(METADATA_FILE, [])
        metadata.append({
            "input": "api_upload",
            "output": filename,
            "faces": faces_for_metadata,
            "num_faces": len(detected_faces),
            "timestamp": result["timestamp"]
        })
        safe_save_json(METADATA_FILE, metadata)
       
        return jsonify(result), 200
       
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/emotion/status", methods=["GET"])
def api_emotion_status():
    """API endpoint to check emotion detection status."""
    return jsonify({
        "available": EMOTION_DETECTION_AVAILABLE,
        "model_type": "DeepFace",
        "emotions": ["Happy", "Fear", "Sad", "Neutral"]
    }), 200

@app.route("/api/reid/status", methods=["GET"])
def api_reid_status():
    """API endpoint to get ReID system statistics and configuration."""
    stats = get_reid_stats()
    system_info = "Production-level ReID with FAISS-powered similarity search"
    if not FAISS_AVAILABLE:
        system_info = "Production-level ReID with cosine similarity (FAISS not available)"
    
    return jsonify({
        "success": True,
        "reid_stats": stats,
        "system": system_info
    }), 200

@app.route("/api/reid/persons", methods=["GET"])
def api_reid_persons():
    """API endpoint to get unique and duplicate persons with their photos."""
    try:
        metadata = safe_load_json(METADATA_FILE, [])
        
        # Organize persons by ID
        persons_dict = {}
        for entry in metadata:
            output_filename = entry.get("output", "")
            faces = entry.get("faces", [])
            
            for face in faces:
                person_id = face.get("person_id", "")
                face_type = face.get("face_type", "unknown")
                face_status = face.get("face_status", "UNKNOWN")
                
                if not person_id:
                    continue
                
                if person_id not in persons_dict:
                    persons_dict[person_id] = {
                        "person_id": person_id,
                        "face_type": face_type,
                        "face_status": face_status,
                        "photos": [],
                        "first_seen": entry.get("timestamp", ""),
                        "total_faces": 0,
                        "avg_similarity": 0.0,
                        "embeddings_count": len(FACE_DB.get(person_id, []))
                    }
                
                # Add photo info
                if output_filename:
                    photo_url = f"/stored_images/{output_filename}"
                    persons_dict[person_id]["photos"].append({
                        "url": photo_url,
                        "filename": output_filename,
                        "timestamp": entry.get("timestamp", ""),
                        "age": face.get("age", ""),
                        "gender": face.get("gender", ""),
                        "emotion": face.get("emotion", ""),
                        "similarity": face.get("similarity", 0.0)
                    })
                
                persons_dict[person_id]["total_faces"] += 1
                persons_dict[person_id]["avg_similarity"] += face.get("similarity", 0.0)
        
        # Calculate average similarity and categorize
        unique_persons = []
        duplicate_persons = []
        
        for person_id, person_data in persons_dict.items():
            if person_data["total_faces"] > 0:
                person_data["avg_similarity"] = round(person_data["avg_similarity"] / person_data["total_faces"], 4)
            
            # Use first photo as thumbnail
            if person_data["photos"]:
                person_data["thumbnail"] = person_data["photos"][0]["url"]
            else:
                person_data["thumbnail"] = None
            
            # Categorize based on face_type
            if person_data["face_type"] == "duplicate" or person_data["total_faces"] > 1:
                duplicate_persons.append(person_data)
            else:
                unique_persons.append(person_data)
        
        return jsonify({
            "success": True,
            "unique_persons": unique_persons,
            "duplicate_persons": duplicate_persons,
            "total_unique": len(unique_persons),
            "total_duplicate": len(duplicate_persons)
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/api/reid/merge_persons", methods=["POST"])
def api_reid_merge_persons():
    """API endpoint to merge duplicate person IDs (e.g., merge ID008 into ID001)."""
    try:
        data = request.get_json()
        source_id = data.get("source_id")  # ID to merge (e.g., ID008)
        target_id = data.get("target_id")   # ID to keep (e.g., ID001)
        
        if not source_id or not target_id:
            return jsonify({
                "success": False,
                "error": "source_id and target_id are required"
            }), 400
        
        if source_id == target_id:
            return jsonify({
                "success": False,
                "error": "Cannot merge person with itself"
            }), 400
        
        # Check if both persons exist
        if source_id not in FACE_DB:
            return jsonify({
                "success": False,
                "error": f"Source person {source_id} not found"
            }), 404
        
        if target_id not in FACE_DB:
            return jsonify({
                "success": False,
                "error": f"Target person {target_id} not found"
            }), 404
        
        # Merge embeddings from source to target
        source_embeddings = FACE_DB[source_id]
        FACE_DB[target_id].extend(source_embeddings)
        
        # Maintain max embeddings per person
        max_emb = REID_CONFIG["max_embeddings_per_person"]
        if len(FACE_DB[target_id]) > max_emb:
            FACE_DB[target_id] = FACE_DB[target_id][-max_emb:]
        
        # Remove source person
        del FACE_DB[source_id]
        REID_STATS["total_embeddings"] = sum(len(emb_list) for emb_list in FACE_DB.values())
        
        # Update metadata: change all source_id to target_id
        metadata = safe_load_json(METADATA_FILE, [])
        updated_count = 0
        for entry in metadata:
            faces = entry.get("faces", [])
            for face in faces:
                if face.get("person_id") == source_id:
                    face["person_id"] = target_id
                    face["reid_status"] = "duplicate"
                    updated_count += 1
        
        safe_save_json(METADATA_FILE, metadata)
        
        # Rebuild FAISS index
        if FAISS_AVAILABLE:
            rebuild_faiss_index()
        
        return jsonify({
            "success": True,
            "message": f"Person {source_id} merged into {target_id}",
            "merged_faces": updated_count,
            "target_embeddings": len(FACE_DB[target_id])
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/api/reid/clear_stats", methods=["POST"])
def api_reid_clear_stats():
    """API endpoint to clear all ReID statistics and reset to 0."""
    global FACE_DB, REID_STATS, FAISS_INDEX, FAISS_INDEX_PERSON_MAP, FAISS_INDEX_EMBEDDING_MAP
    
    try:
        # Clear FACE_DB
        FACE_DB.clear()
        
        # Reset REID_STATS to 0
        REID_STATS["total_matches"] = 0
        REID_STATS["total_new_persons"] = 0
        REID_STATS["total_embeddings"] = 0
        REID_STATS["avg_similarity"] = 0.0
        
        # Clear FAISS index
        if FAISS_AVAILABLE:
            FAISS_INDEX = None
            FAISS_INDEX_PERSON_MAP = []
            FAISS_INDEX_EMBEDDING_MAP = []
        
        # Rebuild empty FAISS index
        if FAISS_AVAILABLE:
            initialize_faiss_index(REID_CONFIG["embedding_dim"])
        
        return jsonify({
            "success": True,
            "message": "All ReID statistics cleared successfully",
            "stats": {
                "total_persons": 0,
                "total_embeddings": 0,
                "total_matches": 0,
                "total_new_persons": 0,
                "avg_similarity": 0.0
            }
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/api/reid/remove_person", methods=["POST"])
def api_reid_remove_person():
    """API endpoint to remove a person and all their data."""
    try:
        data = request.get_json()
        person_id = data.get("person_id")
        
        if not person_id:
            return jsonify({
                "success": False,
                "error": "person_id is required"
            }), 400
        
        # Remove from FACE_DB
        if person_id in FACE_DB:
            del FACE_DB[person_id]
            REID_STATS["total_embeddings"] = sum(len(emb_list) for emb_list in FACE_DB.values())
        
        # Remove from metadata
        metadata = safe_load_json(METADATA_FILE, [])
        updated_metadata = []
        removed_count = 0
        
        for entry in metadata:
            faces = entry.get("faces", [])
            updated_faces = []
            
            for face in faces:
                if face.get("person_id") != person_id:
                    updated_faces.append(face)
                else:
                    removed_count += 1
            
            if updated_faces:
                entry["faces"] = updated_faces
                entry["num_faces"] = len(updated_faces)
                updated_metadata.append(entry)
            elif len(updated_faces) == 0:
                # Remove entry if no faces left
                output_filename = entry.get("output", "")
                if output_filename:
                    image_path = os.path.join(STORED_IMAGES_FOLDER, output_filename)
                    # Safety check: Never delete from source folder
                    is_safe, message = is_safe_to_delete(image_path)
                    if is_safe and os.path.exists(image_path):
                        try:
                            os.remove(image_path)
                        except:
                            pass
                    elif not is_safe:
                        print(f"⚠️ Safety check: {message} - {image_path}")
        
        safe_save_json(METADATA_FILE, updated_metadata)
        
        # Rebuild FAISS index
        if FAISS_AVAILABLE:
            rebuild_faiss_index()
        
        return jsonify({
            "success": True,
            "message": f"Person {person_id} removed successfully",
            "removed_faces": removed_count
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/api/gallery/save", methods=["POST"])
def api_gallery_save():
    """API endpoint to save processed results to Process Gallery."""
    try:
        data = request.get_json()
        results = data.get("results", [])
        statistics = data.get("statistics", {})
        timestamp = data.get("timestamp", datetime.now().isoformat())
        
        if not results:
            return jsonify({
                "success": False,
                "error": "No results to save"
            }), 400
        
        # Load existing gallery
        gallery = safe_load_json(GALLERY_FILE, [])
        
        # Create new gallery entry
        gallery_entry = {
            "id": f"gallery_{int(time.time())}",
            "timestamp": timestamp,
            "statistics": statistics,
            "results": results,
            "total_images": len(results)
        }
        
        # Add to gallery (most recent first)
        gallery.insert(0, gallery_entry)
        
        # Keep only last 50 entries to prevent file from growing too large
        if len(gallery) > 50:
            gallery = gallery[:50]
        
        # Save gallery
        safe_save_json(GALLERY_FILE, gallery)
        
        return jsonify({
            "success": True,
            "message": f"Saved {len(results)} images to Process Gallery",
            "gallery_id": gallery_entry["id"],
            "total_gallery_entries": len(gallery)
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/api/gallery/get", methods=["GET"])
def api_gallery_get():
    """API endpoint to retrieve Process Gallery data."""
    try:
        gallery = safe_load_json(GALLERY_FILE, [])
        
        # Get specific entry if ID provided
        entry_id = request.args.get("id")
        if entry_id:
            entry = next((e for e in gallery if e.get("id") == entry_id), None)
            if entry:
                return jsonify({
                    "success": True,
                    "entry": entry
                }), 200
            else:
                return jsonify({
                    "success": False,
                    "error": "Gallery entry not found"
                }), 404
        
        return jsonify({
            "success": True,
            "gallery": gallery,
            "total_entries": len(gallery)
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/api/gallery/latest", methods=["GET"])
def api_gallery_latest():
    """API endpoint to get the latest auto_process results (for Auto Process Data tab)."""
    try:
        gallery = safe_load_json(GALLERY_FILE, [])
        
        if gallery:
            return jsonify({
                "success": True,
                "latest": gallery[0]  # Most recent entry
            }), 200
        else:
            return jsonify({
                "success": True,
                "latest": None,
                "message": "No gallery entries yet"
            }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/api/gallery/remove", methods=["POST"])
def api_gallery_remove():
    """API endpoint to remove a gallery entry."""
    try:
        data = request.get_json()
        entry_id = data.get("id")
        
        if not entry_id:
            return jsonify({
                "success": False,
                "error": "Entry ID is required"
            }), 400
        
        gallery = safe_load_json(GALLERY_FILE, [])
        original_count = len(gallery)
        gallery = [e for e in gallery if e.get("id") != entry_id]
        
        if len(gallery) == original_count:
            return jsonify({
                "success": False,
                "error": "Gallery entry not found"
            }), 404
        
        safe_save_json(GALLERY_FILE, gallery)
        
        return jsonify({
            "success": True,
            "message": "Gallery entry removed successfully",
            "remaining_entries": len(gallery)
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# Import live detection module
try:
    from live_detection import LiveDetectionManager
    LIVE_DETECTION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Live detection module not available: {e}")
    LIVE_DETECTION_AVAILABLE = False

# Global live detection manager
live_detection_manager = None

def init_live_detection():
    """Initialize live detection manager."""
    global live_detection_manager
    if LIVE_DETECTION_AVAILABLE and live_detection_manager is None:
        try:
            live_detection_manager = LiveDetectionManager(
                face_detection_func=detect_and_extract_faces,
                save_callback=None,  # Save directly in manager
                cropped_dir=CROPPED_DIR
            )
            print("✅ Live detection manager initialized")
        except Exception as e:
            print(f"⚠️ Failed to initialize live detection: {e}")

# Initialize on import
if LIVE_DETECTION_AVAILABLE:
    init_live_detection()

@app.route("/api/live/start", methods=["POST"])
def api_live_start():
    """API endpoint to start live detection."""
    global live_detection_manager
    
    if not LIVE_DETECTION_AVAILABLE:
        return jsonify({
            "success": False,
            "error": "Live detection module not available"
        }), 500
    
    try:
        if live_detection_manager is None:
            init_live_detection()
        
        if live_detection_manager.is_running:
            return jsonify({
                "success": True,
                "message": "Live detection already running",
                "status": "running"
            }), 200
        
        if live_detection_manager.start():
            return jsonify({
                "success": True,
                "message": "Live detection started successfully",
                "status": "running"
            }), 200
        else:
            return jsonify({
                "success": False,
                "error": "Failed to start live detection (camera connection failed)"
            }), 500
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/api/live/stop", methods=["POST"])
def api_live_stop():
    """API endpoint to stop live detection."""
    global live_detection_manager
    
    if not LIVE_DETECTION_AVAILABLE or live_detection_manager is None:
        return jsonify({
            "success": False,
            "error": "Live detection not available"
        }), 500
    
    try:
        if live_detection_manager.stop():
            return jsonify({
                "success": True,
                "message": "Live detection stopped successfully",
                "status": "stopped"
            }), 200
        else:
            return jsonify({
                "success": True,
                "message": "Live detection was not running",
                "status": "stopped"
            }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/api/live/status", methods=["GET"])
def api_live_status():
    """API endpoint to get live detection status."""
    global live_detection_manager
    
    if not LIVE_DETECTION_AVAILABLE or live_detection_manager is None:
        return jsonify({
            "success": True,
            "status": "not_available",
            "is_running": False
        }), 200
    
    try:
        status = live_detection_manager.get_status()
        return jsonify({
            "success": True,
            "status": "running" if status["is_running"] else "stopped",
            **status
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/api/live/latest", methods=["GET"])
def api_live_latest():
    """API endpoint to get the latest frame and detections from live feed."""
    global live_detection_manager
    
    # Try to get from live detection first
    if LIVE_DETECTION_AVAILABLE and live_detection_manager is not None and live_detection_manager.is_running:
        try:
            frame = live_detection_manager.get_latest_frame()
            detections = live_detection_manager.get_latest_detections()
            
            if frame is not None:
                # Save frame as temporary image for serving
                temp_path = os.path.join(STORED_IMAGES_FOLDER, "live_latest.jpg")
                cv2.imwrite(temp_path, frame)
                
                return jsonify({
                    "success": True,
                    "image_url": "/stored_images/live_latest.jpg",
                    "detections": detections,
                    "num_faces": len(detections),
                    "timestamp": datetime.now().isoformat(),
                    "source": "live_stream"
                }), 200
        except Exception as e:
            print(f"⚠️ Error getting live frame: {e}")
    
    # Fallback: Get the most recently processed image from metadata
    try:
        metadata = safe_load_json(METADATA_FILE, [])
        
        if not metadata:
            return jsonify({
                "success": True,
                "image_url": None,
                "detections": [],
                "message": "No processed images yet",
                "source": "metadata"
            }), 200
        
        # Get the most recent entry
        latest_entry = metadata[-1]
        output_filename = latest_entry.get("output", "")
        
        if output_filename:
            image_url = f"/stored_images/{output_filename}"
            faces = latest_entry.get("faces", [])
            
            # Format detections for live view
            detections = []
            for face in faces[:6]:
                detections.append({
                    "age": face.get("age", "Unknown"),
                    "gender": face.get("gender", "Unknown"),
                    "emotion": face.get("emotion", "Unknown"),
                    "person_id": face.get("person_id", ""),
                    "reid_status": face.get("reid_status", "new")
                })
            
            return jsonify({
                "success": True,
                "image_url": image_url,
                "detections": detections,
                "timestamp": latest_entry.get("timestamp", ""),
                "num_faces": latest_entry.get("num_faces", 0),
                "source": "metadata"
            }), 200
        else:
            return jsonify({
                "success": True,
                "image_url": None,
                "detections": [],
                "message": "No image available",
                "source": "metadata"
            }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# Self-Test Function (Optional: Run before app)
def self_test_genders(image_paths, expected_genders):
    """Test fix on your images. expected_genders = ['Male', 'Female']"""
    correct = 0
    for path, expected in zip(image_paths, expected_genders):
        img_bgr = cv2.imread(path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        faces = detect_and_extract_faces(img_rgb)
        if faces:
            gender = faces[0]['gender']
            conf = faces[0]['confidence']['gender']
            print(f"Image {path}: Predicted {gender} ({conf:.2f}) | Expected {expected} | {'✅' if gender == expected else '❌'}")
            if gender == expected:
                correct += 1
    acc = correct / len(image_paths) * 100
    print(f"\nOverall Accuracy: {acc:.1f}%")
if __name__ == "__main__":
    # Run self-test (uncomment and add paths)
    # self_test_genders(['download(1).jpg', '39673568.jpg'], ['Male', 'Female'])
    print("\n" + "="*60)
    print("🚀 Face Detection with Age, Gender, and Emotion...")
    print("="*60)
    print(f"📁 Photo Directory: {CROPPED_DIR}")
    print(f"✅ InsightFace: Loaded")
    print(f"✅ Age & Gender: ON (Active)")
    if EMOTION_DETECTION_AVAILABLE:
        print(f"✅ Emotion Detection: ON (DeepFace)")
    else:
        print(f"❌ Emotion Detection: OFF")
    print("="*60 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)  # Set to False for production speed


