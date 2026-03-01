import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import cv2
from skimage import color
from skimage.feature import graycomatrix, graycoprops
import json

IMG_SIZE = 224

SOIL_TYPES = [
    "Alluvial Soil",
    "Black Soil",
    "Laterite Soil",
    "Red Soil",
    "Yellow Soil",
    "Mountain Soil",
    "Arid Soil"
]

# Load knowledge base
KNOWLEDGE_BASE_PATH = os.path.join(os.path.dirname(__file__), 'soil_knowledge_base.json')
with open(KNOWLEDGE_BASE_PATH, 'r') as kb_file:
    SOIL_KB = json.load(kb_file)

def load_trained_model(model_path):
    return tf.keras.models.load_model(
        model_path,
        custom_objects={
            "preprocess_input": tf.keras.applications.efficientnet_v2.preprocess_input
        }
    )


def predict_images(img_dir, model, class_names=SOIL_TYPES):
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp")

    img_paths = [
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.lower().endswith(valid_exts)
    ]

    if not img_paths:
        return []

    all_predictions = []
    per_image_results = []

    for img_path in img_paths:
        img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img)

        img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array, verbose=0)[0]

        idx = np.argmax(predictions)
        soil = class_names[idx]
        confidence = float(np.max(predictions))

        # --- Deep feature extraction ---
        # Try to get a conv layer, else fallback to penultimate layer
        conv_layers = [l.name for l in model.layers if 'conv' in l.name]
        if conv_layers:
            layer_name = conv_layers[-1]
        else:
            # Fallback: use the second last layer
            if len(model.layers) > 2:
                layer_name = model.layers[-2].name
            else:
                layer_name = model.layers[-1].name
        # Use model.inputs for Sequential models
        intermediate_layer_model = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer(layer_name).output)
        deep_features = intermediate_layer_model.predict(img_array)

        # --- Visual features ---
        visual = compute_visual_features(img_path)

        # --- Rule-based properties ---
        props = estimate_properties(soil, visual)

        # --- Soil health assessment ---
        health = assess_soil_health(soil, props)

        per_image_results.append((
            os.path.basename(img_path), soil, confidence,
            props['moisture'], props['salinity'], props['om_index'], props['ph_tendency'], health
        ))
        all_predictions.append(predictions)

    # Weighted average for the majority class
    from collections import Counter
    predicted_classes = [soil for _, soil, _, _, _, _, _, _ in per_image_results]
    majority_class, _ = Counter(predicted_classes).most_common(1)[0]
    majority_confidences = [conf for (_, soil, conf, *_ ) in per_image_results if soil == majority_class]
    if majority_confidences:
        weights = np.array(majority_confidences)
        weighted_mean = np.average(weights, weights=weights)
        final_confidence = float(weighted_mean)
    else:
        final_confidence = 0.0
    # Aggregate other properties for majority class
    majority_props = [props for (_, soil, _, *props) in per_image_results if soil == majority_class]
    if majority_props:
        moistures = [p[0] for p in majority_props]
        salinities = [p[1] for p in majority_props]
        om_indices = [p[2] for p in majority_props]
        moisture = float(np.mean(moistures))
        salinity = float(np.mean(salinities))
        om_index = float(np.mean(om_indices))
        ph_tendency = majority_props[0][3] if len(majority_props[0]) > 3 else '-'
        health = majority_props[0][4] if len(majority_props[0]) > 4 else '-'
        kb_explanation = majority_props[0][5] if len(majority_props[0]) > 5 else '-'
    else:
        moisture = salinity = om_index = 0.0
        ph_tendency = health = kb_explanation = '-'
    per_image_results.append((
        "Final Decision", majority_class, final_confidence,
        moisture, salinity, om_index, ph_tendency, health, kb_explanation
    ))
    return per_image_results


# --- Deep Feature Extraction ---
def extract_intermediate_features(model, img_array, layer_name=None):
    # Use the last conv layer if not specified
    if layer_name is None:
        layer_name = [l.name for l in model.layers if 'conv' in l.name][-1]
    intermediate_layer_model = tf.keras.Model(inputs=model.inputs,
                                              outputs=model.get_layer(layer_name).output)
    features = intermediate_layer_model.predict(img_array)
    return features


# --- Visual Feature Engine ---
def compute_visual_features(img_path):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    brightness = np.mean(img_rgb)
    color_mean = np.mean(img_rgb, axis=(0,1))  # R, G, B means
    contrast = np.std(img_rgb)
    # Texture using GLCM contrast
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
    texture = graycoprops(glcm, 'contrast')[0,0]
    return {
        'brightness': brightness,
        'r_mean': color_mean[0],
        'g_mean': color_mean[1],
        'b_mean': color_mean[2],
        'contrast': contrast,
        'texture': texture
    }


# --- Rule-based Estimation ---
def estimate_properties(soil_type, visual):
    kb = SOIL_KB.get(soil_type, {})
    # Example formulas (customize as needed)
    moisture = 0.6 * visual['b_mean'] + 0.3 * visual['g_mean'] - 0.2 * visual['r_mean']
    salinity = 0.5 * visual['brightness'] + 0.2 * visual['contrast']
    om_index = (
        2.5 * moisture +
        1.8 * visual['texture'] +
        1.2 * visual['contrast'] +
        0.8 * visual['r_mean'] -
        2.2 * salinity -
        1.5 * visual['brightness']
    )
    # Clamp OM index to realistic range from KB if available
    om_min, om_max = kb.get('om_index_range', [0.5, 8])
    om_index = np.clip(om_index, om_min, om_max)
    # pH tendency from KB
    ph_tendency = kb.get('ph_tendency', 'Neutral')
    return {
        'moisture': moisture,
        'salinity': salinity,
        'om_index': om_index,
        'ph_tendency': ph_tendency,
        'kb_explanation': kb.get('description', '')
    }


# --- Soil Health Assessment ---
def assess_soil_health(soil_type, props):
    kb = SOIL_KB.get(soil_type, {})
    score = props['om_index'] - props['salinity'] + props['moisture']
    # Use KB ranges for health assessment
    moisture_range = kb.get('moisture_range', [0.5, 0.8])
    salinity_range = kb.get('salinity_range', [0.1, 0.3])
    if moisture_range[0] <= props['moisture'] <= moisture_range[1] and salinity_range[0] <= props['salinity'] <= salinity_range[1]:
        health = 'Excellent'
    elif score > 5:
        health = 'Good'
    elif score > 2:
        health = 'Fair'
    else:
        health = 'Poor'
    return health
