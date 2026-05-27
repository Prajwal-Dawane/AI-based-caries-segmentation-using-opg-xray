from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import torch
import numpy as np
from werkzeug.utils import secure_filename
import traceback
import sys

try:
    import torchvision.ops
    TORCHVISION_LOADED = True
except ImportError:
    TORCHVISION_LOADED = False
    print("Warning: torchvision.ops could not be imported. NMS errors are likely.")


# 1.1 Model Path
MODEL_PATH = "mask_rcnn_traced_cpu_accuracy.pt" 
MODEL_LOAD_ERROR = None
model = None # Initialize model globally

# 1.2 Class Definitions and Colors
CLASSES = ['Segmentation', 'C1', 'C2', 'C3', 'H', 'V']
CUSTOM_COLORS = {
    'C1': (0, 255, 255),  # Yellow (BGR)
    'C2': (0, 165, 255),  # Orange (BGR)
    'C3': (0, 0, 255),    # Red (BGR)
    'DEFAULT': (100, 100, 100)
}
TARGET_CLASS_NAMES = ['C1', 'C2', 'C3']
TARGET_CLASS_IDS = [CLASSES.index(name) for name in TARGET_CLASS_NAMES]


# Attempt to load the TorchScript Model during startup
try:
    # map_location="cpu" ensures it runs without a GPU requirement
    model = torch.jit.load(MODEL_PATH, map_location="cpu")
    model.eval()
    print(f"✅ TorchScript Model loaded successfully (CPU: {sys.version.split()[0]} | Torch: {torch.__version__}).")
except Exception as e:
    MODEL_LOAD_ERROR = str(e)
    if "Unknown builtin op: torchvision::nms" in MODEL_LOAD_ERROR:
        MODEL_LOAD_ERROR = (
            "Model Loading Failed (NMS Error). "
            "This usually means the TorchScript file was traced with a version "
            "of PyTorch/TorchVision incompatible with your local environment. "
            "Prediction functionality will not work."
        )
    print(f"❌ Error loading model: {MODEL_LOAD_ERROR}")


# 1.4 Preprocessing (C, H, W tensor)
def preprocess_image(img):
    """
    Converts BGR image (from cv2.imread) to an RGB float tensor (C, H, W).
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Convert to Tensor (H, W, C) -> (C, H, W), float
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
    return tensor


# 1.5 Custom Visualization (Core Logic)
def draw_masks(image, raw_outputs, score_threshold=0.5):
    """
    Draws instance segmentation masks and labels using the raw tuple output
    from the TorchScript model, filtering for TARGET_CLASS_IDS.
    """
    # Assuming output order: (Boxes, Classes, Masks, Scores)
    try:
        # Use .detach() on all output tensors before converting to numpy
        pred_boxes = raw_outputs[0].cpu().detach().squeeze(0) 
        pred_classes = raw_outputs[1].cpu().detach().squeeze(0).long() 
        pred_masks_logits = raw_outputs[2].cpu().detach().squeeze(0) 
        pred_scores = raw_outputs[3].cpu().detach().squeeze(0) 
    except Exception as e:
        print(f"❌ Error during output tensor extraction: {e}")
        return image # Return original image on failure

    # --- Apply Filtering ---
    keep_score_filter = pred_scores > score_threshold
    
    # Keep only target classes (C1, C2, C3)
    target_ids_tensor = torch.tensor(TARGET_CLASS_IDS, device=pred_classes.device)
    keep_class_filter = torch.isin(pred_classes, target_ids_tensor)
    
    # Combine filters
    keep_indices = keep_score_filter & keep_class_filter
    
    # Filter and convert to numpy
    # FIX: .detach() added in the tensor extraction above solves the RuntimeError
    masks = (pred_masks_logits[keep_indices] > 0.0).numpy().astype(np.bool_)
    labels = pred_classes[keep_indices].numpy().astype(int)
    scores = pred_scores[keep_indices].numpy()
    
    
    # --- Drawing Logic ---
    img_out = image.copy()

    for i in range(len(masks)):
        mask = masks[i]
        class_id = labels[i]
        score = scores[i]
        
        class_name = CLASSES[class_id]
        bgr_color = CUSTOM_COLORS.get(class_name, CUSTOM_COLORS['DEFAULT'])

        # Apply mask with transparency
        colored_mask = np.zeros_like(img_out, dtype=np.uint8)
        colored_mask[mask] = bgr_color
        img_out = cv2.addWeighted(img_out, 1.0, colored_mask, 0.5, 0)

        # Find mask contour and draw bounding box/label
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            c = max(contours, key=cv2.contourArea) 
            x, y, w, h = cv2.boundingRect(c)
            
            # Draw bounding box
            cv2.rectangle(img_out, (x, y), (x + w, y + h), bgr_color, 2)
            
            # Put label text
            label_text = f"{class_name} ({score:.2f})"
            cv2.putText(img_out, label_text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr_color, 2, cv2.LINE_AA)

    return img_out

# =======================================================
# 2. FLASK APP SETUP AND ROUTES
# =======================================================
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["RESULT_FOLDER"] = "static/results"
os.makedirs("static", exist_ok=True)
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["RESULT_FOLDER"], exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html", model_error=MODEL_LOAD_ERROR)


@app.route("/predict", methods=["POST"])
def predict():
    global MODEL_LOAD_ERROR

    # Check for model load error first
    if MODEL_LOAD_ERROR:
        return render_template("index.html", model_error=MODEL_LOAD_ERROR), 500

    if "file" not in request.files:
        return redirect(request.url)
    
    file = request.files["file"]
    if file.filename == "":
        return redirect(request.url)
    
    filename = secure_filename(file.filename)
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    
    try:
        file.save(image_path)
        img = cv2.imread(image_path)
        
        tensor = preprocess_image(img)
        outputs = model(tensor)
        
        score_threshold = 0.4
        result_img = draw_masks(img, outputs, score_threshold=score_threshold)

        result_filename = f"result_{filename}"
        result_path = os.path.join(app.config["RESULT_FOLDER"], result_filename)
        cv2.imwrite(result_path, result_img)

        return render_template("index.html",
                               uploaded_image=url_for('static', filename=f'uploads/{filename}'),
                               result_image=url_for('static', filename=f'results/{result_filename}'))

    except Exception as e:
        error_msg = f"Prediction failed due to runtime error: {e}"
        print(f"Prediction Error: {traceback.format_exc()}")
        return render_template("index.html", error_message=error_msg), 500


if __name__ == "__main__":
    app.run(debug=True)
