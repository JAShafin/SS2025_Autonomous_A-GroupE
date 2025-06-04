import cv2
from skimage.feature import hog

def predict_steering(img):
    """
    Predict steering using the trained SVM model and HOG features.
    """
    if not hasattr(predict_steering, "_model"):
        model_path = "C:/Users/bigbo/Downloads/teamE_dataset/svm_lane_model.joblib"
        if not os.path.isfile(model_path):
            print(f"[WARN] SVM file '{model_path}' not found – only random steering will be used.")
            predict_steering._model = None
        else:
            predict_steering._model = joblib.load(model_path)
            print(f"[INFO] Loaded SVM from '{model_path}'")

    model = predict_steering._model
    if model is not None:
        try:
            # Decode CARLA image to NumPy array
            img_data = np.frombuffer(img.raw_data, dtype=np.uint8)
            img_data = img_data.reshape((img.height, img.width, 4))
            bgr = img_data[:, :, :3]

            # Preprocess: resize → grayscale → HOG
            resized = cv2.resize(bgr, (64, 64))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            features = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)

            # Predict steering
            pred = float(model.predict([features])[0])
            pred_clipped = max(-1.0, min(1.0, pred))
            print(f"SVM steering prediction: {pred_clipped:.3f}")
            return pred_clipped

        except Exception as e:
            print(f"[ERR] SVM predict failed: {e}")
            return 0.0

    return 0.0  # fallback if model not available
