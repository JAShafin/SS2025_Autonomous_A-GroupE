#!/usr/bin/env python3
"""
CARLA template: constant forward speed + steering from student model.
"""

import carla
import random
import time
import sys
import math
import joblib
import numpy as np
import os
import cv2  # required for image processing

# ------------------------ CONFIGURATION --------------------------------------
HOST            = "localhost"
PORT            = 2000
TIMEOUT_S       = 5.0          # seconds

THROTTLE        = 0.5          # constant forward throttle (0..1)
DEFAULT_STEER   = 0.0          # fallback if no camera frame yet
PRINT_EVERY_N   = 30           # console frames between logs
# -----------------------------------------------------------------------------


# ------------------------------------------------------------------ STUDENTS --
def predict_steering(img):
    """
    Predict steering using trained SVM model.

    Parameters
    ----------
    img : carla.Image

    Returns
    -------
    float : -1 (left), 0 (straight), or 1 (right)
    """
    if not hasattr(predict_steering, "_model"):
        model_path = "groupAsvm.joblib"
        if not os.path.isfile(model_path):
            print(f"[WARN] Model file '{model_path}' not found.")
            predict_steering._model = None
        else:
            predict_steering._model = joblib.load(model_path)
            print(f"[INFO] Loaded model from '{model_path}'")

    model = predict_steering._model
    if model is not None:
        try:
            img_data = np.frombuffer(img.raw_data, dtype=np.uint8)
            img_data = img_data.reshape((img.height, img.width, 4))
            bgr = img_data[:, :, :3]
            resized = cv2.resize(bgr, (64, 64))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            flat = gray.flatten().reshape(1, -1)

            label = model.predict(flat)[0]

            if label == 0:
                return -1.0  # left
            elif label == 1:
                return 0.0   # straight
            elif label == 2:
                return 1.0   # right
            else:
                return 0.0

        except Exception as e:
            print(f"[ERR] Prediction failed: {e}")
            return 0.0

    return 0.0
# -----------------------------------------------------------------------------


# ---------------------------- UTILITIES --------------------------------------
def parent_of(actor):
    if hasattr(actor, "get_parent"):
        return actor.get_parent()
    return getattr(actor, "parent", None)

def ang_diff_deg(a, b):
    return (a - b + 180.0) % 360.0 - 180.0

def pick_center_camera(world, vehicle):
    v_yaw = vehicle.get_transform().rotation.yaw
    best = None
    for s in world.get_actors().filter("sensor.camera.rgb"):
        p = parent_of(s)
        if p and p.id == vehicle.id:
            delta = abs(ang_diff_deg(s.get_transform().rotation.yaw, v_yaw))
            if best is None or delta < best[0]:
                best = (delta, s)
    return best[1] if best else None
# -----------------------------------------------------------------------------


def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(TIMEOUT_S)
    world  = client.get_world()

    vehicles = world.get_actors().filter("vehicle.*")
    if not vehicles:
        print("No vehicles found. Start a scene first.")
        return
    vehicle = vehicles[0]
    print("Controlling vehicle id=%d type=%s" % (vehicle.id, vehicle.type_id))
    vehicle.set_autopilot(False)

    camera = pick_center_camera(world, vehicle)
    if camera is None:
        print("No center RGB camera attached to the vehicle.")
        return
    print("Using camera id=%d for live feed" % camera.id)

    state = {"frames": 0, "first_ts": None, "latest_img": None}

    def cam_cb(img):
        state["latest_img"] = img
        state["frames"] += 1
        if state["frames"] % PRINT_EVERY_N == 0:
            if state["first_ts"] is None:
                state["first_ts"] = img.timestamp
            elapsed = img.timestamp - state["first_ts"]
            fps = state["frames"] / elapsed if elapsed else 0.0
            print("camera frames: %d   %.1f FPS" % (state["frames"], fps))

    camera.listen(cam_cb)

    try:
        while True:
            img = state["latest_img"]
            if img is not None:
                steer = float(max(-1.0, min(1.0, predict_steering(img))))
            else:
                steer = DEFAULT_STEER
            vehicle.apply_control(carla.VehicleControl(throttle=THROTTLE,
                                                       steer=steer))
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopping.")

    finally:
        camera.stop()
        vehicle.apply_control(carla.VehicleControl(brake=1.0))

if __name__ == "__main__":
    try:
        main()
    except RuntimeError as err:
        sys.stderr.write("[ERROR] " + str(err) + "\n"
                         "Is the CARLA server running on this host/port?\n")