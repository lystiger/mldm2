import argparse
import csv
import os
import time
import urllib.request

import pandas as pd
import serial

# --- CẤU HÌNH ---
PORT = "/dev/ttyACM0"  # Nhớ đổi lại đúng cổng trên máy Linux
BAUD = 115200
NUM_SAMPLES = 100      # Thu 100 mẫu mỗi lần
GLOVE_FILE = "Data/glove_dataset.csv"
CAMERA_FILE = "Data/hand_landmarks.csv"
FUSED_FILE = "Data/fused_dataset.csv"
FUSE_TOLERANCE_MS = 30
HAND_MODEL_PATH = "Data/hand_landmarker.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def collect_glove_data(port, baud, num_samples, output_file):
    label = input("👉 Nhập tên cử chỉ bạn chuẩn bị làm (VD: A, B, C): ").strip().upper()

    print(f"\n⏳ Chuẩn bị tư thế tay cho chữ '{label}'. Bắt đầu thu thập sau 3 giây...")
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)

    try:
        ser = serial.Serial(port, baud, timeout=1)
        ser.reset_input_buffer()
    except Exception as e:
        print(f"❌ Không mở được cổng {port}. Lỗi: {e}")
        return

    file_exists = os.path.isfile(output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            header = [
                "timestamp_ms", "ax", "ay", "az", "gx", "gy", "gz",
                "f1", "f2", "f3", "f4", "f5", "label"
            ]
            writer.writerow(header)

        print("\n🔴 ĐANG GHI DỮ LIỆU... GIỮ NGUYÊN TƯ THẾ NHÉ!")
        count = 0
        while count < num_samples:
            try:
                line = ser.readline().decode("utf-8").strip()
                if not line or any(word in line for word in ["Calibrating", "READY", "timestamp"]):
                    continue

                row = line.split(",")
                if len(row) == 12:
                    row.append(label)
                    writer.writerow(row)
                    count += 1
                    if count % 20 == 0:
                        print(f"   Đã thu được {count}/{num_samples} mẫu...")
            except UnicodeDecodeError:
                continue

    print(f"\n✅ XONG! Đã lưu thành công {num_samples} mẫu của chữ '{label}' vào file {output_file}.")
    ser.close()


def _camera_columns():
    return (
        ["ts", "gesture", "L_exist", "R_exist"]
        + [f"L_{a}{i}" for i in range(21) for a in ["x", "y", "z"]]
        + [f"R_{a}{i}" for i in range(21) for a in ["x", "y", "z"]]
    )


def _extract_hand_task(hand_landmarks):
    feat = []
    for lm in hand_landmarks:
        feat.extend([lm.x, lm.y, lm.z])
    return feat


def collect_camera_data(output_file, camera_id, model_path):
    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except Exception as e:
        print(f"❌ Thiếu package camera/CV: {e}")
        print("   Cài thêm: pip install opencv-python mediapipe")
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if not os.path.isfile(model_path):
        print("⬇️ Downloading hand landmarker model...")
        urllib.request.urlretrieve(HAND_MODEL_URL, model_path)

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"❌ Không mở được camera id={camera_id}")
        landmarker.close()
        return

    current_gesture = "none"
    collecting = False
    records = []

    print(
        "\nq : quit\n"
        "c : toggle collect\n"
        "1-9 : set gesture (gesture_1, gesture_2...)\n"
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts_ms = int(time.time() * 1000)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = landmarker.detect_for_video(mp_image, ts_ms)

        l_feat = [0.0] * 63
        r_feat = [0.0] * 63
        l_exist = 0
        r_exist = 0

        if res.hand_landmarks:
            h, w = frame.shape[:2]
            for hand_lm, handedness in zip(res.hand_landmarks, res.handedness):
                label = handedness[0].category_name
                feat = _extract_hand_task(hand_lm)

                if label == "Left":
                    l_feat = feat
                    l_exist = 1
                else:
                    r_feat = feat
                    r_exist = 1

                for lm in hand_lm:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

        if collecting:
            row = [ts_ms / 1000.0, current_gesture, l_exist, r_exist] + l_feat + r_feat
            records.append(row)

        cv2.putText(
            frame,
            f"Gesture: {current_gesture} | Collecting: {collecting}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if collecting else (0, 0, 255),
            2,
        )
        cv2.imshow("CV Collector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            collecting = not collecting
            print("Collecting:", collecting)
        elif ord("1") <= key <= ord("9"):
            current_gesture = f"gesture_{key - ord('0')}"
            print("Gesture set:", current_gesture)

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

    if not records:
        print("⚠️ Không có dữ liệu camera nào được lưu (records=0).")
        return

    df = pd.DataFrame(records, columns=_camera_columns())
    df.to_csv(output_file, index=False)
    print(f"✅ Camera done. Saved {df.shape[0]} rows -> {output_file}")


def collect_sync_data(port, baud, camera_id, glove_out, camera_out, model_path):
    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except Exception as e:
        print(f"❌ Thiếu package camera/CV: {e}")
        print("   Cài thêm: pip install opencv-python mediapipe")
        return

    try:
        ser = serial.Serial(port, baud, timeout=0.05)
        ser.reset_input_buffer()
    except Exception as e:
        print(f"❌ Không mở được cổng serial {port}. Lỗi: {e}")
        return

    os.makedirs(os.path.dirname(glove_out), exist_ok=True)
    os.makedirs(os.path.dirname(camera_out), exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if not os.path.isfile(model_path):
        print("⬇️ Downloading hand landmarker model...")
        urllib.request.urlretrieve(HAND_MODEL_URL, model_path)

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"❌ Không mở được camera id={camera_id}")
        landmarker.close()
        ser.close()
        return

    current_gesture = "gesture_1"
    collecting = False
    cam_records = []
    glove_records = []

    print(
        "\nq : quit\n"
        "c : toggle collect BOTH (camera + glove)\n"
        "1-9 : set gesture label\n"
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts_ms = int(time.time() * 1000)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = landmarker.detect_for_video(mp_image, ts_ms)

        l_feat = [0.0] * 63
        r_feat = [0.0] * 63
        l_exist = 0
        r_exist = 0

        if res.hand_landmarks:
            h, w = frame.shape[:2]
            for hand_lm, handedness in zip(res.hand_landmarks, res.handedness):
                label = handedness[0].category_name
                feat = _extract_hand_task(hand_lm)

                if label == "Left":
                    l_feat = feat
                    l_exist = 1
                else:
                    r_feat = feat
                    r_exist = 1

                for lm in hand_lm:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

        if collecting:
            cam_row = [ts_ms / 1000.0, current_gesture, l_exist, r_exist] + l_feat + r_feat
            cam_records.append(cam_row)

            while ser.in_waiting > 0:
                try:
                    line = ser.readline().decode("utf-8").strip()
                except UnicodeDecodeError:
                    continue
                if not line or any(word in line for word in ["Calibrating", "READY", "timestamp"]):
                    continue
                row = line.split(",")
                if len(row) == 12:
                    row.append(current_gesture)
                    glove_records.append(row)

        cv2.putText(
            frame,
            f"Gesture: {current_gesture} | Collecting: {collecting}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if collecting else (0, 0, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Cam rows: {len(cam_records)} | Glove rows: {len(glove_records)}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )
        cv2.imshow("Sync Collector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            collecting = not collecting
            print("Collecting BOTH:", collecting)
        elif ord("1") <= key <= ord("9"):
            current_gesture = f"gesture_{key - ord('0')}"
            print("Gesture set:", current_gesture)

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    ser.close()

    if cam_records:
        df_cam = pd.DataFrame(cam_records, columns=_camera_columns())
        df_cam.to_csv(camera_out, index=False)
        print(f"✅ Camera saved: {df_cam.shape[0]} rows -> {camera_out}")
    else:
        print("⚠️ Camera records = 0")

    if glove_records:
        glove_cols = [
            "timestamp_ms", "ax", "ay", "az", "gx", "gy", "gz",
            "f1", "f2", "f3", "f4", "f5", "label",
        ]
        df_glove = pd.DataFrame(glove_records, columns=glove_cols)
        df_glove.to_csv(glove_out, index=False)
        print(f"✅ Glove saved: {df_glove.shape[0]} rows -> {glove_out}")
    else:
        print("⚠️ Glove records = 0")


def fuse_glove_with_camera(glove_file, camera_file, output_file, tolerance_ms):
    if not os.path.isfile(glove_file):
        print(f"❌ Không tìm thấy file glove: {glove_file}")
        return
    if not os.path.isfile(camera_file):
        print(f"❌ Không tìm thấy file camera: {camera_file}")
        print("   Hãy export từ notebook vào Data/hand_landmarks.csv rồi chạy lại mode fuse.")
        return

    df_glove = pd.read_csv(glove_file)
    df_cam = pd.read_csv(camera_file)

    required_glove = {"timestamp_ms", "label"}
    if not required_glove.issubset(df_glove.columns):
        print(f"❌ File glove cần có cột: {sorted(required_glove)}")
        return

    if "ts" in df_cam.columns:
        df_cam = df_cam.rename(columns={"ts": "timestamp_s"})
    if "timestamp_s" not in df_cam.columns:
        print("❌ File camera cần có cột thời gian 'ts' hoặc 'timestamp_s'.")
        return

    # Đưa về cùng đơn vị ms để join theo thời gian gần nhất.
    df_glove["timestamp_ms"] = pd.to_numeric(df_glove["timestamp_ms"], errors="coerce")
    df_cam["timestamp_ms"] = pd.to_numeric(df_cam["timestamp_s"], errors="coerce") * 1000.0

    df_glove = df_glove.dropna(subset=["timestamp_ms"]).sort_values("timestamp_ms")
    df_cam = df_cam.dropna(subset=["timestamp_ms"]).sort_values("timestamp_ms")

    df_fused = pd.merge_asof(
        df_glove,
        df_cam,
        on="timestamp_ms",
        direction="nearest",
        tolerance=float(tolerance_ms)
    )

    if "label_x" in df_fused.columns and "label_y" in df_fused.columns:
        df_fused = df_fused.rename(columns={"label_x": "label_glove", "label_y": "label_camera"})

    before = len(df_fused)
    df_fused = df_fused.dropna()
    after = len(df_fused)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_fused.to_csv(output_file, index=False)

    print(f"✅ Fuse xong: {after}/{before} dòng hợp lệ.")
    print(f"📁 Saved: {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect glove sensor data and fuse with camera landmarks."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    collect_p = subparsers.add_parser("collect", help="Collect glove data from serial.")
    collect_p.add_argument("--port", default=PORT)
    collect_p.add_argument("--baud", type=int, default=BAUD)
    collect_p.add_argument("--samples", type=int, default=NUM_SAMPLES)
    collect_p.add_argument("--out", default=GLOVE_FILE)

    collect_cv_p = subparsers.add_parser("collect-cv", help="Collect camera hand landmarks.")
    collect_cv_p.add_argument("--out", default=CAMERA_FILE)
    collect_cv_p.add_argument("--camera-id", type=int, default=0)
    collect_cv_p.add_argument("--model", default=HAND_MODEL_PATH)

    collect_sync_p = subparsers.add_parser(
        "collect-sync",
        help="Collect camera landmarks + glove sensor with one toggle key.",
    )
    collect_sync_p.add_argument("--port", default=PORT)
    collect_sync_p.add_argument("--baud", type=int, default=BAUD)
    collect_sync_p.add_argument("--camera-id", type=int, default=0)
    collect_sync_p.add_argument("--glove-out", default=GLOVE_FILE)
    collect_sync_p.add_argument("--camera-out", default=CAMERA_FILE)
    collect_sync_p.add_argument("--model", default=HAND_MODEL_PATH)

    fuse_p = subparsers.add_parser("fuse", help="Fuse glove CSV with camera CSV by timestamp.")
    fuse_p.add_argument("--glove", default=GLOVE_FILE)
    fuse_p.add_argument("--camera", default=CAMERA_FILE)
    fuse_p.add_argument("--out", default=FUSED_FILE)
    fuse_p.add_argument("--tol-ms", type=float, default=FUSE_TOLERANCE_MS)

    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "collect":
        collect_glove_data(args.port, args.baud, args.samples, args.out)
    elif args.mode == "collect-cv":
        collect_camera_data(args.out, args.camera_id, args.model)
    elif args.mode == "collect-sync":
        collect_sync_data(
            args.port,
            args.baud,
            args.camera_id,
            args.glove_out,
            args.camera_out,
            args.model,
        )
    elif args.mode == "fuse":
        fuse_glove_with_camera(args.glove, args.camera, args.out, args.tol_ms)


if __name__ == "__main__":
    main()
