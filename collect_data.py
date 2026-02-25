import argparse
import csv
import os
import time

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
    elif args.mode == "fuse":
        fuse_glove_with_camera(args.glove, args.camera, args.out, args.tol_ms)


if __name__ == "__main__":
    main()
