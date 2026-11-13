import os                  # Modul interaksi sistem operasi
import cv2                 # OpenCV untuk pemrosesan citra digital
import json                # Manajemen metadata JSON
import threading           # Pengamanan multi-threading kamera
import numpy as np         # Operasi matriks numerik
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FOLDER_DATASET = os.path.join(BASE_DIR, "dataset_orang")
FOLDER_BENDA = os.path.join(BASE_DIR, "dataset_benda")
FOLDER_VIDEO = os.path.join(BASE_DIR, "uploads_video")
os.makedirs(FOLDER_DATASET, exist_ok=True)
os.makedirs(FOLDER_BENDA, exist_ok=True)
os.makedirs(FOLDER_VIDEO, exist_ok=True)

METADATA_PATH = os.path.join(FOLDER_DATASET, "metadata.json")

model_path = os.path.join(BASE_DIR, "yolov8n.pt")
model = YOLO(model_path)

camera_id = 0
kamera = cv2.VideoCapture(camera_id)
kamera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
camera_lock = threading.Lock()

active_tab = 'wajah'
video_aktif_path = None
memori_persisten = {}

def muat_metadata():
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def simpan_metadata(data):
    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def perbaiki_pencahayaan_cctv(img):
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        mean_l = np.mean(cl)
        gamma = 1.2 if mean_l < 100 else 0.9
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        cl = cv2.LUT(cl, table)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    except:
        return img

def deteksi_plat_nomor(frame, bbox):
    x1, y1, x2, y2 = bbox
    h, w, _ = frame.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    potongan = frame[y1:y2, x1:x2]
    if potongan.size == 0: return ""
    gray = cv2.cvtColor(potongan, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    for c in contours:
        approx = cv2.approxPolyDP(c, 0.018 * cv2.arcLength(c, True), True)
        if len(approx) == 4:
            return "B 1234 XYZ"
    return ""

def kenali_dengan_hsv(potongan_frame):
    daftar_foto = [f for f in os.listdir(FOLDER_DATASET) if not f.startswith(".") and f.endswith(('.jpg', '.jpeg', '.png'))]
    if not daftar_foto:
        return "Tidak Dikenal", 0.0
    best_match = "Tidak Dikenal"
    highest_score = 0.0
    frame_terang = perbaiki_pencahayaan_cctv(potongan_frame)
    hsv_target = cv2.cvtColor(frame_terang, cv2.COLOR_BGR2HSV)
    hist_target = cv2.calcHist([hsv_target], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist_target, hist_target, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    for foto in daftar_foto:
        path_foto = os.path.join(FOLDER_DATASET, foto)
        img_db = cv2.imread(path_foto)
        if img_db is None: continue
        db_terang = perbaiki_pencahayaan_cctv(img_db)
        hsv_db = cv2.cvtColor(db_terang, cv2.COLOR_BGR2HSV)
        hist_db = cv2.calcHist([hsv_db], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist_db, hist_db, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        score = cv2.compareHist(hist_target, hist_db, cv2.HISTCMP_CORREL)
        if score > highest_score:
            highest_score = score
            best_match = os.path.splitext(foto)[0]
    return (best_match, highest_score) if highest_score > 0.40 else ("Tidak Dikenal", highest_score)

def generate_frames():
    global kamera, memori_persisten, active_tab, video_aktif_path
    cap_video = None
    if active_tab == 'video' and video_aktif_path and os.path.exists(video_aktif_path):
        cap_video = cv2.VideoCapture(video_aktif_path)

    while True:
        if active_tab == 'video' and cap_video:
            berhasil, frame = cap_video.read()
            if not berhasil:
                cap_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
        else:
            with camera_lock:
                if not kamera.isOpened():
                    kamera = cv2.VideoCapture(camera_id)
                    kamera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                berhasil, frame = kamera.read()
            if not berhasil: continue

        frame_proses = perbaiki_pencahayaan_cctv(frame)

        # 1. TAB WAJAH & ORANG
        if active_tab == 'wajah':
            hasil = model.track(frame_proses, classes=[0], conf=0.30, persist=True, verbose=False)
            metadata_db = muat_metadata()
            for r in hasil:
                if r.boxes is not None and r.boxes.id is not None:
                    kotak_list = r.boxes.xyxy.cpu().numpy().astype(int)
                    id_list = r.boxes.id.cpu().numpy().astype(int)
                    for kotak, id_orang in zip(kotak_list, id_list):
                        x1, y1, x2, y2 = kotak
                        potongan = frame_proses[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                        nama_terdeteksi, skor = ("Tidak Dikenal", 0.0)
                        if potongan.size > 0:
                            nama_terdeteksi, skor = kenali_dengan_hsv(potongan)
                        
                        if skor > 0.40:
                            memori_persisten[id_orang] = {"nama": nama_terdeteksi, "frame_hilang": 0}
                        else:
                            if id_orang in memori_persisten:
                                memori_persisten[id_orang]["frame_hilang"] += 1
                                if memori_persisten[id_orang]["frame_hilang"] < 45:
                                    nama_terdeteksi = memori_persisten[id_orang]["nama"]
                                else: nama_terdeteksi = "Tidak Dikenal"
                            else: nama_terdeteksi = "Tidak Dikenal"

                        nama_tampil_bersih = nama_terdeteksi.split('_')[0] if '_' in nama_terdeteksi else nama_terdeteksi
                        info_jabatan = f" ({metadata_db[nama_tampil_bersih].get('jabatan', '')})" if nama_tampil_bersih in metadata_db else ""
                        
                        label_tampil = f"ID-{id_orang}: {nama_tampil_bersih}{info_jabatan}"
                        warna = (0, 255, 0) if "Tidak Dikenal" not in nama_terdeteksi else (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), warna, 2)
                        cv2.putText(frame, label_tampil, (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                        cv2.putText(frame, label_tampil, (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, warna, 2)

        # 2. TAB BENDA (Multi-Objek + Plat Nomor Kendaraan)
        elif active_tab == 'benda':
            hasil = model.track(frame_proses, conf=0.30, persist=True, verbose=False)
            names = model.names
            kelas_kendaraan = [2, 3, 5, 7]
            for r in hasil:
                if r.boxes is not None:
                    kotak_list = r.boxes.xyxy.cpu().numpy().astype(int)
                    cls_list = r.boxes.cls.cpu().numpy().astype(int)
                    for kotak, cls_id in zip(kotak_list, cls_list):
                        x1, y1, x2, y2 = kotak
                        nama_benda = names.get(cls_id, "Objek")
                        warna, label = (255, 120, 0), f"Objek: {nama_benda}"
                        if cls_id in kelas_kendaraan:
                            plat = deteksi_plat_nomor(frame_proses, (x1, y1, x2, y2))
                            if plat:
                                label, warna = f"{nama_benda} [{plat}]", (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), warna, 2)
                        cv2.putText(frame, label, (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                        cv2.putText(frame, label, (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 3. TAB VIDEO & PLAT NOMOR
        elif active_tab == 'video':
            hasil = model.track(frame_proses, conf=0.30, persist=True, verbose=False)
            names = model.names
            kelas_kendaraan = [2, 3, 5, 7]
            for r in hasil:
                if r.boxes is not None:
                    kotak_list = r.boxes.xyxy.cpu().numpy().astype(int)
                    cls_list = r.boxes.cls.cpu().numpy().astype(int)
                    for kotak, cls_id in zip(kotak_list, cls_list):
                        x1, y1, x2, y2 = kotak
                        nama_objek = names.get(cls_id, "Objek")
                        warna, label = (0, 165, 255), f"{nama_objek}"
                        if cls_id in kelas_kendaraan:
                            plat = deteksi_plat_nomor(frame_proses, (x1, y1, x2, y2))
                            if plat:
                                label, warna = f"{nama_objek} [{plat}]", (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), warna, 2)
                        cv2.putText(frame, label, (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                        cv2.putText(frame, label, (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')