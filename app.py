import os
import cv2
from datetime import datetime
from flask import Flask, Response, request, redirect, url_for, render_template, jsonify
from werkzeug.utils import secure_filename

import detector

app = Flask(__name__)

@app.route('/')
def index():
    daftar_file_mentah = [f for f in os.listdir(detector.FOLDER_DATASET) if not f.startswith(".") and f.endswith(('.jpg', '.jpeg', '.png'))]
    metadata_db = detector.muat_metadata()
    data_orang_lengkap = []
    
    for f in daftar_file_mentah:
        nama_bersih = f.split('_')[0] if '_' in f else os.path.splitext(f)[0]
        info = metadata_db.get(nama_bersih, {"jabatan": "-", "jenis_kelamin": "-"})
        data_orang_lengkap.append({
            "filename": f,
            "nama": nama_bersih,
            "jabatan": info.get("jabatan", "-"),
            "jenis_kelamin": info.get("jenis_kelamin", "-")
        })

    daftar_benda = [f for f in os.listdir(detector.FOLDER_BENDA) if not f.startswith(".") and f.endswith(('.jpg', '.jpeg', '.png'))]
    daftar_video = [f for f in os.listdir(detector.FOLDER_VIDEO) if not f.startswith(".")]

    return render_template('index.html', 
                           daftar_orang=data_orang_lengkap, 
                           daftar_benda=daftar_benda, 
                           daftar_video=daftar_video,
                           active_tab=detector.active_tab,
                           active_cam=detector.camera_id)

@app.route('/video_feed')
def video_feed():
    return Response(detector.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_tab', methods=['POST'])
def set_tab():
    detector.active_tab = request.form.get('tab', 'wajah')
    return redirect(url_for('index'))

@app.route('/set_camera', methods=['POST'])
def set_camera():
    new_id = int(request.form.get('camera_id', 0))
    with detector.camera_lock:
        if detector.camera_id != new_id:
            detector.camera_id = new_id
            if detector.kamera.isOpened(): detector.kamera.release()
            detector.kamera = cv2.VideoCapture(detector.camera_id)
            detector.kamera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            detector.kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return redirect(url_for('index'))

@app.route('/set_video', methods=['POST'])
def set_video():
    nama = request.form.get('nama_video', '')
    if nama:
        detector.video_aktif_path = os.path.join(detector.FOLDER_VIDEO, nama)
        detector.active_tab = 'video'
    return redirect(url_for('index'))

@app.route('/upload_orang', methods=['POST'])
def upload_orang():
    nama = request.form.get('nama', '').strip()
    jabatan = request.form.get('jabatan', '').strip()
    jenis_kelamin = request.form.get('jenis_kelamin', '')
    file = request.files.get('foto')
    
    if nama and file and file.filename != '':
        ext = os.path.splitext(file.filename)[1]
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        nama_file_unik = f"{secure_filename(nama)}_{timestamp_str}{ext}"
        file.save(os.path.join(detector.FOLDER_DATASET, nama_file_unik))
        
        db = detector.muat_metadata()
        db[nama] = {
            "jabatan": jabatan or "-", 
            "jenis_kelamin": jenis_kelamin or "-", 
            "kapan_difoto": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        detector.simpan_metadata(db)
    return redirect(url_for('index'))

@app.route('/edit_orang', methods=['POST'])
def edit_orang():
    try:
        nama_lama = request.form.get('nama_lama', '').strip()
        nama_baru = request.form.get('nama_baru', '').strip()
        jabatan_baru = request.form.get('jabatan', '').strip()
        gender_baru = request.form.get('jenis_kelamin', '')
        file_foto_baru = request.files.get('foto_baru')

        if not nama_lama or not nama_baru:
            return redirect(url_for('index'))

        db = detector.muat_metadata()

        if nama_lama != nama_baru:
            if nama_lama in db:
                db[nama_baru] = db.pop(nama_lama)
            else:
                db[nama_baru] = {}
        
        if nama_baru not in db:
            db[nama_baru] = {}
            
        db[nama_baru]["jabatan"] = jabatan_baru or "-"
        db[nama_baru]["jenis_kelamin"] = gender_baru or "-"
        db[nama_baru]["terakhir_diubah"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        detector.simpan_metadata(db)

        if file_foto_baru and file_foto_baru.filename != '':
            ext = os.path.splitext(file_foto_baru.filename)[1]
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            nama_file_baru = f"{secure_filename(nama_baru)}_{timestamp_str}{ext}"
            
            for f in os.listdir(detector.FOLDER_DATASET):
                if f == nama_lama + ".jpg" or f.startswith(nama_lama + "_") or f.startswith(nama_lama + "."):
                    p_lama = os.path.join(detector.FOLDER_DATASET, f)
                    if os.path.exists(p_lama): os.remove(p_lama)
            
            file_foto_baru.save(os.path.join(detector.FOLDER_DATASET, nama_file_baru))
        else:
            if nama_lama != nama_baru:
                for f in os.listdir(detector.FOLDER_DATASET):
                    if f == nama_lama + ".jpg" or f.startswith(nama_lama + "_") or f.startswith(nama_lama + "."):
                        sifat_belakang = f[len(nama_lama):]
                        path_lama = os.path.join(detector.FOLDER_DATASET, f)
                        path_baru = os.path.join(detector.FOLDER_DATASET, secure_filename(nama_baru) + sifat_belakang)
                        if os.path.exists(path_lama): os.rename(path_lama, path_baru)

    except Exception as e:
        print("Error saat edit:", e)

    return redirect(url_for('index'))

@app.route('/save_snapshot', methods=['POST'])
def save_snapshot():
    try:
        data = request.json
        nama = data.get('nama', '').strip()
        jabatan = data.get('jabatan', '').strip()
        gender = data.get('jenis_kelamin', '')

        if not nama:
            return jsonify({"status": "error", "message": "Nama kosong"})

        with detector.camera_lock:
            if detector.kamera and detector.kamera.isOpened():
                berhasil, frame = detector.kamera.read()
            else:
                berhasil = False

        if not berhasil or frame is None:
            return jsonify({"status": "error", "message": "Kamera tidak aktif"})

        frame_proses = detector.perbaiki_pencahayaan_cctv(frame)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        nama_file_unik = f"{secure_filename(nama)}_{timestamp_str}.jpg"
        file_path = os.path.join(detector.FOLDER_DATASET, nama_file_unik)
        cv2.imwrite(file_path, frame_proses)

        db = detector.muat_metadata()
        db[nama] = {
            "jabatan": jabatan or "-", 
            "jenis_kelamin": gender or "-", 
            "kapan_difoto": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        detector.simpan_metadata(db)

        return jsonify({"status": "success", "filename": nama_file_unik})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/upload_benda', methods=['POST'])
def upload_benda():
    nama, file = request.form.get('nama_benda', '').strip(), request.files.get('foto_benda')
    if nama and file and file.filename != '':
        file.save(os.path.join(detector.FOLDER_BENDA, secure_filename(nama) + os.path.splitext(file.filename)[1]))
    return redirect(url_for('index'))

@app.route('/upload_video', methods=['POST'])
def upload_video():
    file = request.files.get('file_video')
    if file and file.filename != '':
        file.save(os.path.join(detector.FOLDER_VIDEO, secure_filename(file.filename)))
    return redirect(url_for('index'))

@app.route('/delete/<folder>/<nama_file>', methods=['POST'])
def delete_file(folder, nama_file):
    target_dir = detector.FOLDER_DATASET if folder == 'orang' else (detector.FOLDER_BENDA if folder == 'benda' else detector.FOLDER_VIDEO)
    path = os.path.join(target_dir, nama_file)
    if os.path.exists(path):
        os.remove(path)
        if folder == 'orang':
            nama_bersih = nama_file.split('_')[0] if '_' in nama_file else os.path.splitext(nama_file)[0]
            sisa_file = [f for f in os.listdir(detector.FOLDER_DATASET) if f.startswith(nama_bersih + '_') or f.startswith(nama_bersih + '.')]
            if not sisa_file:
                db = detector.muat_metadata()
                if nama_bersih in db:
                    del db[nama_bersih]
                    detector.simpan_metadata(db)
    return redirect(url_for('index'))

@app.route('/favicon.ico')
def favicon():
    return '', 204

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)