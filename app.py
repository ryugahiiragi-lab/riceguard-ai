import os
import json
import uuid
import secrets
import torch
import timm
from torchvision import transforms
from PIL import Image, ImageOps, UnidentifiedImageError
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'riceguard_secret_key_skripsi')

# ==========================================
# 1. KONFIGURASI DATABASE & FOLDER UPLOAD
# ==========================================
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get(
    'DATABASE_URL', 'mysql+pymysql://root:@localhost/riceguard_db_v3'
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # Batas upload: 5MB

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

db = SQLAlchemy(app)

# ==========================================
# 2. MODEL DATABASE (ARSITEKTUR BINTANG / 4 TABEL)
# ==========================================
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False, unique=True)
    password = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='admin')
    reviews = db.relationship('PakarReview', backref='pakar', lazy=True)

class Disease(db.Model):
    __tablename__ = 'diseases'
    id = db.Column(db.Integer, primary_key=True)
    disease_name = db.Column(db.String(50), nullable=False, unique=True)
    description = db.Column(db.Text, nullable=True)
    solution = db.Column(db.Text, nullable=False)
    corrected_reviews = db.relationship('PakarReview', backref='corrected_disease', lazy=True)

class Scan(db.Model):
    __tablename__ = 'scans'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100), nullable=False)
    
    # PERUBAHAN: predicted_class (string) dihapus, diganti disease_id (Integer FK)
    disease_id = db.Column(db.Integer, db.ForeignKey('diseases.id'), nullable=False)
    
    confidence = db.Column(db.String(20), nullable=False)
    all_probabilities = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # PERUBAHAN: Relasi agar tabel Scan bisa memangil nama penyakit (scan.disease.disease_name)
    disease = db.relationship('Disease', backref='scans_history')
    review = db.relationship('PakarReview', backref='scan_data', uselist=False, cascade="all, delete-orphan")

class PakarReview(db.Model):
    __tablename__ = 'pakar_reviews'
    id = db.Column(db.Integer, primary_key=True)
    scan_id = db.Column(db.Integer, db.ForeignKey('scans.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    corrected_disease_id = db.Column(db.Integer, db.ForeignKey('diseases.id'), nullable=True)
    pakar_note = db.Column(db.Text, nullable=False)
    reviewed_at = db.Column(db.DateTime, default=datetime.utcnow)

# ==========================================
# 3. KONFIGURASI AI (EFFICIENTNET-B0)
# ==========================================
CLASSES = ['Bacterialblight', 'Blast', 'Brownspot', 'Tungro']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best_efficientnet_padi.pth')

def load_model():
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=len(CLASSES))
    # Pastikan file best_efficientnet_padi.pth ada di folder yang sama dengan app.py
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 4. MIDDLEWARE AUTENTIKASI
# ==========================================
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in') or session.get('role') != 'admin':
            flash('Akses ditolak! Halaman ini hanya untuk Pengawas Lahan.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ==========================================
# 4B. PROTEKSI CSRF (ringan, tanpa dependency tambahan)
# ==========================================
def get_csrf_token():
    if 'csrf_token' not in session:
        session['csrf_token'] = secrets.token_hex(16)
    return session['csrf_token']

app.jinja_env.globals['csrf_token'] = get_csrf_token

# Endpoint yang dikecualikan dari pengecekan CSRF (bukan form HTML biasa)
CSRF_EXEMPT_ENDPOINTS = {'check_reviews'}

@app.before_request
def csrf_protect():
    if request.method == 'POST' and request.endpoint not in CSRF_EXEMPT_ENDPOINTS:
        token_session = session.get('csrf_token')
        token_form = request.form.get('csrf_token')
        if not token_session or not token_form or not secrets.compare_digest(token_session, token_form):
            flash('Sesi Anda kadaluarsa atau tidak valid, silakan coba lagi.', 'error')
            return redirect(request.referrer or '/')

@app.errorhandler(413)
def file_too_large(e):
    flash('Ukuran file terlalu besar. Maksimal 5MB.', 'error')
    return redirect('/deteksi')

# ==========================================
# 5. ROUTING SISI PETANI (PUBLIK / TANPA LOGIN)
# ==========================================
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

@app.route('/riwayat', methods=['GET'])
def riwayat():
    return render_template('riwayat.html')

@app.route('/deteksi', methods=['GET'])
def deteksi():
    return render_template('deteksi.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('Tidak ada file yang diunggah.', 'error')
        return redirect('/deteksi')

    file = request.files['file']
    if file.filename == '':
        flash('File tidak terpilih.', 'error')
        return redirect('/deteksi')

    if not allowed_file(file.filename):
        flash('Format file tidak didukung. Gunakan JPG, JPEG, PNG, atau WEBP.', 'error')
        return redirect('/deteksi')

    if file:
        # Nama file dibuat unik (UUID) agar tidak saling menimpa antar-upload
        original_name = secure_filename(file.filename)
        extension = original_name.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}.{extension}"

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        web_filepath = filepath.replace('\\', '/')

        try:
            img = ImageOps.exif_transpose(Image.open(filepath)).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                outputs = model(img_tensor)
                prob = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted = torch.max(prob, 0)
        except (UnidentifiedImageError, OSError):
            if os.path.exists(filepath):
                os.remove(filepath)
            flash('File yang diunggah bukan gambar yang valid atau rusak.', 'error')
            return redirect('/deteksi')
        except Exception:
            if os.path.exists(filepath):
                os.remove(filepath)
            flash('Terjadi kesalahan saat memproses gambar oleh sistem AI.', 'error')
            return redirect('/deteksi')

        persentase = confidence.item() * 100
        prob_dict = {CLASSES[i]: f"{p.item() * 100:.2f}%" for i, p in enumerate(prob)}
        prob_json = json.dumps(prob_dict)

        if persentase < 60.0:
            result = {
                "class": "Tidak Dikenali",
                "confidence": f"{persentase:.2f}%",
                "image_path": web_filepath,
                "solusi": "Gambar kabur atau tidak sesuai pola daun padi.",
                "all_prob": prob_dict
            }
        else:
            class_name = CLASSES[predicted.item()]
            # PERUBAHAN: Ambil Objek Penyakit dari database untuk mendapatkan ID-nya
            disease_data = Disease.query.filter_by(disease_name=class_name).first()

            if not disease_data:
                if os.path.exists(filepath):
                    os.remove(filepath)
                flash('Sistem gagal menemukan data master penyakit di database!', 'error')
                return redirect('/deteksi')

            solusi_teks = disease_data.solution

            # PERUBAHAN: Menyimpan riwayat scan menggunakan disease_id
            new_scan = Scan(
                filename=web_filepath,
                disease_id=disease_data.id,
                confidence=f"{persentase:.2f}%",
                all_probabilities=prob_json
            )
            db.session.add(new_scan)
            db.session.commit()

            result = {
                "id": new_scan.id,
                "class": class_name,
                "confidence": f"{persentase:.2f}%",
                "image_path": web_filepath,
                "solusi": solusi_teks,
                "all_prob": prob_dict
            }

        return render_template('deteksi.html', result=result)

@app.route('/delete-scan/<int:id>', methods=['POST'])
def delete_scan(id):
    scan = Scan.query.get_or_404(id)
    try:
        if os.path.exists(scan.filename):
            os.remove(scan.filename)
    except:
        pass
    db.session.delete(scan)
    db.session.commit()
    flash('Sukses: Hasil analisis dibatalkan.', 'success')
    return redirect('/deteksi')

# ==========================================
# 6. ROUTING KHUSUS PENGAWAS (ADMIN)
# ==========================================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('logged_in'):
        return redirect('/admin/dashboard')

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username, role='admin').first()
        if user and check_password_hash(user.password, password):
            session['logged_in'] = True
            session['user_id'] = user.id
            session['username'] = user.username
            session['role'] = user.role
            return redirect('/admin/dashboard')
        else:
            flash('Kredensial Pengawas salah!', 'error')
            return render_template('login.html')
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Anda berhasil keluar dari sistem.', 'success')
    return redirect('/')

@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    disease_filter = request.args.get('disease', 'all')
    sort_order = request.args.get('sort', 'newest')
    page = request.args.get('page', 1, type=int)

    # PERUBAHAN: Karena kolom predicted_class dihapus, kita lakukan JOIN ke tabel Disease
    query = Scan.query.join(Disease)
    if disease_filter != 'all':
        query = query.filter(Disease.disease_name == disease_filter)
    
    if sort_order == 'oldest':
        query = query.order_by(Scan.created_at.asc())
    else:
        query = query.order_by(Scan.created_at.desc())

    pagination = query.paginate(page=page, per_page=10, error_out=False)
    histories = pagination.items
    for h in histories:
        try:
            h.parsed_prob = json.loads(h.all_probabilities)
        except:
            h.parsed_prob = {}

    # PERUBAHAN: Menghitung statistik menggunakan relasi JOIN
    stats = {
        'total': Scan.query.count(),
        'Bacterialblight': Scan.query.join(Disease).filter(Disease.disease_name == 'Bacterialblight').count(),
        'Blast': Scan.query.join(Disease).filter(Disease.disease_name == 'Blast').count(),
        'Brownspot': Scan.query.join(Disease).filter(Disease.disease_name == 'Brownspot').count(),
        'Tungro': Scan.query.join(Disease).filter(Disease.disease_name == 'Tungro').count()
    }
    
    master_diseases = Disease.query.all()

    return render_template('dashboard.html', histories=histories, stats=stats, current_filter=disease_filter, current_sort=sort_order, diseases=master_diseases, pagination=pagination)

@app.route('/admin/add-note/<int:scan_id>', methods=['POST'])
@login_required
def add_note(scan_id):
    scan = Scan.query.get_or_404(scan_id)
    pakar_note = request.form.get('pakar_note')
    koreksi_penyakit = request.form.get('pakar_correction')
    
    corrected_id = None
    if koreksi_penyakit:
        disease_record = Disease.query.filter_by(disease_name=koreksi_penyakit).first()
        if disease_record:
            corrected_id = disease_record.id

    existing_review = PakarReview.query.filter_by(scan_id=scan.id).first()
    if existing_review:
        existing_review.pakar_note = pakar_note
        existing_review.corrected_disease_id = corrected_id
        existing_review.reviewed_at = datetime.utcnow()
    else:
        new_review = PakarReview(
            scan_id=scan.id,
            user_id=session.get('user_id'),
            corrected_disease_id=corrected_id,
            pakar_note=pakar_note
        )
        db.session.add(new_review)
        
    db.session.commit()
    flash('Review dan catatan pengawas berhasil disimpan ke dalam Laporan!', 'success')
    return redirect('/admin/dashboard')

@app.route('/admin/edit-disease/<int:id>', methods=['POST'])
@login_required
def edit_disease(id):
    disease = Disease.query.get_or_404(id)
    disease.solution = request.form.get('solution')
    db.session.commit()
    flash(f'Solusi penanganan untuk {disease.disease_name} berhasil diperbarui!', 'success')
    return redirect('/admin/dashboard')

@app.route('/admin/delete/<int:id>', methods=['POST'])
@login_required
def admin_delete(id):
    scan = Scan.query.get_or_404(id)
    try:
        if os.path.exists(scan.filename):
            os.remove(scan.filename)
    except:
        pass
    db.session.delete(scan)
    db.session.commit()
    flash('Data riwayat scan berhasil dihapus.', 'success')
    return redirect('/admin/dashboard')

@app.route('/api/check-reviews', methods=['POST'])
def check_reviews():
    data = request.get_json()
    if not data or 'scan_ids' not in data:
        return jsonify({})
        
    scan_ids = data['scan_ids']
    reviews = PakarReview.query.filter(PakarReview.scan_id.in_(scan_ids)).all()
    
    review_map = {}
    for r in reviews:
        review_map[r.scan_id] = {
            'pakar_note': r.pakar_note,
            'corrected_disease': r.corrected_disease.disease_name if r.corrected_disease else None
        }
        
    return jsonify(review_map)

# ==========================================
# 7. INISIALISASI APLIKASI & SEEDER
# ==========================================
# PENTING: blok ini sengaja diletakkan di level modul (bukan di dalam
# `if __name__ == '__main__':`) supaya tetap jalan walau aplikasi dijalankan
# lewat Gunicorn/WSGI server saat deploy (mis. di Render), bukan cuma lewat
# `python app.py` secara lokal.
def init_db():
    with app.app_context():
        try:
            # Karena ini baru pertama kali dengan struktur baru, pastikan db kosong/drop dulu di phpMyAdmin
            db.create_all()

            # 1. Seeder Admin
            if not User.query.filter_by(username='admin').first():
                default_admin = User(username='admin', password=generate_password_hash('password123'), role='admin')
                db.session.add(default_admin)

            # 2. Seeder Master Penyakit
            if not Disease.query.first():
                default_diseases = [
                    Disease(disease_name='Bacterialblight', solution='Gunakan bakterisida berbahan aktif tembaga hidroksida.'),
                    Disease(disease_name='Blast', solution='Semprotkan fungisida trisiklazol dan kurangi pupuk urea berlebih.'),
                    Disease(disease_name='Brownspot', solution='Berikan pupuk Kalium yang cukup dan gunakan fungisida propikonazol.'),
                    Disease(disease_name='Tungro', solution='Segera cabut tanaman terinfeksi dan kendalikan wereng hijau dengan insektisida.')
                ]
                db.session.bulk_save_objects(default_diseases)

            db.session.commit()
        except Exception as e:
            # Jangan sampai proses booting worker Gunicorn mati total hanya karena
            # DB sempat belum siap/telat konek saat cold start di hosting gratis.
            db.session.rollback()
            print(f"[init_db] Peringatan: gagal inisialisasi database saat startup: {e}")

init_db()

if __name__ == '__main__':
    # Mode debug HANYA aktif jika FLASK_DEBUG=True di environment (jangan aktif saat production/demo publik)
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=debug_mode)