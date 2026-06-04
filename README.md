# Eksperimen SML Devani - Wine Quality Dataset

Repository untuk submission **Membangun Sistem Machine Learning** - Kriteria 1 (Eksperimen & Preprocessing)

## Dataset

**Wine Quality (White Wine)** dari UCI Machine Learning Repository
- **Sumber**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- **Ukuran**: 4,898 baris × 12 kolom (11 fitur + 1 target)
- **Format**: CSV dengan delimiter semicolon (`;`), ada header
- **Target**: `quality` (skor 0-10, dibinarisasi menjadi 0/1)

### Fitur Dataset

1. **fixed acidity**: Asam tetap
2. **volatile acidity**: Asam volatil
3. **citric acid**: Asam sitrat
4. **residual sugar**: Gula sisa
5. **chlorides**: Klorida
6. **free sulfur dioxide**: Sulfur dioksida bebas
7. **total sulfur dioxide**: Total sulfur dioksida
8. **density**: Densitas
9. **pH**: Tingkat keasaman
10. **sulphates**: Sulfat
11. **alcohol**: Kadar alkohol
12. **quality**: Kualitas wine (target)

## Struktur Repository

```
Eksperimen_SML_Devani/
├── .github/
│   └── workflows/
│       └── preprocessing.yml          # GitHub Actions workflow
├── preprocessing/
│   ├── automate_Devani.py             # Script preprocessing otomatis
│   ├── Eksperimen_Devani.ipynb        # Notebook EDA & preprocessing
│   └── wine-quality-white_preprocessing.csv  # Dataset hasil
├── wine-quality-white_raw.csv         # Dataset mentah
├── .gitignore
└── README.md
```

## Preprocessing Pipeline

### Langkah-langkah:

1. **Data Loading**: Baca CSV dengan delimiter semicolon
2. **Validasi Numerik**: Pastikan semua kolom numerik
3. **Binarisasi Target**: 
   - `quality >= 6` → 1 (good wine)
   - `quality < 6` → 0 (bad wine)
4. **Handling Missing Values**: Imputasi median (jika ada)
5. **Drop Duplicates**: Hapus baris duplikat
6. **Output**: Dataset bersih siap untuk modelling

### Hasil Preprocessing

- **Baris awal**: 4,898
- **Baris setelah preprocessing**: ~3,961 (setelah drop duplikat)
- **Missing values**: 0
- **Target balance**: 
  - Good (1): ~2,613 (66%)
  - Bad (0): ~1,348 (34%)

## Cara Penggunaan

### 1. Manual - Menjalankan Script

```bash
cd preprocessing
python automate_Devani.py
```

### 2. Notebook - EDA & Preprocessing

Buka `preprocessing/Eksperimen_Devani.ipynb` untuk:
- Exploratory Data Analysis (EDA)
- Visualisasi distribusi fitur
- Analisis korelasi
- Preprocessing step-by-step

### 3. Automated - GitHub Actions

Workflow CI akan otomatis berjalan ketika:
- Ada perubahan pada `wine-quality-white_raw.csv`
- Ada perubahan pada `preprocessing/automate_Devani.py`
- Ada perubahan pada workflow file
- Trigger manual via workflow_dispatch

Workflow akan:
1. Install dependencies (pandas, numpy)
2. Jalankan script preprocessing
3. Upload hasil sebagai artifact
4. Commit hasil ke repository (jika ada perubahan)

## Dependencies

```
pandas==2.2.3
numpy==2.2.6
matplotlib>=3.7.0
seaborn>=0.12.0
```

## Target Kriteria

✅ **Advance (4 poin)**
- Script preprocessing self-contained dan mandiri
- GitHub Actions workflow berjalan sukses
- Dataset hasil preprocessing ter-track di git
- Artifact preprocessing ter-upload
- Dokumentasi lengkap

## Author

**Devani** - Submission Membangun Sistem Machine Learning

---

## Next Steps (Kriteria 2-4)

1. **Kriteria 2**: Modelling dengan MLflow (lokal + DagsHub)
2. **Kriteria 3**: MLflow Project + CI untuk re-training
3. **Kriteria 4**: Serving + Monitoring dengan Prometheus & Grafana
