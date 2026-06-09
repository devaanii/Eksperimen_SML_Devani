# Eksperimen SML Devani - Breast Cancer Wisconsin Dataset

Repository untuk submission **Membangun Sistem Machine Learning** - Kriteria 1 (Eksperimen & Preprocessing)

## Dataset

**Breast Cancer Wisconsin (Diagnostic)** dari UCI Machine Learning Repository
- **Sumber**: [UCI ML Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) (juga tersedia via scikit-learn)
- **Ukuran**: 569 baris × 31 kolom (30 fitur + 1 target)
- **Format**: CSV dengan delimiter koma (`,`), ada header
- **Target**: `target` (biner: 0 = malignant/ganas, 1 = benign/jinak)

### Fitur Dataset

Dataset memuat 30 fitur numerik hasil pengukuran citra digital inti sel. Untuk setiap dari 10 karakteristik berikut tersedia nilai **mean**, **standard error**, dan **worst** (rata-rata tiga nilai terbesar):

1. **radius**: jarak dari pusat ke titik pada keliling
2. **texture**: standar deviasi nilai grayscale
3. **perimeter**: keliling inti sel
4. **area**: luas inti sel
5. **smoothness**: variasi lokal panjang radius
6. **compactness**: (perimeter² / area) − 1
7. **concavity**: tingkat kecekungan kontur
8. **concave points**: jumlah bagian cekung kontur
9. **symmetry**: simetri inti sel
10. **fractal dimension**: aproksimasi "coastline" − 1

Kolom `target` adalah label diagnosis (0 = malignant, 1 = benign).

## Struktur Repository

```
Eksperimen_SML_Devani/
├── .github/
│   └── workflows/
│       └── preprocessing.yml          # GitHub Actions workflow
├── preprocessing/
│   ├── automate_Devani.py             # Script preprocessing otomatis
│   ├── Eksperimen_Devani.ipynb        # Notebook EDA & preprocessing
│   └── breast-cancer_preprocessing.csv  # Dataset hasil
├── breast-cancer_raw.csv              # Dataset mentah
├── .gitignore
└── README.md
```

## Preprocessing Pipeline

### Langkah-langkah:

1. **Data Loading**: Baca CSV dengan delimiter koma
2. **Validasi Numerik**: Pastikan semua kolom numerik
3. **Normalisasi Target**: Pastikan target bertipe integer biner (0 = malignant, 1 = benign)
4. **Handling Missing Values**: Imputasi median (jika ada)
5. **Drop Duplicates**: Hapus baris duplikat
6. **Output**: Dataset bersih siap untuk modelling

### Hasil Preprocessing

- **Baris**: 569 (tidak ada duplikat)
- **Kolom**: 31 (30 fitur + 1 target)
- **Missing values**: 0
- **Komposisi target**:
  - Benign (1)    : 357 (~62.7%)
  - Malignant (0) : 212 (~37.3%)

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
- Ada perubahan pada `breast-cancer_raw.csv`
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
scikit-learn>=1.5.0
```

## Author

**Devani** - Submission Membangun Sistem Machine Learning

---
