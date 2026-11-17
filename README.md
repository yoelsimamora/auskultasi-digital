# Auskultasi Digital

Dashboard Streamlit untuk mengklasifikasikan siklus napas dari rekaman stetoskop
(Normal, Crackle, Wheeze, Both) memakai model yang dilatih pada Respiratory Sound Database ICBHI 2017.

## Menjalankan di komputer sendiri

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Isi folder

- `app.py` — tampilan dashboard
- `icbhi_inference.py` — ekstraksi 160 fitur dan pembungkus model (harus sama dengan saat pelatihan)
- `models/` — file model `.joblib` dan `ringkasan_cv.csv`
- `.streamlit/config.toml` — tema warna

Untuk riset dan pembelajaran, bukan alat diagnosis klinis.
