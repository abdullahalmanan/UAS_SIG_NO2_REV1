# Sistem Rekomendasi Lokasi UMKM Kuliner Baru Berbasis Random Forest (Streamlit)

Aplikasi ini memakai **data dummy yang dibuat otomatis** di dalam kode
(grid kecamatan + titik pasar/kampus acak), jadi bisa langsung dijalankan
tanpa perlu upload file apapun.

## Cara menjalankan
```bash
pip install -r requirements.txt
streamlit run app.py
```
Browser akan terbuka otomatis ke `http://localhost:8501`.

## Isi aplikasi (4 tab)
1. **📂 Data Dummy** — peta interaktif menampilkan kecamatan (grid poligon),
   titik pasar, dan titik kampus/perkantoran dummy yang di-generate.
2. **🧮 Bagian A: Feature Engineering** — tabel hasil perhitungan
   `Jumlah_Pasar_dalam_Radius_2KM`, `Jumlah_Kampus_Perkantoran_dalam_Radius_1KM`,
   `Kepadatan_UMKM_Sama`, dan `Daya_Beli` per kecamatan (bisa didownload CSV).
3. **🌳 Bagian B: Random Forest** — training model, metrik evaluasi
   (accuracy/precision/recall/F1), confusion matrix, feature importance
   chart, plus jawaban konseptual (RF vs Naive Bayes, strategi dari feature
   importance) dalam expander yang bisa dibuka.
4. **🗺️ Bagian C: Pemetaan** — peta kategorisasi hasil prediksi ("Potensi
   Tinggi" = hijau, "Potensi Rendah" = merah) dengan popup info per
   kecamatan, legenda, dan tombol download GeoJSON hasil prediksi.

## Parameter yang bisa diubah (di sidebar)
- Ukuran grid kecamatan dummy, jumlah titik pasar/kampus dummy, random seed.
- Jumlah pohon (`n_estimators`) dan proporsi data uji untuk model Random Forest.

Mengubah parameter ini otomatis meregenerasi data & melatih ulang model
(hasil di-cache dengan `@st.cache_data` agar tidak lambat).

## Mengganti dengan data ASLI
Kalau sudah siap pakai data sungguhan (5 shapefile: `Kecamatan_Surabaya.shp`,
`Pasar_Modern_Tradisional.shp`, `Kampus_Perkantoran.shp`,
`Daya_Beli_Per_Kecamatan.shp`, `Kepadatan_UMKM_Eksisting.shp`), ganti isi
fungsi `generate_dummy_layers()` di `app.py` dengan:
```python
kecamatan = gpd.read_file("data/Kecamatan_Surabaya.shp")
pasar = gpd.read_file("data/Pasar_Modern_Tradisional.shp")
kampus = gpd.read_file("data/Kampus_Perkantoran.shp")
daya_beli = gpd.read_file("data/Daya_Beli_Per_Kecamatan.shp")
kepadatan_umkm = gpd.read_file("data/Kepadatan_UMKM_Eksisting.shp")
```
dan sesuaikan `KOLOM_ID` (saat ini `"nama_kec"`) dengan nama kolom ID
kecamatan di data asli Anda. Juga hapus baris pembuatan `Label_Potensi`
dummy di `hitung_fitur()` karena label asli sudah ada di
`Kecamatan_Surabaya.shp` (kolom `Label_Potensi`, ingat: shapefile memotong
nama kolom jadi maksimal 10 karakter, cek nama aslinya dulu).

## Publikasi (opsional)
Sama seperti aplikasi sebelumnya: upload ke GitHub lalu deploy gratis lewat
https://share.streamlit.io/ dengan **Main file path**: `app.py`.
