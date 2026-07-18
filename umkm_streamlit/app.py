"""
Sistem Rekomendasi Lokasi UMKM Kuliner Baru Berbasis Random Forest
=====================================================================
Versi Streamlit dengan DATA DUMMY (dibuat otomatis di dalam aplikasi).

Jalankan: streamlit run app.py

Ganti bagian generate_dummy_layers() dengan gpd.read_file(...) ke shapefile
asli Anda kalau sudah punya data sungguhan (lihat README.md).
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import folium
from streamlit_folium import st_folium
from shapely.geometry import Point, Polygon
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

st.set_page_config(page_title="Rekomendasi Lokasi UMKM - Random Forest", page_icon="🍜", layout="wide")

CRS_PROYEKSI = "EPSG:32749"  # UTM 49S (meter), cocok untuk Surabaya
KOLOM_ID = "nama_kec"
FITUR = [
    "Jumlah_Pasar_dalam_Radius_2KM",
    "Jumlah_Kampus_Perkantoran_dalam_Radius_1KM",
    "Kepadatan_UMKM_Sama",
    "Daya_Beli",
]

# --------------------------------------------------------------------------
# SIDEBAR: PARAMETER
# --------------------------------------------------------------------------
st.sidebar.header("⚙️ Parameter Data Dummy")
n_grid = st.sidebar.slider("Ukuran grid kecamatan (n x n)", 3, 12, 8)
seed = st.sidebar.number_input("Random seed", value=42, step=1)
n_pasar = st.sidebar.slider("Jumlah titik pasar (dummy)", 5, 60, 25)
n_kampus = st.sidebar.slider("Jumlah titik kampus/perkantoran (dummy)", 5, 60, 20)

st.sidebar.header("⚙️ Parameter Model")
n_estimators = st.sidebar.slider("Jumlah pohon (n_estimators)", 50, 500, 200, step=50)
test_size = st.sidebar.slider("Proporsi data uji (test size)", 0.1, 0.4, 0.2, step=0.05)

RADIUS_PASAR_M = 2000
RADIUS_KAMPUS_M = 1000

# Bounding box yang dipilih supaya grid dummy tetap berada di daratan Kota
# Surabaya (menjauhi garis pantai utara/timur ke arah Selat Madura).
AREA_MINLON, AREA_MAXLON = 112.665, 112.755
AREA_MINLAT, AREA_MAXLAT = -7.335, -7.245
CENTER_LAT = (AREA_MINLAT + AREA_MAXLAT) / 2
CENTER_LON = (AREA_MINLON + AREA_MAXLON) / 2


# --------------------------------------------------------------------------
# GENERATE DATA DUMMY
# --------------------------------------------------------------------------
@st.cache_data
def generate_dummy_layers(n_grid, seed, n_pasar, n_kampus):
    rng = np.random.default_rng(seed)
    step_x = (AREA_MAXLON - AREA_MINLON) / n_grid
    step_y = (AREA_MAXLAT - AREA_MINLAT) / n_grid

    # --- Kecamatan: grid poligon, ukuran petak menyesuaikan n_grid ---
    polys, names = [], []
    for i in range(n_grid):
        for j in range(n_grid):
            x0 = AREA_MINLON + i * step_x
            y0 = AREA_MINLAT + j * step_y
            polys.append(Polygon([(x0, y0), (x0 + step_x, y0), (x0 + step_x, y0 + step_y), (x0, y0 + step_y)]))
            names.append(f"Kecamatan {i}-{j}")
    kecamatan = gpd.GeoDataFrame({KOLOM_ID: names}, geometry=polys, crs="EPSG:4326")

    minx, miny, maxx, maxy = kecamatan.total_bounds

    # --- Titik pasar & kampus acak dalam area ---
    pasar = gpd.GeoDataFrame(
        {"nama": [f"Pasar Dummy {i}" for i in range(n_pasar)]},
        geometry=[Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy)) for _ in range(n_pasar)],
        crs="EPSG:4326",
    )
    kampus = gpd.GeoDataFrame(
        {"nama": [f"Kampus/Kantor Dummy {i}" for i in range(n_kampus)]},
        geometry=[Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy)) for _ in range(n_kampus)],
        crs="EPSG:4326",
    )

    # --- Daya beli & kepadatan UMKM: nilai acak per kecamatan (poligon sama) ---
    daya_beli = kecamatan.copy()
    daya_beli["daya_beli"] = rng.uniform(1, 10, len(daya_beli))
    daya_beli = daya_beli[["daya_beli", "geometry"]]

    kepadatan = kecamatan.copy()
    kepadatan["kepadatan"] = rng.uniform(10, 200, len(kepadatan))
    kepadatan = kepadatan[["kepadatan", "geometry"]]

    return kecamatan, pasar, kampus, daya_beli, kepadatan


kecamatan, pasar, kampus, daya_beli, kepadatan_umkm = generate_dummy_layers(n_grid, seed, n_pasar, n_kampus)


# --------------------------------------------------------------------------
# BAGIAN A: SPATIAL FEATURE ENGINEERING
# --------------------------------------------------------------------------
@st.cache_data
def hitung_fitur(_kecamatan, _pasar, _kampus, _daya_beli, _kepadatan_umkm):
    kec = _kecamatan.to_crs(CRS_PROYEKSI).copy()
    pasar_m = _pasar.to_crs(CRS_PROYEKSI)
    kampus_m = _kampus.to_crs(CRS_PROYEKSI)
    daya_beli_m = _daya_beli.to_crs(CRS_PROYEKSI)
    kepadatan_m = _kepadatan_umkm.to_crs(CRS_PROYEKSI)

    kec["centroid"] = kec.geometry.centroid
    centroid_gdf = gpd.GeoDataFrame(kec[[KOLOM_ID]], geometry=kec["centroid"], crs=CRS_PROYEKSI)

    def hitung_radius(centroid_gdf, titik_gdf, radius_m, nama_kolom):
        buf = centroid_gdf.copy()
        buf["geometry"] = buf.geometry.buffer(radius_m)
        joined = gpd.sjoin(titik_gdf, buf, how="inner", predicate="within")
        h = joined.groupby(KOLOM_ID).size()
        out = centroid_gdf[[KOLOM_ID]].copy()
        out[nama_kolom] = out[KOLOM_ID].map(h).fillna(0).astype(int)
        return out

    fitur_pasar = hitung_radius(centroid_gdf, pasar_m, RADIUS_PASAR_M, "Jumlah_Pasar_dalam_Radius_2KM")
    fitur_kampus = hitung_radius(centroid_gdf, kampus_m, RADIUS_KAMPUS_M, "Jumlah_Kampus_Perkantoran_dalam_Radius_1KM")

    def ambil_overlap(kec_gdf, poly_gdf, kval, nama):
        ov = gpd.overlay(kec_gdf[[KOLOM_ID, "geometry"]], poly_gdf[[kval, "geometry"]], how="intersection")
        ov["luas"] = ov.geometry.area
        idx = ov.groupby(KOLOM_ID)["luas"].idxmax()
        return ov.loc[idx, [KOLOM_ID, kval]].rename(columns={kval: nama})

    fitur_kepadatan = ambil_overlap(kec, kepadatan_m, "kepadatan", "Kepadatan_UMKM_Sama")
    fitur_daya_beli = ambil_overlap(kec, daya_beli_m, "daya_beli", "Daya_Beli")

    df = kec[[KOLOM_ID]].copy()
    for f in [fitur_pasar, fitur_kampus, fitur_kepadatan, fitur_daya_beli]:
        df = df.merge(f, on=KOLOM_ID, how="left")
    df = df.fillna(0)

    # Label dummy: dibuat berdasarkan kombinasi fitur + sedikit noise, supaya
    # ada pola yang bisa dipelajari model (bukan acak murni)
    rng = np.random.default_rng(seed)
    skor = (
        0.4 * (df["Jumlah_Kampus_Perkantoran_dalam_Radius_1KM"] / (df["Jumlah_Kampus_Perkantoran_dalam_Radius_1KM"].max() + 1e-9))
        + 0.3 * (df["Daya_Beli"] / df["Daya_Beli"].max())
        + 0.2 * (df["Jumlah_Pasar_dalam_Radius_2KM"] / (df["Jumlah_Pasar_dalam_Radius_2KM"].max() + 1e-9))
        + 0.1 * rng.uniform(0, 1, len(df))
    )
    df["Label_Potensi"] = (skor > skor.median()).astype(int)

    return df


df_fitur = hitung_fitur(kecamatan, pasar, kampus, daya_beli, kepadatan_umkm)


# --------------------------------------------------------------------------
# HALAMAN UTAMA
# --------------------------------------------------------------------------
st.title("🍜 Sistem Rekomendasi Lokasi UMKM Kuliner Baru Berbasis Random Forest")
st.caption(
    "Versi demo dengan **data dummy** yang dibuat otomatis (grid kecamatan + titik pasar/kampus acak). "
    "Ganti fungsi `generate_dummy_layers()` di app.py dengan data asli Anda kapan pun siap."
)

tab_data, tab_a, tab_b, tab_c = st.tabs([
    "📂 Data Dummy", "🧮 Bagian A: Feature Engineering", "🌳 Bagian B: Random Forest", "🗺️ Bagian C: Pemetaan"
])

# --- TAB: Data Dummy ---
with tab_data:
    st.subheader("Peta Data Dummy: Kecamatan, Pasar, Kampus/Perkantoran")
    m0 = folium.Map(location=[CENTER_LAT, CENTER_LON], zoom_start=13, tiles="CartoDB positron")
    folium.GeoJson(
        kecamatan,
        style_function=lambda x: {"fillColor": "#cccccc", "color": "#555", "weight": 1, "fillOpacity": 0.15},
        tooltip=folium.GeoJsonTooltip(fields=[KOLOM_ID]),
    ).add_to(m0)
    for _, row in pasar.iterrows():
        folium.CircleMarker([row.geometry.y, row.geometry.x], radius=4, color="#1565C0", fill=True, fill_opacity=0.9, tooltip=row["nama"]).add_to(m0)
    for _, row in kampus.iterrows():
        folium.CircleMarker([row.geometry.y, row.geometry.x], radius=4, color="#8E24AA", fill=True, fill_opacity=0.9, tooltip=row["nama"]).add_to(m0)
    st_folium(m0, height=500, use_container_width=True)
    st.caption("🔵 Pasar dummy • 🟣 Kampus/Perkantoran dummy • Poligon abu-abu = kecamatan dummy")

# --- TAB: Bagian A ---
with tab_a:
    st.subheader("Tabel Fitur Hasil Spatial Feature Engineering")
    st.markdown(
        "Fitur dihitung dari **centroid** tiap kecamatan, dengan buffer radius dalam CRS meter "
        "(UTM 49S / EPSG:32749) agar jaraknya akurat: `Jumlah_Pasar_dalam_Radius_2KM`, "
        "`Jumlah_Kampus_Perkantoran_dalam_Radius_1KM`, `Kepadatan_UMKM_Sama` (overlay area terbesar), "
        "dan `Daya_Beli` (overlay area terbesar)."
    )
    st.dataframe(df_fitur, use_container_width=True)
    st.download_button(
        "⬇️ Download features_kecamatan.csv",
        df_fitur.to_csv(index=False),
        file_name="features_kecamatan.csv",
        mime="text/csv",
    )

# --- TAB: Bagian B ---
with tab_b:
    st.subheader("Training Random Forest & Evaluasi")

    X = df_fitur[FITUR]
    y = df_fitur["Label_Potensi"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=int(seed), stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=int(seed),
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
    col2.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.2f}")
    col3.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.2f}")
    col4.metric("F1-Score", f"{f1_score(y_test, y_pred, zero_division=0):.2f}")

    colcm, colfi = st.columns(2)
    with colcm:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=["Aktual: Rendah", "Aktual: Tinggi"], columns=["Prediksi: Rendah", "Prediksi: Tinggi"])
        st.dataframe(cm_df, use_container_width=True)
        with st.expander("Classification Report lengkap"):
            st.text(classification_report(y_test, y_pred, target_names=["Rendah (0)", "Tinggi (1)"]))

    with colfi:
        st.markdown("**Feature Importance**")
        importance_df = pd.DataFrame({
            "fitur": FITUR, "importance": model.feature_importances_,
        }).sort_values("importance", ascending=True)
        st.bar_chart(importance_df.set_index("fitur"))

    with st.expander("📖 Jawaban Konseptual: Random Forest vs Naive Bayes"):
        st.markdown(
            "Naive Bayes mengasumsikan semua fitur saling independen. Padahal `Daya_Beli` dan "
            "`Kepadatan_UMKM_Sama` kemungkinan besar berkorelasi tinggi (kecamatan dengan daya beli "
            "tinggi biasanya juga sudah ramai UMKM). Pelanggaran asumsi ini membuat Naive Bayes bisa "
            "menghitung ganda pengaruh fitur yang tumpang tindih dan gagal menangkap interaksi antar-"
            "fitur. Random Forest tidak butuh asumsi independensi, bisa membelah data berdasarkan "
            "kombinasi fitur, dan sebagai ensemble dari banyak pohon lebih stabil untuk data kecil "
            "seperti jumlah kecamatan yang terbatas."
        )

    with st.expander("📖 Jawaban Konseptual: Strategi dari Feature Importance"):
        fitur_teratas = importance_df.iloc[-1]["fitur"]
        st.markdown(
            f"Pada model saat ini, fitur dengan importance tertinggi adalah **{fitur_teratas}**. "
            "Jika itu `Jumlah_Kampus_Perkantoran_dalam_Radius_1KM`, artinya populasi *captive* "
            "(mahasiswa/karyawan) adalah faktor paling menentukan potensi UMKM kuliner. Strategi: "
            "prioritaskan lokasi dekat kampus/perkantoran, sesuaikan jam operasional & menu dengan "
            "pola aktivitas mereka, tapi waspadai saturasi karena lokasi semacam ini biasanya sudah "
            "dilirik banyak kompetitor (cek `Kepadatan_UMKM_Sama`) — perlu diferensiasi produk."
        )

    df_fitur["Prediksi_Potensi"] = model.predict(X)
    df_fitur["Probabilitas_Tinggi"] = model.predict_proba(X)[:, 1]

# --- TAB: Bagian C ---
with tab_c:
    st.subheader("Peta Hasil Prediksi per Kecamatan")

    hasil = kecamatan.merge(
        df_fitur[[KOLOM_ID, "Prediksi_Potensi", "Probabilitas_Tinggi"]], on=KOLOM_ID, how="left"
    )
    hasil["Kategori_Potensi"] = hasil["Prediksi_Potensi"].map({1: "Potensi Tinggi", 0: "Potensi Rendah"})

    m1 = folium.Map(location=[CENTER_LAT, CENTER_LON], zoom_start=13, tiles="CartoDB positron")

    def style_fn(feature):
        kategori = feature["properties"].get("Kategori_Potensi")
        color = "#2ecc71" if kategori == "Potensi Tinggi" else "#e74c3c"
        return {"fillColor": color, "color": "#333", "weight": 1, "fillOpacity": 0.55}

    folium.GeoJson(
        hasil,
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(fields=[KOLOM_ID, "Kategori_Potensi", "Probabilitas_Tinggi"]),
        popup=folium.GeoJsonPopup(fields=[KOLOM_ID, "Kategori_Potensi", "Probabilitas_Tinggi"]),
    ).add_to(m1)

    legend_html = """
    {% macro html(this, kwargs) %}
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
        background-color: white; padding: 10px 14px; border: 2px solid #666;
        border-radius: 6px; font-size: 13px;">
    <b>Legenda</b><br>
    <span style="color:#2ecc71;">■</span> Potensi Tinggi<br>
    <span style="color:#e74c3c;">■</span> Potensi Rendah
    </div>
    {% endmacro %}
    """
    from branca.element import MacroElement, Template
    legend = MacroElement()
    legend._template = Template(legend_html)
    m1.get_root().add_child(legend)

    st_folium(m1, height=550, use_container_width=True)

    st.markdown("**Ringkasan jumlah kecamatan per kategori:**")
    st.write(hasil["Kategori_Potensi"].value_counts())

    st.download_button(
        "⬇️ Download hasil_prediksi_kecamatan.geojson",
        hasil.to_json(),
        file_name="hasil_prediksi_kecamatan.geojson",
        mime="application/geo+json",
    )
