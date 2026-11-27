"""
Dashboard Potensi Permintaan Sparepart Apple Device - PT X
Project Data Mining Course 2026 - Universitas Maritim Raja Ali Haji

Dashboard ini hanya MEMBACA file hasil yang sudah dibuat notebook
(folder hasil_akhir/). Jalankan notebook (Restart & Run All) terlebih dahulu.

Cara menjalankan (dari folder project, di terminal/CMD):
    pip install streamlit
    streamlit run app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Dashboard Potensi Permintaan Sparepart - PT X",
    layout="wide",
)

HASIL = Path(__file__).resolve().parent / "hasil_akhir"
THRESHOLD_FINAL = 0.20

NAMA_BULAN = {
    "Oct": "Okt", "May": "Mei", "Aug": "Agu", "Dec": "Des",
    "Sep": "Sep", "Jan": "Jan", "Feb": "Feb", "Mar": "Mar",
    "Apr": "Apr", "Jun": "Jun", "Jul": "Jul", "Nov": "Nov",
}


@st.cache_data
def muat(nama_file):
    return pd.read_csv(HASIL / nama_file, encoding="utf-8-sig")


# ------------------------------------------------------------------
# Membaca file hasil
# ------------------------------------------------------------------
try:
    top10 = muat("09_Top10_Sparepart.csv")
    tren = muat("10_Tren_Bulanan.csv")
    rekap = muat("11_Rekap_Bulanan.csv")
    evaluasi = muat("13_Evaluasi_Final.csv")
    cm = muat("14_Confusion_Matrix.csv")
    prediksi = muat("16_Prediksi_September_2026.csv")
    uji_threshold = muat("17_Hasil_Uji_Threshold.csv")
except FileNotFoundError as galat:
    st.error(
        "File hasil belum ditemukan. Jalankan notebook (Restart & Run All) "
        "sampai selesai agar folder 'hasil_akhir' terbentuk, lalu muat ulang "
        f"halaman ini. Detail: {galat}"
    )
    st.stop()

# Rapikan label bulan pada tabel tren (Oct -> Okt, dst.)
tren = tren.rename(
    columns={
        k: " ".join([NAMA_BULAN.get(k.split()[0], k.split()[0])] + k.split()[1:])
        for k in tren.columns
        if k != "Nama_Produk_Normalisasi"
    }
)

# ------------------------------------------------------------------
# Judul
# ------------------------------------------------------------------
st.title("Dashboard Potensi Permintaan Sparepart Apple Device")
st.caption(
    "PT X | Periode data: September 2025 - Agustus 2026 | "
    "Metode: Logistic Regression (klasifikasi potensi permintaan bulan berikutnya)"
)

tab_ringkasan, tab_tren, tab_prediksi, tab_evaluasi = st.tabs(
    ["Ringkasan", "Top 10 dan Tren", "Prediksi September 2026", "Evaluasi Model"]
)

# ------------------------------------------------------------------
# Tab 1 - Ringkasan
# ------------------------------------------------------------------
with tab_ringkasan:
    total_kelompok = int(rekap["Total_Permintaan"].sum())
    baris_puncak = rekap.loc[rekap["Total_Permintaan"].idxmax()]
    produk_teratas = top10.iloc[0]

    k1, k2, k3 = st.columns(3)
    k1.metric("Total kelompok permintaan", f"{total_kelompok:,}".replace(",", "."))
    k2.metric(
        "Bulan permintaan tertinggi",
        str(baris_puncak["Bulan_Analisis"]),
        f"{int(baris_puncak['Total_Permintaan'])} kelompok",
        delta_color="off",
    )
    k3.metric(
        "Sparepart paling sering diminta",
        str(produk_teratas["Nama_Produk_Normalisasi"]),
        f"{int(produk_teratas['Jumlah_Permintaan'])} permintaan",
        delta_color="off",
    )

    st.subheader("Jumlah kelompok permintaan per bulan")
    st.line_chart(rekap.set_index("Bulan_Analisis")["Total_Permintaan"])
    st.caption(
        "Satuan: kelompok permintaan sparepart (bukan jumlah transaksi mentah)."
    )

# ------------------------------------------------------------------
# Tab 2 - Top 10 dan Tren
# ------------------------------------------------------------------
with tab_tren:
    st.subheader("Top 10 sparepart berdasarkan jumlah permintaan")
    st.bar_chart(top10.set_index("Nama_Produk_Normalisasi")["Jumlah_Permintaan"])
    st.dataframe(
        top10.rename(
            columns={
                "Nama_Produk_Normalisasi": "Sparepart",
                "Jumlah_Permintaan": "Jumlah Permintaan",
                "Jumlah_ID_Transaksi": "Jumlah ID Transaksi",
            }
        ),
        hide_index=True,
    )

    st.subheader("Tren permintaan bulanan")
    daftar_produk = tren["Nama_Produk_Normalisasi"].tolist()
    pilihan = st.multiselect(
        "Pilih sparepart",
        daftar_produk,
        default=daftar_produk[:3],
    )
    if pilihan:
        data_tren = (
            tren.set_index("Nama_Produk_Normalisasi").loc[pilihan].T
        )
        st.line_chart(data_tren)
    else:
        st.info("Pilih minimal satu sparepart untuk menampilkan tren.")
    st.caption(
        "Nilai 0 berarti tidak ada permintaan tercatat pada bulan tersebut, "
        "bukan berarti produk tidak akan dibutuhkan di masa depan."
    )

# ------------------------------------------------------------------
# Tab 3 - Prediksi September 2026
# ------------------------------------------------------------------
with tab_prediksi:
    st.subheader("Klasifikasi potensi permintaan September 2026")

    threshold = st.slider(
        "Threshold klasifikasi",
        min_value=0.05,
        max_value=0.95,
        value=THRESHOLD_FINAL,
        step=0.05,
        help="Threshold final penelitian adalah 0,20 (dipilih pada data validasi).",
    )
    if abs(threshold - THRESHOLD_FINAL) > 1e-9:
        st.warning(
            f"Threshold diubah menjadi {threshold:.2f}. "
            f"Hasil penelitian memakai threshold final {THRESHOLD_FINAL:.2f}."
        )

    tabel = prediksi.copy()
    tabel["Hasil"] = np.where(
        tabel["Probabilitas_Model"] >= threshold,
        "Ada Potensi",
        "Tidak Ada Potensi",
    )
    tabel["Probabilitas (%)"] = (tabel["Probabilitas_Model"] * 100).round(2)
    tabel = tabel.sort_values("Probabilitas_Model", ascending=False)

    jumlah_ada = int((tabel["Hasil"] == "Ada Potensi").sum())
    c1, c2 = st.columns(2)
    c1.metric("Sparepart: Ada Potensi", jumlah_ada)
    c2.metric("Sparepart: Tidak Ada Potensi", len(tabel) - jumlah_ada)

    st.dataframe(
        tabel[
            ["Nama_Sparepart", "Permintaan_Agustus", "Probabilitas (%)", "Hasil"]
        ].rename(
            columns={
                "Nama_Sparepart": "Sparepart",
                "Permintaan_Agustus": "Permintaan Agustus",
            }
        ),
        hide_index=True,
        column_config={
            "Probabilitas (%)": st.column_config.ProgressColumn(
                "Probabilitas (%)",
                min_value=0,
                max_value=100,
                format="%.2f%%",
            )
        },
    )

    st.bar_chart(tabel.set_index("Nama_Sparepart")["Probabilitas (%)"])

    st.info(
        "Kategori 'Ada Potensi' tidak sama dengan rekomendasi jumlah pembelian. "
        "Hasil ini simulasi berdasarkan riwayat sampai Agustus 2026 dan perlu "
        "dipadukan dengan data stok aktual, pemasok, dan biaya sebelum dipakai "
        "untuk keputusan pengadaan."
    )

# ------------------------------------------------------------------
# Tab 4 - Evaluasi model
# ------------------------------------------------------------------
with tab_evaluasi:
    st.subheader("Hasil evaluasi pada data testing (threshold 0,20)")

    nilai = dict(zip(evaluasi["Metrik"], evaluasi["Nilai"]))
    kolom = st.columns(len(nilai))
    for tempat, (nama, v) in zip(kolom, nilai.items()):
        tempat.metric(nama, f"{v * 100:.2f}%")

    st.subheader("Confusion matrix")
    st.dataframe(
        cm.rename(
            columns={
                "Aktual": "Aktual",
                "Prediksi 0": "Prediksi: Tidak Ada Potensi",
                "Prediksi 1": "Prediksi: Ada Potensi",
            }
        ).replace({"Aktual 0": "Tidak Ada Potensi", "Aktual 1": "Ada Potensi"}),
        hide_index=True,
    )

    st.subheader("Pengujian threshold (data validasi)")
    tabel_uji = uji_threshold.copy()
    for k in ["Accuracy", "Precision", "Recall", "F1_Score"]:
        tabel_uji[k] = (tabel_uji[k] * 100).round(2)
    st.dataframe(
        tabel_uji.rename(columns={"F1_Score": "F1-Score"}),
        hide_index=True,
    )
    st.caption(
        "Threshold dipilih pada data validasi (Maret-April 2026), sedangkan "
        "data testing (Mei-Agustus 2026) hanya dipakai sebagai pengujian akhir. "
        "Nilai pada tabel ini dalam persen."
    )
