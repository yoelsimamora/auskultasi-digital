"""
Auskultasi Digital: dashboard klasifikasi suara paru (ICBHI 2017)

Struktur repository:
    app.py
    icbhi_inference.py
    requirements.txt
    .streamlit/config.toml
    models/
        model_terbaik_4kelas.joblib
        model_terbaik_biner.joblib
        ringkasan_cv.csv            (opsional, untuk tab "Tentang model")
"""
import io
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from icbhi_inference import ModelICBHI, baca_anotasi, segmen_jendela_tetap

st.set_page_config(page_title="Auskultasi Digital", page_icon="🫁", layout="wide")

FOLDER_MODEL = Path(__file__).parent / "models"
TUGAS = {
    "Jenis bunyi (Normal, Crackle, Wheeze, Both)": "4kelas",
    "Normal atau abnormal": "biner",
}

# Palet: tinta "scrub" + warna tetap per kelas bunyi
INK = "#0E4D57"
MUTED = "#5B6B73"
GRID_MINOR = "#F6D9DE"
GRID_MAJOR = "#EBB3BD"
WARNA = {
    "Normal": "#2F7FB8",
    "Crackle": "#D9822B",
    "Wheeze": "#2E9E6B",
    "Both": "#8E5BB5",
    "Abnormal": "#C2410C",
}
URUTAN_KELAS = ["Normal", "Crackle", "Wheeze", "Both", "Abnormal"]
DESKRIPSI = {
    "Normal": "Desis napas vesikuler yang lembut, energi terutama di frekuensi rendah.",
    "Crackle": "Letupan singkat dan terputus (kurang dari 20 ms), seperti gemeresik. "
               "Di spektrogram tampak sebagai garis vertikal sesaat.",
    "Wheeze": "Bunyi bernada kontinu seperti siulan (lebih dari 100 ms). "
              "Di spektrogram tampak sebagai pita horizontal.",
    "Both": "Crackle dan wheeze muncul dalam satu siklus napas yang sama.",
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, label {{
    font-family: 'Atkinson Hyperlegible', system-ui, sans-serif !important;
}}
.block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}

.hero {{ display: grid; grid-template-columns: 1fr auto; gap: 2rem; align-items: end;
        border-bottom: 2px solid {INK}; padding-bottom: 1.1rem; margin-bottom: 1.4rem; }}
.hero .judul {{ font-size: 2.6rem; line-height: 1.05; margin: 0; font-weight: 700; color: {INK}; letter-spacing: -0.015em; }}
.hero p {{ color: {MUTED}; font-size: 1.05rem; max-width: 60ch; margin: .6rem 0 0 0; }}
.hero svg {{ width: 132px; height: auto; }}

.stMarkdown p.lead, p.lead {{ font-size: 1.3rem !important; color: {INK}; max-width: 62ch; line-height: 1.45; margin: .2rem 0 .8rem 0; }}
.komposisi {{ display: flex; height: 18px; border-radius: 9px; overflow: hidden; margin: .4rem 0 .5rem 0; }}
.komposisi span {{ display: block; height: 100%; }}
.legenda {{ display: flex; flex-wrap: wrap; gap: .3rem 1.4rem; color: {INK}; font-size: .95rem; }}
.legenda i {{ display: inline-block; width: 11px; height: 11px; border-radius: 3px;
              margin-right: .45rem; vertical-align: -1px; }}
.kelas {{ border-left: 5px solid; padding: .2rem 0 .2rem 1rem; margin-bottom: 1.1rem; }}
.kelas b {{ font-size: 1.1rem; }}
.kelas p {{ margin: .2rem 0 0 0; color: #23414A; max-width: 64ch; }}
.catatan {{ color: {MUTED}; font-size: .9rem; max-width: 70ch; }}
@media (max-width: 640px) {{
  .hero {{ grid-template-columns: 1fr; }}
  .hero svg {{ display: none; }}
  .hero .judul {{ font-size: 2rem; }}
}}
</style>
"""

# Ilustrasi garis sederhana: sepasang paru-paru dengan trakea (dibuat sendiri)
PARU_SVG = f"""
<svg viewBox="0 0 120 100" fill="none" stroke="{INK}" stroke-width="3" stroke-linecap="round"
     stroke-linejoin="round" aria-hidden="true">
  <path d="M60 6 V40 M60 40 C54 46 50 48 46 50 M60 40 C66 46 70 48 74 50"/>
  <path d="M46 26 C30 20 12 40 10 66 C9 82 18 92 32 90 C42 88 48 80 48 68 V50"/>
  <path d="M74 26 C90 20 108 40 110 66 C111 82 102 92 88 90 C78 88 72 80 72 68 V50"/>
  <path d="M22 70 h6 l3 -8 l4 14 l3 -6 h6" stroke="{WARNA['Crackle']}" stroke-width="2.4"/>
  <path d="M80 66 c4 -6 8 6 12 0 s8 6 12 0" stroke="{WARNA['Wheeze']}" stroke-width="2.4"/>
</svg>
"""


def _ambil_secret(kunci):
    try:
        return st.secrets.get(kunci)
    except Exception:
        return None


@st.cache_resource(max_entries=1, show_spinner="Memuat model...")
def muat_model(slug):
    path = FOLDER_MODEL / f"model_terbaik_{slug}.joblib"
    if not path.exists():
        url = _ambil_secret(f"MODEL_URL_{slug.upper()}")
        if not url:
            raise FileNotFoundError(
                f"File models/{path.name} belum ada. Salin file model dari Google Drive ke folder models/, "
                f"atau isi secret MODEL_URL_{slug.upper()} di pengaturan aplikasi."
            )
        import gdown
        FOLDER_MODEL.mkdir(exist_ok=True)
        gdown.download(url, str(path), quiet=True, fuzzy=True)
    return ModelICBHI.muat(path)


def simpan_sementara(berkas, akhiran):
    with tempfile.NamedTemporaryFile(suffix=akhiran, delete=False) as f:
        f.write(berkas.getvalue())
        return f.name


def strip_ritme(audio_bytes, hasil):
    """Gelombang suara di atas grid kertas EKG, dengan pita prediksi per siklus di bawahnya."""
    x, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    x = x - x.mean()
    x = x / (np.abs(x).max() + 1e-9)
    durasi = len(x) / sr
    langkah = max(1, len(x) // 24000)
    t = np.arange(len(x))[::langkah] / sr

    fig, (ax, ax_p) = plt.subplots(
        2, 1, figsize=(12, 3.6), sharex=True, gridspec_kw={"height_ratios": [4, 1], "hspace": 0.06}
    )
    fig.patch.set_alpha(0)
    ax.set_facecolor("white")
    for v in np.arange(0, durasi + 0.2, 0.2):
        ax.axvline(v, color=GRID_MAJOR if abs(v - round(v)) < 1e-6 else GRID_MINOR, lw=0.9 if abs(v - round(v)) < 1e-6 else 0.5, zorder=0)
    for h in np.arange(-1, 1.01, 0.25):
        ax.axhline(h, color=GRID_MAJOR if h in (-1, 0, 1) else GRID_MINOR, lw=0.5, zorder=0)
    for r in hasil.itertuples():
        if r.prediksi != "Normal":
            ax.axvspan(r.start, r.end, color=WARNA.get(r.prediksi, MUTED), alpha=0.10, lw=0, zorder=1)
    ax.plot(t, x[::langkah], color=INK, lw=0.55, zorder=2)
    ax.set_ylim(-1.05, 1.05)
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    ax_p.set_facecolor("none")
    for r in hasil.itertuples():
        w = r.end - r.start
        ax_p.barh(0, w, left=r.start, height=0.8, color=WARNA.get(r.prediksi, MUTED), edgecolor="white", lw=1.2)
        if w > durasi / 28:
            ax_p.text(r.start + w / 2, 0, r.prediksi, ha="center", va="center", color="white",
                      fontsize=8, fontweight="bold")
    ax_p.set_ylim(-0.5, 0.5)
    ax_p.set_yticks([])
    ax_p.set_xlim(0, durasi)
    ax_p.set_xlabel("Waktu (detik)", color=MUTED)
    ax_p.tick_params(colors=MUTED)
    for s in ["top", "right", "left"]:
        ax_p.spines[s].set_visible(False)
    ax_p.spines["bottom"].set_color(MUTED)
    return fig


def spektrogram(audio_bytes):
    import librosa
    import librosa.display
    from icbhi_inference import ke_4khz, bandpass, zscore, TARGET_SR, BAND
    x, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    x = zscore(bandpass(ke_4khz(x - x.mean(), sr)))
    S = librosa.power_to_db(librosa.feature.melspectrogram(
        y=x, sr=TARGET_SR, n_fft=256, win_length=100, hop_length=40, n_mels=40, fmin=BAND[0], fmax=BAND[1]
    ), ref=np.max)
    fig, ax = plt.subplots(figsize=(12, 2.6))
    fig.patch.set_alpha(0)
    librosa.display.specshow(S, sr=TARGET_SR, hop_length=40, x_axis="time", y_axis="mel",
                             fmin=BAND[0], fmax=BAND[1], ax=ax, cmap="mako" if "mako" in plt.colormaps() else "viridis")
    ax.set_xlabel("Waktu (detik)", color=MUTED)
    ax.set_ylabel("Hz", color=MUTED)
    ax.tick_params(colors=MUTED)
    fig.tight_layout()
    return fig


def komposisi_html(jumlah, total):
    bar = "".join(
        f'<span style="width:{100 * v / total:.2f}%;background:{WARNA.get(k, MUTED)}" title="{k}: {v}"></span>'
        for k, v in jumlah.items() if v > 0
    )
    leg = "".join(
        f'<span><i style="background:{WARNA.get(k, MUTED)}"></i>{k}: {v} siklus ({v / total:.0%})</span>'
        for k, v in jumlah.items()
    )
    return f'<div class="komposisi">{bar}</div><div class="legenda">{leg}</div>'


# ---------------------------------------------------------------------------
# Tata letak
# ---------------------------------------------------------------------------
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    f"""<div class="hero"><div>
    <div class="judul" role="heading" aria-level="1">Auskultasi digital</div>
    <p>Unggah rekaman stetoskop dan lihat bunyi tambahan yang terdeteksi pada setiap siklus napas.
    Model dilatih pada Respiratory Sound Database ICBHI 2017.</p>
    </div>{PARU_SVG}</div>""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Pengaturan")
    pilihan = st.radio("Yang ingin diketahui", list(TUGAS))
    slug = TUGAS[pilihan]
    panjang_jendela = st.slider(
        "Panjang potongan tanpa anotasi (detik)", 1.0, 5.0, 2.5, 0.5,
        help="Dipakai hanya jika file anotasi siklus (.txt) tidak diunggah.",
    )
    st.markdown(
        '<p class="catatan">Hasil ini untuk keperluan riset dan pembelajaran. '
        'Bukan pengganti pemeriksaan oleh tenaga kesehatan.</p>',
        unsafe_allow_html=True,
    )

try:
    model = muat_model(slug)
except Exception as e:
    st.error(f"Model belum bisa dimuat. {e}")
    st.stop()

tab_analisis, tab_bunyi, tab_model = st.tabs(["Analisis rekaman", "Kenali bunyi paru", "Tentang model"])

with tab_analisis:
    mode = st.segmented_control(
        "Sumber suara", ["Unggah file", "Rekam langsung"], default="Unggah file",
        selection_mode="single", key="mode_sumber",
    ) or "Unggah file"

    txt_up = None
    if mode == "Unggah file":
        kiri, kanan = st.columns(2)
        wav_up = kiri.file_uploader("Rekaman stetoskop (.wav)", type=["wav"])
        txt_up = kanan.file_uploader(
            "Anotasi siklus napas (.txt, opsional)", type=["txt"],
            help="Format ICBHI: satu siklus per baris, berisi waktu mulai dan selesai dalam detik. "
                 "Tanpa anotasi, rekaman dipotong dengan panjang tetap.",
        )
        petunjuk = ("Unggah file .wav untuk memulai. Anda bisa memakai rekaman dari folder audio_and_txt_files "
                    "di dataset ICBHI beserta file .txt dengan nama yang sama.")
    else:
        st.markdown(
            '<p class="catatan">Tekan tombol mikrofon, izinkan akses mikrofon di browser, lalu bernapas '
            'perlahan dan dalam selama 15 sampai 20 detik. Tekan lagi untuk berhenti. Hasil terbaik didapat '
            'dengan stetoskop digital yang tersambung ke komputer atau ponsel, di ruangan yang tenang.</p>',
            unsafe_allow_html=True,
        )
        wav_up = st.audio_input("Rekam suara napas", sample_rate=16000)
        petunjuk = "Belum ada rekaman. Tekan tombol mikrofon untuk mulai merekam."
        st.warning(
            "Model dilatih dengan rekaman stetoskop. Suara dari mikrofon laptop atau ponsel biasa berbeda "
            "karakternya, sehingga hasil prediksi dari rekaman ini hanya untuk demonstrasi."
        )

    if wav_up is None:
        st.info(petunjuk)
    else:
        audio = wav_up.getvalue()
        nama_file = "rekaman_mikrofon.wav" if mode == "Rekam langsung" else wav_up.name
        wav_path = simpan_sementara(wav_up, ".wav")
        txt_path = simpan_sementara(txt_up, ".txt") if txt_up is not None else None
        hasil = None
        try:
            durasi = sf.info(wav_path).duration
            if txt_path:
                segmen = [(s, min(e, durasi)) for s, e in baca_anotasi(txt_path) if s < durasi and e > s]
                sumber = "siklus napas dari file anotasi"
            else:
                segmen = segmen_jendela_tetap(durasi, panjang=panjang_jendela)
                sumber = (f"potongan tetap {panjang_jendela:g} detik. Unggah anotasi siklus "
                          f"agar hasil lebih sesuai cara model dilatih")
            if durasi < 3:
                segmen = []
                st.error(f"Rekaman terlalu pendek ({durasi:.1f} detik). Rekam minimal beberapa siklus napas, "
                         f"sekitar 15 detik.")
            elif not segmen:
                st.error("Tidak ada siklus yang berada dalam durasi rekaman. Periksa waktu pada file anotasi.")
            else:
                with st.spinner("Menganalisis rekaman..."):
                    hasil = model.prediksi_rekaman(wav_path, segmen=segmen)
        except Exception as e:
            st.error(f"File tidak bisa diproses: {e}. Pastikan file berformat .wav yang valid.")
        finally:
            for p in (wav_path, txt_path):
                if p and os.path.exists(p):
                    os.remove(p)

        if hasil is not None:
            st.audio(audio, format="audio/wav")
            n = len(hasil)
            n_abn = int((hasil.prediksi != "Normal").sum())
            if n_abn == 0:
                kalimat = f"Semua {n} siklus napas terdengar normal menurut model."
            else:
                kalimat = f"Bunyi tambahan terdeteksi pada {n_abn} dari {n} siklus napas."
            st.markdown(f'<p class="lead">{kalimat}</p>', unsafe_allow_html=True)
            urut = [k for k in URUTAN_KELAS if k in model.kelas]
            st.markdown(komposisi_html(hasil.prediksi.value_counts().reindex(urut, fill_value=0), n), unsafe_allow_html=True)
            st.caption(f"Durasi rekaman {durasi:.1f} detik. Segmen dianalisis per {sumber}.")

            st.markdown("#### Strip ritme")
            st.pyplot(strip_ritme(audio, hasil), clear_figure=True)

            with st.expander("Lihat spektrogram"):
                st.pyplot(spektrogram(audio), clear_figure=True)
                st.caption("Crackle tampak sebagai garis vertikal singkat, wheeze sebagai pita horizontal.")

            st.markdown("#### Rincian per siklus")
            tampil = hasil.copy()
            tampil.insert(0, "siklus", np.arange(1, n + 1))
            kol_p = [f"p_{k}" for k in URUTAN_KELAS if f"p_{k}" in tampil.columns]
            tampil = tampil[["siklus", "start", "end", "prediksi"] + kol_p]
            st.dataframe(
                tampil, hide_index=True, width="stretch",
                column_config={
                    "start": st.column_config.NumberColumn("mulai (dtk)", format="%.2f"),
                    "end": st.column_config.NumberColumn("selesai (dtk)", format="%.2f"),
                    **{c: st.column_config.ProgressColumn(f"peluang {c[2:]}", min_value=0, max_value=1, format="%.2f")
                       for c in kol_p},
                },
            )
            st.download_button("Unduh hasil (.csv)", tampil.to_csv(index=False).encode("utf-8"),
                               file_name=f"hasil_{Path(nama_file).stem}.csv", mime="text/csv")

with tab_bunyi:
    st.markdown("Empat kategori siklus napas yang dipakai dalam dataset ICBHI 2017:")
    for k, teks in DESKRIPSI.items():
        st.markdown(f'<div class="kelas" style="border-color:{WARNA[k]}"><b style="color:{WARNA[k]}">{k}</b>'
                    f'<p>{teks}</p></div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="catatan">Pada mode "Normal atau abnormal", crackle, wheeze, dan both digabung menjadi satu '
        'kelas abnormal.</p>', unsafe_allow_html=True)

with tab_model:
    b, m = model.bundle, model.bundle["metrik_cv"]
    st.markdown(f"Model yang dipakai: **{b['nama_model']}**, dilatih pada {b['n_siklus_latih']:,} siklus napas "
                f"dari {b['n_pasien_latih']} pasien.")
    c1, c2, c3 = st.columns(3)
    c1.metric("ICBHI Score", f"{m['ICBHI']:.3f}", help="Rata-rata specificity dan sensitivity.")
    c2.metric("Specificity", f"{m['Sp']:.3f}", help="Proporsi siklus normal yang dikenali normal.")
    c3.metric("Sensitivity", f"{m['Se']:.3f}", help="Proporsi siklus abnormal yang ditebak tepat kelasnya.")
    st.markdown(
        f'<p class="catatan">Metrik dihitung dengan validasi silang 5-fold yang membagi data per pasien, '
        f'sehingga tidak ada pasien yang sama di data latih dan uji. Sebagai patokan, model yang selalu '
        f'menebak "Normal" mendapat ICBHI Score 0,50.</p>', unsafe_allow_html=True)
    path_cv = FOLDER_MODEL / "ringkasan_cv.csv"
    if path_cv.exists():
        st.markdown("#### Perbandingan model saat validasi silang")
        cv = pd.read_csv(path_cv)
        kol = [c for c in ["task", "model", "ICBHI", "ICBHI_std", "Sp", "Se", "macro_F1"] if c in cv.columns]
        st.dataframe(cv[kol].round(3), hide_index=True, width="stretch")
