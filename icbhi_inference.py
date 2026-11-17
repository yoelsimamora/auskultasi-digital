"""
icbhi_inference.py — Modul inferensi untuk model klasifikasi suara pernapasan ICBHI 2017.

Isi modul ini adalah salinan PERSIS pipeline 'utama' di notebook
(mono + DC removal -> resample 4 kHz -> bandpass 50-1800 Hz -> z-score -> 160 fitur
dengan CMN tingkat rekaman), ditambah kelas ICBHICalibratedClassifier, agar model
yang disimpan (.joblib) dapat dimuat ulang di luar notebook (mis. Streamlit/Gradio).

Contoh pemakaian:
    from icbhi_inference import ModelICBHI
    model = ModelICBHI.muat("model_terbaik_4kelas.joblib")
    hasil = model.prediksi_rekaman("rekaman.wav", anotasi_txt="rekaman.txt")
"""
from math import gcd
import warnings

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from scipy import signal, stats
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

SEED = 42

# ---------------------------------------------------------------------------
# Parameter DSP (harus identik dengan notebook pelatihan)
# ---------------------------------------------------------------------------
TARGET_SR = 4000
BAND = (50, 1800)
SOS = signal.butter(4, BAND, btype="bandpass", fs=TARGET_SR, output="sos")
N_FFT, WIN, HOP, N_MELS, N_MFCC = 256, 100, 40, 26, 13
SUBBAND = [(50, 250), (250, 600), (600, 1200), (1200, 1800)]
SPEK = ["centroid", "bandwidth", "rolloff", "flatness"]
ENER = ["zcr", "rms"]

NAMA_FITUR = (
    [f"mfcc{k}_{s}" for s in ["mean", "std", "p10", "p90"] for k in range(N_MFCC)]
    + [f"d_mfcc{k}_std" for k in range(N_MFCC)] + [f"dd_mfcc{k}_std" for k in range(N_MFCC)]
    + [f"{n}_{s}" for s in ["mean", "std", "max"] for n in SPEK + ENER]
    + [f"contrast_b{j}_{s}" for s in ["mean", "std"] for j in range(5)]
    + ["flatness_p10", "rms_p95_p50", "crest_factor", "kurtosis", "skewness"]
    + [f"eband_{lo}_{hi}" for lo, hi in SUBBAND] + ["cycle_duration"]
    + [f"h1_mfcc{k}" for k in range(N_MFCC)] + ["h1_centroid", "h1_flatness", "h1_rms"]
    + [f"h2_mfcc{k}" for k in range(N_MFCC)] + ["h2_centroid", "h2_flatness", "h2_rms"]
    + [f"diff_mfcc{k}" for k in range(9)] + ["diff_centroid", "diff_flatness", "diff_rms"]
)


# ---------------------------------------------------------------------------
# Preprocessing sinyal
# ---------------------------------------------------------------------------
def muat_audio(wav_path):
    x, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    return (x - x.mean()).astype(np.float32), int(sr)


def ke_4khz(x, sr):
    if sr == TARGET_SR:
        return x
    g = gcd(int(sr), TARGET_SR)
    return signal.resample_poly(x, TARGET_SR // g, int(sr) // g).astype(np.float32)


def bandpass(x):
    return signal.sosfiltfilt(SOS, x).astype(np.float32)


def zscore(x):
    return ((x - x.mean()) / (x.std() + 1e-8)).astype(np.float32)


# ---------------------------------------------------------------------------
# Metrik & kalibrasi ICBHI (dibutuhkan agar model hasil pickle bisa dimuat)
# ---------------------------------------------------------------------------
def skor_icbhi(y_true, y_pred, normal="Normal"):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    m = y_true == normal
    sp = float(np.mean(y_pred[m] == normal)) if m.any() else np.nan
    se = float(np.mean(y_pred[~m] == y_true[~m])) if (~m).any() else np.nan
    return sp, se, 0.5 * (sp + se)


def _optimalkan_pengali_icbhi(proba_oof, y_true, kelas):
    kelas = np.asarray(kelas)
    n_c = len(kelas)
    w = np.ones(n_c, dtype=float)
    best_sc = skor_icbhi(y_true, kelas[np.argmax(proba_oof, axis=1)])[2]
    grid = np.linspace(0.55, 1.85, 27)
    for _ in range(3):
        for c in range(n_c):
            best_val = w[c]
            for cand in grid:
                w_try = w.copy()
                w_try[c] = cand
                pred = kelas[np.argmax(proba_oof * w_try, axis=1)]
                sc = skor_icbhi(y_true, pred)[2]
                if sc > best_sc + 1e-6:
                    best_sc = sc
                    best_val = cand
            w[c] = best_val
    return w


class ICBHICalibratedClassifier(BaseEstimator, ClassifierMixin):
    """Classifier probabilistik + pengali kelas hasil inner-CV untuk memaksimalkan ICBHI Score."""

    def __init__(self, base_estimator, inner_splits=3, random_state=SEED):
        self.base_estimator = base_estimator
        self.inner_splits = inner_splits
        self.random_state = random_state

    def fit(self, X, y, groups=None):
        y = np.asarray(y)
        self.model_ = clone(self.base_estimator)
        self.model_.fit(X, y)
        self.classes_ = np.asarray(self.model_.classes_)
        n_c = len(self.classes_)
        p_inner = np.zeros((len(y), n_c), dtype=float)
        if groups is not None and len(np.unique(groups)) >= self.inner_splits:
            splitter = StratifiedGroupKFold(n_splits=self.inner_splits, shuffle=True, random_state=self.random_state)
            splits = splitter.split(X, y, groups)
        else:
            splitter = StratifiedKFold(n_splits=self.inner_splits, shuffle=True, random_state=self.random_state)
            splits = splitter.split(X, y)
        for itr, iva in splits:
            m_in = clone(self.base_estimator)
            m_in.fit(X[itr], y[itr])
            p_va = m_in.predict_proba(X[iva])
            for j, c in enumerate(m_in.classes_):
                col_idx = list(self.classes_).index(c)
                p_inner[iva, col_idx] = p_va[:, j]
        self.class_multipliers_ = _optimalkan_pengali_icbhi(p_inner, y, self.classes_)
        return self

    def predict_proba(self, X):
        p = self.model_.predict_proba(X) * self.class_multipliers_
        return p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ---------------------------------------------------------------------------
# Ekstraksi 160 fitur (konfigurasi 'utama')
# ---------------------------------------------------------------------------
def _fitur_frame(x, sr):
    win, hop, n_fft, fmin, fmax = WIN, HOP, N_FFT, BAND[0], BAND[1]
    Smag = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop, win_length=win, center=True))
    mel = librosa.feature.melspectrogram(S=Smag ** 2, sr=sr, n_mels=N_MELS, fmin=fmin, fmax=fmax)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel, ref=1.0, amin=1e-10), n_mfcc=N_MFCC)
    spek = np.vstack([
        librosa.feature.spectral_centroid(S=Smag, sr=sr),
        librosa.feature.spectral_bandwidth(S=Smag, sr=sr),
        librosa.feature.spectral_rolloff(S=Smag, sr=sr, roll_percent=0.85),
        librosa.feature.spectral_flatness(S=Smag),
    ])
    ener = np.vstack([
        librosa.feature.zero_crossing_rate(x, frame_length=win, hop_length=hop),
        librosa.feature.rms(S=Smag, frame_length=n_fft),
    ])
    contrast = librosa.feature.spectral_contrast(S=Smag, sr=sr, n_fft=n_fft, fmin=50, n_bands=4)
    T = min(mfcc.shape[1], spek.shape[1], ener.shape[1], contrast.shape[1])
    mfcc = mfcc[:, :T]
    return dict(
        hop=hop, T=T, mfcc=mfcc, mfcc_cmn=mfcc - mfcc.mean(axis=1, keepdims=True),
        d=librosa.feature.delta(mfcc), dd=librosa.feature.delta(mfcc, order=2),
        spek_ener=np.vstack([spek[:, :T], ener[:, :T]]), contrast=contrast[:, :T],
    )


def _ringkas(x, sr, F, s, e):
    a = min(max(int(np.floor(s * sr / F["hop"])), 0), F["T"] - 1)
    b = min(max(int(np.ceil(e * sr / F["hop"])) + 1, a + 2), F["T"])
    m = F["mfcc_cmn"][:, a:b]
    se, ct = F["spek_ener"][:, a:b], F["contrast"][:, a:b]

    mid = a + max(1, (b - a) // 2)
    m_h1 = m[:, :max(1, mid - a)]
    m_h2 = m[:, max(0, mid - a):] if mid < b else m_h1
    se_h1 = se[:, :max(1, mid - a)]
    se_h2 = se[:, max(0, mid - a):] if mid < b else se_h1

    h1_vec = np.concatenate([m_h1.mean(1), se_h1[[0, 3, 5]].mean(1)])
    h2_vec = np.concatenate([m_h2.mean(1), se_h2[[0, 3, 5]].mean(1)])
    diff_vec = np.concatenate([m_h2[:9].mean(1) - m_h1[:9].mean(1),
                               se_h2[[0, 3, 5]].mean(1) - se_h1[[0, 3, 5]].mean(1)])

    flat_p10 = float(np.percentile(se[3], 10))
    rms_p95_p50 = float(np.percentile(se[5], 95) / (np.median(se[5]) + 1e-8))

    seg = x[int(s * sr): int(e * sr)]
    if len(seg) < 16:
        seg = np.pad(seg, (0, 16 - len(seg)))
    rms = float(np.sqrt(np.mean(seg ** 2)) + 1e-12)
    f, p = signal.welch(seg, fs=sr, nperseg=min(len(seg), int(2 ** np.ceil(np.log2(sr / 16)))))
    tot = p.sum() + 1e-20
    pita = [float(p[(f >= lo) & (f < hi)].sum() / tot) for lo, hi in SUBBAND]
    wkt = np.array([
        flat_p10, rms_p95_p50,
        float(np.abs(seg).max() / rms), float(stats.kurtosis(seg)),
        float(stats.skew(seg)), *pita, float(e - s),
    ], dtype=np.float32)
    return np.concatenate([
        m.mean(1), m.std(1), np.percentile(m, 10, axis=1), np.percentile(m, 90, axis=1),
        F["d"][:, a:b].std(1), F["dd"][:, a:b].std(1),
        se.mean(1), se.std(1), se.max(1), ct.mean(1), ct.std(1), wkt,
        h1_vec, h2_vec, diff_vec,
    ]).astype(np.float32)


def baca_anotasi(txt_path):
    """Membaca file anotasi ICBHI (.txt: start end crackles wheezes) -> list (start, end)."""
    ann = pd.read_csv(txt_path, sep=r"\s+", header=None, names=["start", "end", "crackles", "wheezes"])
    return list(zip(ann["start"].astype(float), ann["end"].astype(float)))


def segmen_jendela_tetap(durasi, panjang=2.5, langkah=None, minimal=0.5):
    """Segmentasi cadangan bila tidak ada anotasi siklus napas (jendela tetap).

    Catatan: model dilatih pada siklus napas hasil anotasi pakar, sehingga hasil
    pada jendela tetap hanyalah pendekatan dan bisa kurang akurat.
    """
    langkah = langkah or panjang
    seg, s = [], 0.0
    while s + minimal <= durasi:
        seg.append((s, min(s + panjang, durasi)))
        s += langkah
    return seg or [(0.0, durasi)]


def ekstrak_fitur_rekaman(wav_path, segmen):
    """Mengembalikan matriks fitur (n_segmen, 160) untuk satu rekaman utuh.

    Rekaman HARUS dimuat utuh karena CMN dihitung pada seluruh rekaman.
    segmen: list (start_detik, end_detik).
    """
    x_raw, sr0 = muat_audio(wav_path)
    x_u = zscore(bandpass(ke_4khz(x_raw, sr0)))
    F_u = _fitur_frame(x_u, TARGET_SR)
    X = np.vstack([_ringkas(x_u, TARGET_SR, F_u, float(s), float(e)) for s, e in segmen])
    return X, len(x_raw) / sr0


# ---------------------------------------------------------------------------
# Pembungkus model untuk dashboard
# ---------------------------------------------------------------------------
class ModelICBHI:
    def __init__(self, bundle):
        self.bundle = bundle
        self.model = bundle["model"]
        self.kelas = list(bundle["kelas"])
        self.median_fitur = np.asarray(bundle["median_fitur"], dtype=np.float32)

    @classmethod
    def muat(cls, path):
        import joblib, sklearn, lightgbm
        bundle = joblib.load(path)
        ver = bundle.get("versi_pustaka", {})
        for nama, modul in [("scikit-learn", sklearn), ("lightgbm", lightgbm)]:
            if ver.get(nama) and ver[nama] != modul.__version__:
                warnings.warn(f"Versi {nama} berbeda: dilatih {ver[nama]}, sekarang {modul.__version__}. "
                              f"Sebaiknya samakan versi (lihat requirements.txt).")
        return cls(bundle)

    def _imputasi(self, X):
        bad = ~np.isfinite(X)
        if bad.any():
            X = X.copy()
            X[bad] = np.take(self.median_fitur, np.where(bad)[1])
        return X

    def prediksi_fitur(self, X):
        X = self._imputasi(np.asarray(X, dtype=np.float32))
        proba = self.model.predict_proba(X)
        kelas_model = list(self.model.classes_)
        return np.asarray(kelas_model)[np.argmax(proba, axis=1)], proba, kelas_model

    def prediksi_rekaman(self, wav_path, segmen=None, anotasi_txt=None, panjang_jendela=2.5):
        """Prediksi per siklus/segmen untuk satu file .wav.

        Prioritas segmen: argumen `segmen` -> file `anotasi_txt` -> jendela tetap.
        Mengembalikan DataFrame: start, end, prediksi, dan probabilitas tiap kelas.
        """
        if segmen is None and anotasi_txt is not None:
            segmen = baca_anotasi(anotasi_txt)
        if segmen is None:
            durasi = sf.info(str(wav_path)).duration
            segmen = segmen_jendela_tetap(durasi, panjang=panjang_jendela)
        X, _ = ekstrak_fitur_rekaman(wav_path, segmen)
        pred, proba, kelas_model = self.prediksi_fitur(X)
        df = pd.DataFrame({"start": [s for s, _ in segmen], "end": [e for _, e in segmen], "prediksi": pred})
        for j, k in enumerate(kelas_model):
            df[f"p_{k}"] = proba[:, j]
        return df
