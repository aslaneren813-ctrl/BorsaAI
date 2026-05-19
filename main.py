from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

app = FastAPI(title="BorsaAI Yapay Zeka Portalı", version="0.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Arayüzü doğrudan kodun içine alıyoruz ki klasör bulamama hatası tamamen ortadan kalksın
HTML_SAYFASI = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BorsaAI - Yapay Zeka Analiz Paneli</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ background-color: #f4f6f9; font-family: 'Segoe UI', sans-serif; }}
        .card {{ border: none; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }}
        .bg-yukari {{ background-color: #d1e7dd !important; color: #0f5132 !important; }}
        .bg-asagi {{ background-color: #f8d7da !important; color: #842029 !important; }}
    </style>
</head>
<body>
    <nav class="navbar navbar-dark bg-dark mb-4">
        <div class="container">
            <a class="navbar-brand fw-bold" href="#">📊 BorsaAI Yapay Zeka Portalı</a>
        </div>
    </nav>
    <div class="container">
        <div class="row justify-content-center">
            <div class="col-md-6">
                <div class="card p-4 mb-4">
                    <h5 class="card-title fw-bold mb-3">Hisse Yapay Zeka Analizi</h5>
                    <form method="GET" action="/">
                        <div class="input-group">
                            <input type="text" name="hisse_kodu" class="form-control text-uppercase" placeholder="Örn: THYAO, ASELS, TUPRS" value="{sembol_degeri}" required>
                            <button class="btn btn-primary" type="submit">Analiz Et</button>
                        </div>
                    </form>
                </div>
                {sonuc_alani}
            </div>
        </div>
    </div>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def web_arayuzu(hisse_kodu: str = None):
    if not hisse_kodu:
        return HTML_SAYFASI.format(sembol_degeri="", sonuc_alani="")
        
    bist_sembol = f"{hisse_kodu.upper()}.IS"
    hisse = yf.Ticker(bist_sembol)
    df = hisse.history(period="1y")
    
    if df.empty or len(df) < 20:
        hata_kutusu = f'<div class="card p-4"><div class="alert alert-danger m-0">Hata: {hisse_kodu.upper()} için veri alınamadı. Kodun doğruluğunu kontrol edin.</div></div>'
        return HTML_SAYFASI.format(sembol_degeri=hisse_kodu.upper(), sonuc_alani=hata_kutusu)
        
    # İndikatör Hesaplamaları
    df['SMA_14'] = df['Close'].rolling(window=14).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    df['Fiyat_Degisim'] = df['Close'].pct_change()
    
    # Yapay Zeka Hazırlığı
    df['Hedef'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    df_model = df.dropna().copy()
    
    if df_model.empty:
        hata_kutusu = '<div class="card p-4"><div class="alert alert-danger m-0">Hata: Veri seti model için yetersiz.</div></div>'
        return HTML_SAYFASI.format(sembol_degeri=hisse_kodu.upper(), sonuc_alani=hata_kutusu)
        
    oznitelikler = ['SMA_14', 'RSI_14', 'Fiyat_Degisim']
    X = df_model[oznitelikler]
    y = df_model['Hedef']
    
    # Random Forest Model Eğitimi
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Son günün verisi ile yarını tahmin etme
    son_gun_verisi = df[oznitelikler].iloc[-1].values.reshape(1, -1)
    tahmin = model.predict(son_gun_verisi)[0]
    tahmin_olasiligi = model.predict_proba(son_gun_verisi)[0]
    
    ai_sinyali = "YUKARI (Yükseliş Beklentisi)" if tahmin == 1 else "AŞAĞI (Düşüş Beklentisi)"
    guven_skoru = tahmin_olasiligi[1] if tahmin == 1 else tahmin_olasiligi[0]
    rsi_anlik = df['RSI_14'].iloc[-1]
    
    bg_renk = "bg-yukari" if tahmin == 1 else "bg-asagi"
    rsi_renk = "text-danger" if rsi_anlik > 70 else ("text-success" if rsi_anlik < 30 else "text-warning")
    
    kart_html = f"""
    <div class="card p-4">
        <div class="d-flex justify-content-between align-items-center mb-3">
            <h3 class="fw-bold m-0">{hisse_kodu.upper()}</h3>
            <span class="text-muted small">Tarih: {str(df.index[-1].date())}</span>
        </div>
        <hr>
        <div class="row text-center my-3">
            <div class="col">
                <div class="text-muted small">Son Kapanış</div>
                <div class="fs-4 fw-bold text-dark">{round(df['Close'].iloc[-1], 2)} TL</div>
            </div>
            <div class="col">
                <div class="text-muted small">RSI (14)</div>
                <div class="fs-4 fw-bold {rsi_renk}">{round(rsi_anlik, 2)}</div>
            </div>
        </div>
        <div class="card p-3 text-center {bg_renk}">
            <div class="small fw-bold text-uppercase">Yapay Zeka Öngörüsü</div>
            <div class="fs-5 fw-bold my-1">{ai_sinyali}</div>
            <div class="small">Model Güven Oranı: <strong>%{round(guven_skoru * 100, 2)}</strong></div>
        </div>
    </div>
    """
    return HTML_SAYFASI.format(sembol_degeri=hisse_kodu.upper(), sonuc_alani=kart_html)
