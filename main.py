from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import json

app = FastAPI(title="AslanYatırım Yapay Zeka Portalı", version="0.8.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Arka planda taranacak sabit popüler hisseler listesi
POPULER_HISSELER = ["THYAO", "ASELS", "TUPRS"]

def hisse_analiz_motoru(hisse_kodu: str):
    """Verilen hisse için yapay zeka modelini eğitip tahmin ve detayları döner."""
    try:
        bist_sembol = f"{hisse_kodu.upper()}.IS"
        hisse = yf.Ticker(bist_sembol)
        df = hisse.history(period="1y")
        
        if df.empty or len(df) < 20:
            return None
            
        df['SMA_14'] = df['Close'].rolling(window=14).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        df['Fiyat_Degisim'] = df['Close'].pct_change()
        
        df['Hedef'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
        df_model = df.dropna().copy()
        
        if df_model.empty:
            return None
            
        oznitelikler = ['SMA_14', 'RSI_14', 'Fiyat_Degisim']
        X = df_model[oznitelikler]
        y = df_model['Hedef']
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        son_gun_verisi = df[oznitelikler].iloc[-1].values.reshape(1, -1)
        tahmin = model.predict(son_gun_verisi)[0]
        tahmin_olasiligi = model.predict_proba(son_gun_verisi)[0]
        
        guven_skoru = tahmin_olasiligi[1] if tahmin == 1 else tahmin_olasiligi[0]
        
        return {
            "hisse": hisse_kodu.upper(),
            "fiyat": round(df['Close'].iloc[-1], 2),
            "rsi": round(df['RSI_14'].iloc[-1], 2),
            "tahmin": tahmin,
            "guven": round(guven_skoru * 100, 2),
            "tarih": str(df.index[-1].date()),
            "df": df
        }
    except:
        return None

@app.get("/", response_class=HTMLResponse)
def web_arayuzu(hisse_kodu: str = None):
    sonuc_alani = ""
    sembol_degeri = ""
    
    # 🦁 ASLAN FİLTRESİ: Sabit listeyi tara ve en yüksek güven oranına sahip olanı bul
    en_iyi_tahmin = None
    for h in POPULER_HISSELER:
        res = hisse_analiz_motoru(h)
        if res:
            if en_iyi_tahmin is None or res["guven"] > en_iyi_tahmin["guven"]:
                en_iyi_tahmin = res

    # Aslan Filtresi Banner Alanı Tasarımı
    if en_iyi_tahmin:
        yon_yazi = "YUKARI (Yükseliş)" if en_iyi_tahmin["tahmin"] == 1 else "AŞAĞI (Düşüş)"
        badge_renk = "bg-success" if en_iyi_tahmin["tahmin"] == 1 else "bg-danger"
        aslan_filtresi_html = f"""
        <div class="card p-3 mb-
