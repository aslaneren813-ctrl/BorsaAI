from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

app = FastAPI(title="BorsaAI Yapay Zeka Portalı", version="0.3.0")

# CORS ayarları (Telefondan veya dışarıdan istek atarken sorun çıkmaması için)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# HTML Şablonlarının yerini tanımlıyoruz
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def web_arayuzu(request: Request, hisse_kodu: str = None):
    # Eğer kullanıcı arama kutusuna bir şey yazmadıysa boş sayfa göster
    if not hisse_kodu:
        return templates.TemplateResponse("index.html", {"request": request, "sembol": None})
        
    bist_sembol = f"{hisse_kodu.upper()}.IS"
    hisse = yf.Ticker(bist_sembol)
    df = hisse.history(period="1y")
    
    if df.empty or len(df) < 20:
        return templates.TemplateResponse("index.html", {"request": request, "sembol": f"Hata: {hisse_kodu} verisi alınamadı."})
        
    # --- TEKNİK İNDİKATÖRLER ---
    df['SMA_14'] = df['Close'].rolling(window=14).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    df['Fiyat_Degisim'] = df['Close'].pct_change()
    
    # --- YAPAY ZEKA MODELİ HAZIRLIĞI ---
    df['Hedef'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    df_model = df.dropna().copy()
    
    if df_model.empty:
        return templates.TemplateResponse("index.html", {"request": request, "sembol": "Hata: Veri seti hazırlanamadı."})
        
    oznitelikler = ['SMA_14', 'RSI_14', 'Fiyat_Degisim']
    X = df_model[oznitelikler]
    y = df_model['Hedef']
    
    # Modeli oluştur ve eğit
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # --- GELECEK TAHMİNİ ---
    son_gun_verisi = df[oznitelikler].iloc[-1].values.reshape(1, -1)
    tahmin = model.predict(son_gun_verisi)[0]
    tahmin_olasiligi = model.predict_proba(son_gun_verisi)[0]
    
    ai_sinyali = "YUKARI (Yükseliş Beklentisi)" if tahmin == 1 else "AŞAĞI (Düşüş Beklentisi)"
    guven_skoru = tahmin_olasiligi[1] if tahmin == 1 else tahmin_olasiligi[0]
    rsi_anlik = df['RSI_14'].iloc[-1]
    
    # Verileri HTML sayfasına (Jinja2 şablonuna) gönderiyoruz
    return templates.TemplateResponse("index.html", {
        "request": request,
        "sembol": hisse_kodu.upper(),
        "tarih": str(df.index[-1].date()),
        "son_kapanis": round(df['Close'].iloc[-1], 2),
        "rsi_degeri": round(rsi_anlik, 2) if not pd.isna(rsi_anlik) else None,
        "yapay_zeka_ongorusu": ai_sinyali,
        "model_guven_orani": f"%{round(guven_skoru * 100, 2)}"
    })

# API olarak veri çekmek isteyenler için ham endpoint'leri de koruyoruz
@app.get("/api/hisse/{sembol}")
def hisse_bilgi(sembol: str):
    bist_sembol = f"{sembol.upper()}.IS"
    hisse = yf.Ticker(bist_sembol)
    veri = hisse.history(period="1d")
    if veri.empty:
        return {"hata": f"{sembol} sembolü için veri bulunamadı."}
    son_fiyat = veri['Close'].iloc[-1]
    return {"sembol": sembol.upper(), "anlik_fiyat": round(son_fiyat, 2), "para_birimi": "TRY"}
