from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
from datetime import datetime

app = FastAPI(title="BorsaAI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def ana_sayfa():
    return {"mesaj": "BorsaAI çalışıyor", "zaman": str(datetime.now())}

@app.get("/hisse/{sembol}")
def hisse_bilgi(sembol: str):
    hisse = yf.Ticker(sembol + ".IS")
    bilgi = hisse.info
    return {
        "sembol": sembol,
        "fiyat": bilgi.get("currentPrice"),
        "onceki_kapanis": bilgi.get("previousClose"),
        "hacim": bilgi.get("volume"),
        "sirket_adi": bilgi.get("longName"),
    }

@app.get("/analiz/{sembol}")
def hisse_analiz(sembol: str):
    hisse = yf.Ticker(sembol + ".IS")
    gecmis = hisse.history(period="1mo")
    if gecmis.empty:
        return {"hata": "Veri bulunamadı"}
    son_fiyat = gecmis["Close"].iloc[-1]
    ortalama = gecmis["Close"].mean()
    return {
        "sembol": sembol,
        "son_fiyat": round(son_fiyat, 2),
        "aylik_ortalama": round(ortalama, 2),
        "sinyal": "AL" if son_fiyat < ortalama else "BEKLE",
    }
from fastapi import FastAPI
import yfinance as yf
import pandas as pd # Veri işleme ve indikatör hesaplama için eklendi

app = FastAPI(title="BorsaAI API", version="0.1.0")

# ... (Burada senin daha önce yazdığın ana_sayfa ve hisse_bilgi fonksiyonların kalacak) ...

@app.get("/analiz/{sembol}")
def hisse_analiz(sembol: str):
    bist_sembol = f"{sembol.upper()}.IS"
    hisse = yf.Ticker(bist_sembol)
    
    # İndikatörleri sağlıklı hesaplamak için geriye dönük 3 aylık veri çekiyoruz
    df = hisse.history(period="3mo")
    
    if df.empty:
        return {"hata": f"{sembol} için veri bulunamadı."}
        
    # 1. İndikatör: 14 Günlük SMA (Basit Hareketli Ortalama)
    df['SMA_14'] = df['Close'].rolling(window=14).mean()
    
    # 2. İndikatör: 14 Günlük RSI (Göreceli Güç Endeksi)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # En güncel (son günün) verilerini çekelim
    son_veri = df.iloc[-1]
    
    # Yapay Zeka Öncesi Basit Karar Algoritması
    rsi_degeri = son_veri['RSI_14']
    tavsiye = "TUT (Piyasa Nötr)"
    
    if pd.isna(rsi_degeri):
        tavsiye = "Hesaplanamadı (Yetersiz Veri)"
    elif rsi_degeri < 30:
        tavsiye = "AL (Aşırı Satım Bölgesi - Fiyat Ucuzlamış Olabilir)"
    elif rsi_degeri > 70:
        tavsiye = "SAT (Aşırı Alım Bölgesi - Fiyat Şişmiş Olabilir)"

    return {
        "sembol": sembol.upper(),
        "tarih": str(son_veri.name.date()), # Verinin tarihi
        "kapanis_fiyati": round(son_veri['Close'], 2),
        "sma_14": round(son_veri['SMA_14'], 2) if not pd.isna(son_veri['SMA_14']) else None,
        "rsi_14": round(rsi_degeri, 2) if not pd.isna(rsi_degeri) else None,
        "teknik_sinyal": tavsiye
    }
from fastapi import FastAPI
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

app = FastAPI(title="BorsaAI API", version="0.2.0")

@app.get("/")
def ana_sayfa():
    return {"mesaj": "BorsaAI Yapay Zeka Destekli Sistemine Hoş Geldiniz!"}

@app.get("/hisse/{sembol}")
def hisse_bilgi(sembol: str):
    bist_sembol = f"{sembol.upper()}.IS"
    hisse = yf.Ticker(bist_sembol)
    veri = hisse.history(period="1d")
    
    if veri.empty:
        return {"hata": f"{sembol} sembolü için veri bulunamadı."}
    
    son_fiyat = veri['Close'].iloc[-1]
    return {
        "sembol": sembol.upper(),
        "anlik_fiyat": round(son_fiyat, 2),
        "para_birimi": "TRY"
    }

@app.get("/analiz/{sembol}")
def hisse_analiz(sembol: str):
    bist_sembol = f"{sembol.upper()}.IS"
    hisse = yf.Ticker(bist_sembol)
    
    # Yapay zekanın öğrenmesi için geriye dönük 1 yıllık geniş veri çekiyoruz
    df = hisse.history(period="1y")
    
    if len(df) < 20:
        return {"hata": f"{sembol} için yetersiz veri (En az 20 günlük veri gerekli)."}
        
    # --- TEKNİK İNDİKATÖRLER ---
    # 1. SMA 14
    df['SMA_14'] = df['Close'].rolling(window=14).mean()
    # 2. RSI 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # Fiyat değişim oranını ekleyelim (Öznitelik / Feature olarak)
    df['Fiyat_Degisim'] = df['Close'].pct_change()
    
    # --- YAPAY ZEKA MODELİ HAZIRLIĞI ---
    # Hedef (Target): Yarınki kapanış bugünkünden yüksekse 1 (YUKARI), düşükse 0 (AŞAĞI)
    df['Hedef'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    
    # Eksik (NaN) verileri temizleyelim
    df_model = df.dropna().copy()
    
    if df_model.empty:
        return {"hata": "Veri setini hazırlarken hata oluştu."}
        
    # Yapay zekaya vereceğimiz ipuçları (Öznitelikler)
    oznitelikler = ['SMA_14', 'RSI_14', 'Fiyat_Degisim']
    
    X = df_model[oznitelikler] # Giriş verileri
    y = df_model['Hedef']      # Gerçek sonuçlar
    
    # Modeli oluştur ve eğit (Geçmiş verileri öğrensin)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # --- GELECEK TAHMİNİ ---
    # En son günün verilerini alıp yarını tahmin ettiriyoruz
    son_gun_verisi = df[oznitelikler].iloc[-1].values.reshape(1, -1)
    tahmin = model.predict(son_gun_verisi)[0]
    tahmin_olasiligi = model.predict_proba(son_gun_verisi)[0]
    
    # Sinyal üretimi
    ai_sinyali = "YUKARI (Yükseliş Beklentisi)" if tahmin == 1 else "AŞAĞI (Düşüş Beklentisi)"
    guven_skoru = tahmin_olasiligi[1] if tahmin == 1 else tahmin_olasiligi[0]
    
    # Klasik RSI yorumu (Karşılaştırmak için)
    rsi_anlik = df['RSI_14'].iloc[-1]
    
    return {
        "sembol": sembol.upper(),
        "tarih": str(df.index[-1].date()),
        "son_kapanis": round(df['Close'].iloc[-1], 2),
        "rsi_degeri": round(rsi_anlik, 2) if not pd.isna(rsi_anlik) else None,
        "yapay_zeka_ongorusu": ai_sinyali,
        "model_guven_orani": f"%{round(guven_skoru * 100, 2)}"
    }