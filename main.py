from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

app = FastAPI(title="AslanYatırım Yapay Zeka Portalı", version="0.6.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def web_arayuzu(hisse_kodu: str = None):
    sonuc_alani = ""
    sembol_degeri = ""
    
    if hisse_kodu:
        sembol_degeri = hisse_kodu.upper()
        bist_sembol = f"{sembol_degeri}.IS"
        hisse = yf.Ticker(bist_sembol)
        df = hisse.history(period="1y")
        
        if df.empty or len(df) < 20:
            sonuc_alani = f"""
            <div class="card p-4 text-center border-danger animate-fade w-100">
                <div class="text-danger fw-bold fs-5">⚠️ Hata: {sembol_degeri}</div>
                <div class="text-muted small mt-1">Borsa İstanbul verisi alınamadı. Sembolü doğru girdiğinizden emin olun (Örn: THYAO).</div>
            </div>
            """
        else:
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
                sonuc_alani = '<div class="card p-4 text-center border-danger w-100"><div class="text-danger">Hata: Model için yetersiz veri.</div></div>'
            else:
                oznitelikler = ['SMA_14', 'RSI_14', 'Fiyat_Degisim']
                X = df_model[oznitelikler]
                y = df_model['Hedef']
                
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X, y)
                
                son_gun_verisi = df[oznitelikler].iloc[-1].values.reshape(1, -1)
                tahmin = model.predict(son_gun_verisi)[0]
                tahmin_olasiligi = model.predict_proba(son_gun_verisi)[0]
                
                ai_sinyali = "YUKARI (Yükseliş Beklentisi)" if tahmin == 1 else "AŞAĞI (Düşüş Beklentisi)"
                guven_skoru = tahmin_olasiligi[1] if tahmin == 1 else tahmin_olasiligi[0]
                rsi_anlik = df['RSI_14'].iloc[-1]
                
                bg_renk = "status-yukari" if tahmin == 1 else "status-asagi"
                rsi_renk = "text-danger" if rsi_anlik > 70 else ("text-success" if rsi_anlik < 30 else "text-warning")
                
                sonuc_alani = f"""
                <div class="card p-4 mb-4 animate-fade w-100">
                    <div class="d-flex justify-content-between align-items-center mb-3">
                        <h2 class="fw-bold text-gold m-0">{sembol_degeri}</h2>
                        <span class="text-muted small">Son Veri: {str(df.index[-1].date())}</span>
                    </div>
                    <div class="row text-center my-4">
                        <div class="col">
                            <div class="text-muted small uppercase fw-bold tracking">Kapanış Fiyatı</div>
                            <div class="fs-3 fw-bold text-white mt-1">{round(df['Close'].iloc[-1], 2)} TL</div>
                        </div>
                        <div class="col">
                            <div class="text-muted small uppercase fw-bold tracking">RSI (14)</div>
                            <div class="fs-3 fw-bold {rsi_renk} mt-1">{round(rsi_anlik, 2)}</div>
                        </div>
                    </div>
                    <div class="card p-3 text-center {bg_renk} border-0">
                        <div class="small fw-bold text-uppercase tracking" style="opacity: 0.8;">Yapay Zeka Sinyali</div>
                        <div class="fs-4 fw-bold my-1 text-white">{ai_sinyali}</div>
                        <div class="small text-white-50">Model Güven Oranı: <strong class="text-white">%{round(guven_skoru * 100, 2)}</strong></div>
                    </div>
                </div>

                <div class="card p-4 animate-fade w-100">
                    <div class="d-flex align-items-center justify-content-between mb-3">
                        <h6 class="text-gold m-0 fw-bold uppercase tracking">📈 Teknik Grafik & Trend Analizi</h6>
                        <span class="badge bg-warning text-dark px-2 py-1 small fw-bold">YAKINDA</span>
                    </div>
                    <div class="placeholder-container text-start">
                        <div class="placeholder-bar mb-2" style="width: 85%;"></div>
                        <div class="placeholder-bar mb-2" style="width: 60%;"></div>
                        <div class="placeholder-bar" style="width: 40%;"></div>
                    </div>
                    <div class="text-center text-muted small mt-3 italic">
                        Detaylı indikatör veri tabloları ve yapay zeka grafik modülü çok yakında buraya entegre edilecektir.
                    </div>
                </div>
                """

    return f"""
    <!DOCTYPE html>
    <html lang="tr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AslanYatırım - Premium Yapay Zeka Analiz Paneli</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{ 
                background-color: #0d0f12; 
                color: #f8f9fa;
                font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                margin: 0;
            }}
            .text-gold {{ color: #FFD700 !important; }}
            .bg-gold {{ background-color: #FFD700 !important; color: #0d0f12 !important; }}
            
            .card {{ 
                background-color: #161a22 !important; 
                border: 1px solid #242b37 !important; 
                border-radius: 16px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            }}
            
            .form-control {{
                background-color: #1f242e !important;
                border: 1px solid #323b4c !important;
                color: #ffffff !important;
                border-radius: 10px 0 0 10px;
                padding: 12px 16px;
            }}
            .form-control:focus {{
                border-color: #FFD700 !important;
                box-shadow: 0 0 0 0.25rem rgba(255, 215, 0, 0.15) !important;
                color: #ffffff !important;
            }}
            
            .btn-gold {{
                background-color: #FFD700 !important;
                color: #0d0f12 !important;
                font-weight: bold;
                border-radius: 0 10px 10px 0;
                padding: 12px 24px;
                transition: all 0.2s ease;
            }}
            
            .status-yukari {{ background: linear-gradient(135deg, #198754 0%, #0f5132 100%); }}
            .status-asagi {{ background: linear-gradient(135deg, #dc3545 0%, #842029 100%); }}
            
            .uppercase {{ text-transform: uppercase; }}
            .tracking {{ letter-spacing: 1px; }}
            .italic {{ font-style: italic; }}
            
            .main-content {{
                flex: 1;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 40px 20px;
            }}
            .center-container {{
                width: 100%;
                max-width: 500px;
                display: flex;
                flex-direction: column;
                align-items: center;
            }}
            
            .placeholder-container {{
                background: #1f242e;
                padding: 20px;
                border-radius: 10px;
                border: 1px dashed #323b4c;
            }}
            .placeholder-bar {{
                height: 12px;
                background: linear-gradient(90deg, #242b37 25%, #323b4c 50%, #242b37 75%);
                background-size: 200% 100%;
                animation: loading 1.5s infinite;
                border-radius: 6px;
            }}
            
            @keyframes loading {{
                0% {{ background-position: 200% 0; }}
                100% {{ background-position: -200% 0; }}
            }}
            
            .animate-fade {{
                animation: fadeIn 0.4s ease-out forwards;
            }}
            @keyframes fadeIn {{
                from {{ opacity: 0; transform: translateY(10px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
        </style>
    </head>
    <body>
        <nav class="navbar navbar-dark py-3" style="background-color: #0a0c0e; border-bottom: 1px solid #1a1f29; width: 100%;">
            <div class="container justify-content-center">
                <span class="navbar-brand fw-bold text-gold fs-4 tracking">🦁 ASLAN YATIRIM PORTALI</span>
            </div>
        </nav>
        
        <div class="main-content">
            <div class="center-container">
                
                <div class="card p-4 mb-4 w-100">
                    <h5 class="card-title text-gold fw-bold mb-3 uppercase tracking text-center">Yapay Zeka Hisse Analiz Sistemi</h5>
                    <form method="GET" action="/">
                        <div class="input-group">
                            <input type="text" name="hisse_kodu" class="form-control text-uppercase" placeholder="Örnek: THYAO, ASELS, TUPRS" value="{sembol_degeri}" required>
                            <button class="btn btn-gold uppercase" type="submit">Analiz</button>
                        </div>
                    </form>
                </div>
                
                {sonuc_alani}
                
            </div>
        </div>
        
        <footer class="text-center py-3 text-white-50" style="background-color: #0a0c0e; border-top: 1px solid #1a1f29; font-size: 0.85rem; width: 100%;">
            © 2026 AslanYatırım. Tüm Hakları Saklıdır. | Yapay Zeka Destekli Borsa Analiz Modülü
        </footer>
    </body>
    </html>
    """
