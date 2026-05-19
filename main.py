from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import json

app = FastAPI(title="AslanYatırım Yapay Zeka Portalı", version="0.8.5")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

POPULER_HISSELER = ["THYAO", "ASELS", "TUPRS"]

def hisse_analiz_motoru(hisse_kodu: str):
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
    sembol_degeri = hisse_kodu.upper() if hisse_kodu else ""
    sonuc_alani = ""
    
    # 🦁 ASLAN FİLTRESİ HESAPLAMA
    en_iyi_tahmin = None
    for h in POPULER_HISSELER:
        res = hisse_analiz_motoru(h)
        if res:
            if en_iyi_tahmin is None or res["guven"] > en_iyi_tahmin["guven"]:
                en_iyi_tahmin = res

    aslan_filtresi_html = ""
    if en_iyi_tahmin:
        yon_yazi = "YUKARI (Yükseliş)" if en_iyi_tahmin["tahmin"] == 1 else "AŞAĞI (Düşüş)"
        badge_renk = "bg-success" if en_iyi_tahmin["tahmin"] == 1 else "bg-danger"
        aslan_filtresi_html = f"""
        <div class="card p-3 mb-4 border-gold position-relative overflow-hidden header-aslan animate-fade w-100">
            <div class="d-flex align-items-center justify-content-between">
                <div class="d-flex align-items-center">
                    <span class="fs-3 me-2">🦁</span>
                    <div>
                        <div class="small fw-bold text-gold text-uppercase tracking" style="font-size:0.75rem;">Aslan Filtresi Öne Çıkan</div>
                        <h4 class="fw-bold text-white m-0">{en_iyi_tahmin["hisse"]} <span class="fs-6 text-muted">({en_iyi_tahmin["fiyat"]} TL)</span></h4>
                    </div>
                </div>
                <div class="text-end">
                    <span class="badge {badge_renk} px-2 py-1 small fw-bold mb-1">{yon_yazi}</span>
                    <div class="small text-white-50" style="font-size:0.75rem;">Güven Oranı: <strong class="text-gold">%{en_iyi_tahmin["guven"]}</strong></div>
                </div>
            </div>
        </div>
        """

    # KULLANICI ARAMA YAPTIYSA SONUÇ ALANI
    script_alani = ""
    if hisse_kodu:
        analiz = hisse_analiz_motoru(sembol_degeri)
        if not analiz:
            sonuc_alani = f"""
            <div class="card p-4 text-center border-danger animate-fade w-100">
                <div class="text-danger fw-bold fs-5">⚠️ Hata: {sembol_degeri}</div>
                <div class="text-muted small mt-1">Borsa İstanbul verisi alınamadı. Sembolü doğru girdiğinizden emin olun (Örn: THYAO).</div>
            </div>
            """
        else:
            ai_sinyali = "YUKARI (Yükseliş Beklentisi)" if analiz["tahmin"] == 1 else "AŞAĞI (Düşüş Beklentisi)"
            bg_renk = "status-yukari" if analiz["tahmin"] == 1 else "status-asagi"
            rsi_renk = "text-danger" if analiz["rsi"] > 70 else ("text-success" if analiz["rsi"] < 30 else "text-warning")
            
            son_30_gun = analiz["df"].tail(30)
            tarihler = [str(date.date()) for date in son_30_gun.index]
            fiyatlar = [round(float(val), 2) for val in son_30_gun['Close'].values]
            
            sonuc_alani = f"""
            <div class="card p-4 mb-4 animate-fade w-100">
                <div class="d-flex justify-content-between align-items-center mb-3">
                    <h2 class="fw-bold text-gold m-0">{sembol_degeri}</h2>
                    <span class="text-muted small">Son Veri: {analiz["tarih"]}</span>
                </div>
                <div class="row text-center my-4">
                    <div class="col">
                        <div class="text-muted small uppercase fw-bold tracking">Kapanış Fiyatı</div>
                        <div class="fs-3 fw-bold text-white mt-1">{analiz["fiyat"]} TL</div>
                    </div>
                    <div class="col">
                        <div class="text-muted small uppercase fw-bold tracking">RSI (14)</div>
                        <div class="fs-3 fw-bold {rsi_renk} mt-1">{analiz["rsi"]}</div>
                    </div>
                </div>
                <div class="card p-3 text-center {bg_renk} border-0">
                    <div class="small fw-bold text-uppercase tracking" style="opacity: 0.8;">Yapay Zeka Sinyali</div>
                    <div class="fs-4 fw-bold my-1 text-white">{ai_sinyali}</div>
                    <div class="small text-white-50">Model Güven Oranı: <strong class="text-white">%{analiz["guven"]}</strong></div>
                </div>
            </div>

            <div class="card p-4 animate-fade w-100">
                <h6 class="text-gold mb-3 fw-bold uppercase tracking text-start">📈 Son 30 Günlük Fiyat Trendi</h6>
                <div style="position: relative; height:220px; width:100%;">
                    <canvas id="fiyatGrafik"></canvas>
                </div>
            </div>
            """
            
            # JavaScript kodunu f-string çakışması olmasın diye ham metin olarak ekliyoruz
            script_alani = """
            <script>
                const ctx = document.getElementById('fiyatGrafik').getContext('2d');
                new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: """ + json.dumps(tarihler) + """,
                        datasets: [{
                            label: 'Kapanış Fiyatı (TL)',
                            data: """ + json.dumps(fiyatlar) + """,
                            borderColor: '#FFD700',
                            backgroundColor: 'rgba(255, 215, 0, 0.05)',
                            borderWidth: 2,
                            pointRadius: 2,
                            pointHoverRadius: 5,
                            tension: 0.2
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { legend: { display: false } },
                        scales: {
                            x: { grid: { display: false }, ticks: { color: '#888', font: { size: 10 } } },
                            y: { grid: { color: '#242b37' }, ticks: { color: '#888', font: { size: 10 } } }
                        }
                    }
                });
            </script>
            """

    # ANA HTML ŞABLONU
    html_sablonu = f"""
    <!DOCTYPE html>
    <html lang="tr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AslanYatırım - Premium Yapay Zeka Analiz Paneli</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/bootstrap.min.css" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
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
            .card {{ 
                background-color: #161a22 !important; 
                border: 1px solid #242b37 !important; 
                border-radius: 16px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            }}
            .border-gold {{ border: 1px solid #FFD700 !important; }}
            .header-aslan {{ background: linear-gradient(135deg, #1d1805 0%, #11141a 100%); }}
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
            }}
            .status-yukari {{ background: linear-gradient(135deg, #198754 0%, #0f5132 100%); }}
            .status-asagi {{ background: linear-gradient(135deg, #dc3545 0%, #842029 100%); }}
            .uppercase {{ text-transform: uppercase; }}
            .tracking {{ letter-spacing: 1px; }}
            .main-content {{
                flex: 1;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 40px 20px;
            }}
            .center-container {{ width: 100%; max-width: 500px; display: flex; flex-direction: column; align-items: center; }}
            .animate-fade {{ animation: fadeIn 0.4s ease-out forwards; }}
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
                {aslan_filtresi_html}
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
        {script_alani}
    </body>
    </html>
    """
    return html_sablonu
