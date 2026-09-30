import math
import os
import requests
import re
from typing import List, Dict, Any
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

# ==========================================
# 1. CAPA DE MODELOS PREDICTIVOS
# ==========================================

class ApuestasAlgoritmo:
    def calcular_cuota_implicita(self, cuota_decimal: float) -> float:
        """Convierte la cuota ofrecida por la casa en probabilidad implícita."""
        return 1.0 / cuota_decimal

    def calcular_ev(self, probabilidad_real: float, cuota_oficial: float) -> float:
        """
        Calcula el Valor Esperado (Expected Value - EV).
        Si el resultado es > 0, la apuesta tiene valor matemático y es rentable a largo plazo.
        """
        return (probabilidad_real * cuota_oficial) - 1.0

class FutbolPredictor(ApuestasAlgoritmo):
    def __init__(self, xg_local: float, xg_visitante: float):
        self.xg_local = xg_local
        self.xg_visitante = xg_visitante

    def _poisson_probability(self, lam: float, k: int) -> float:
        return (math.exp(-lam) * (lam ** k)) / math.factorial(k)

    def calcular_probabilidades_partido(self, max_goles: int = 5) -> Dict[str, float]:
        """Genera la matriz de resultados usando distribución de Poisson."""
        prob_local = 0.0
        prob_empate = 0.0
        prob_visitante = 0.0

        for goles_local in range(max_goles + 1):
            for goles_visitante in range(max_goles + 1):
                prob = self._poisson_probability(self.xg_local, goles_local) * \
                       self._poisson_probability(self.xg_visitante, goles_visitante)
                
                if goles_local > goles_visitante:
                    prob_local += prob
                elif goles_local == goles_visitante:
                    prob_empate += prob
                else:
                    prob_visitante += prob

        return {
            "1": round(prob_local, 4), 
            "X": round(prob_empate, 4), 
            "2": round(prob_visitante, 4),
            # Calculamos las probabilidades reales conjuntas para Doble Oportunidad
            "1X": round(prob_local + prob_empate, 4),
            "X2": round(prob_visitante + prob_empate, 4)
        }

# ==========================================
# 2. CAPA DE SCRAPING Y NLP (NOTICIAS)
# ==========================================

class NewsSentimentExtractor:
    """
    Analiza noticias y reportes textuales para extraer factores clave 
    que impacten directamente en las probabilidades (bajas, rotaciones, fatiga).
    """
    def __init__(self):
        # Palabras clave críticas para ponderar el impacto en el rendimiento
        self.keywords_bajas = ["lesionado", "baja", "sancionado", "descanso", "ausente", "suplente", "duda"]
        self.keywords_positivas = ["recupera", "titular", "en racha", "vuelve", "disponible", "entrena"]

    def scrapear_noticias_equipo(self, equipo: str) -> str:
        """
        Simula un web scraping básico a portales deportivos buscando noticias del equipo.
        En producción, usarías BeautifulSoup para extraer los titulares reales.
        """
        simulaciones = {
            "Real Madrid": "El delantero titular queda fuera por lesión y el portero titular descansará por rotación.",
            "Barcelona": "El equipo recupera a su estrella tras un mes de baja, entrena con normalidad."
        }
        return simulaciones.get(equipo, "Noticias estándar, sin bajas relevantes reportadas hoy.")

    def analizar_texto_partido(self, equipo: str) -> Dict[str, Any]:
        """
        Escanea el texto en busca de impactos en la alineación.
        Retorna un multiplicador de ajuste de rendimiento para el equipo.
        """
        titular_o_texto = self.scrapear_noticias_equipo(equipo)
        texto_limpio = titular_o_texto.lower()
        
        impacto_negativo = sum(1 for kw in self.keywords_bajas if re.search(r'\b' + kw + r'\b', texto_limpio))
        impacto_positivo = sum(1 for kw in self.keywords_positivas if re.search(r'\b' + kw + r'\b', texto_limpio))
        
        # Factor base: 1.0 = rendimiento estándar proyectado
        factor_ajuste = 1.0 + (impacto_positivo * 0.05) - (impacto_negativo * 0.08)
        factor_ajuste = max(0.60, min(1.30, factor_ajuste))
        
        return {
            "factor_ajuste": round(factor_ajuste, 3),
            "noticias_extraidas": titular_o_texto,
            "impacto": "Positivo" if factor_ajuste > 1 else "Negativo" if factor_ajuste < 1 else "Neutro"
        }

# ==========================================
# 3. CAPA DE CONSTRUCCIÓN DE APUESTAS
# ==========================================

class BetBuilderOptimizer:
    """
    Modela selecciones de mercados correlacionados estilo 'Crear Apuesta' de Bet365
    para buscar selecciones conservadoras que compongan una cuota objetivo.
    """
    def __init__(self, target_odds: float = 2.00):
        self.target_odds = target_odds

    def generar_combinada_segura(self, selecciones_bajas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Toma una lista de apuestas de cuota baja (seguras) y agrupa de 1 a 3 
        para construir los peldaños del reto escalera.
        """
        combinadas_sugeridas = []
        
        # Ordenamos por mayor probabilidad real primero
        selecciones_bajas.sort(key=lambda x: x["prob_real_num"], reverse=True)
        
        # Agrupar de 2 en 2 o de 3 en 3
        if len(selecciones_bajas) >= 2:
            sel_1 = selecciones_bajas[0]
            sel_2 = selecciones_bajas[1]
            cuota_comb = round(sel_1["cuota_oficial"] * sel_2["cuota_oficial"], 2)
            prob_comb = sel_1["prob_real_num"] * sel_2["prob_real_num"]
            
            combinadas_sugeridas.append({
                "tipo": "Combinada Doble Segura (2 partidos)",
                "partidos": [sel_1["partido"], sel_2["partido"]],
                "pronosticos": [sel_1["pronostico"], sel_2["pronostico"]],
                "cuota_total": cuota_comb,
                "probabilidad_conjunta": f"{round(prob_comb * 100, 1)}%",
                "ev_conjunto": f"+{round(((prob_comb * cuota_comb) - 1) * 100, 2)}%"
            })
            
        return combinadas_sugeridas

# ==========================================
# 4. CAPA DE ESCANEO DE MERCADO
# ==========================================

class MarketScanner:
    """
    Se conecta a The-Odds-API en tiempo real para obtener cuotas reales,
    aplica el NLP de noticias y filtra para el reto escalera.
    """
    def __init__(self):
        # Lee la clave desde las variables de entorno del servidor
        self.api_key = os.environ.get("ODDS_API_KEY", "")
        # Ampliamos a 3 grandes ligas europeas
        self.sport_keys = ["soccer_spain_la_liga", "soccer_epl", "soccer_italy_serie_a"]
        self.nlp = NewsSentimentExtractor()
        self.builder = BetBuilderOptimizer(target_odds=2.00)

    def obtener_partidos_del_dia(self) -> List[Dict[str, Any]]:
        """Hace un GET a The-Odds-API para extraer partidos y cuotas en vivo."""
        if not self.api_key:
            return []

        partidos = []
        for liga in self.sport_keys:
            url = f"https://api.the-odds-api.com/v4/sports/{liga}/odds/"
            params = {"apiKey": self.api_key, "regions": "eu", "markets": "h2h", "oddsFormat": "decimal"}
            
            try:
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    for match in data:
                        home_team = match.get("home_team")
                        away_team = match.get("away_team")
                        cuotas = {"1": 0.0, "X": 0.0, "2": 0.0}
                        
                        for bookmaker in match.get("bookmakers", []):
                            if bookmaker["key"] == "bet365" or cuotas["1"] == 0.0:
                                for market in bookmaker.get("markets", []):
                                    if market["key"] == "h2h":
                                        for outcome in market["outcomes"]:
                                            if outcome["name"] == home_team: cuotas["1"] = outcome["price"]
                                            elif outcome["name"] == away_team: cuotas["2"] = outcome["price"]
                                            elif outcome["name"] == "Draw": cuotas["X"] = outcome["price"]
                                if bookmaker["key"] == "bet365": break
                        
                        if cuotas["1"] > 0 and cuotas["X"] > 0 and cuotas["2"] > 0:
                            # Sintetizar la cuota de la casa para la Doble Oportunidad (1X y X2)
                            cuotas["1X"] = round(1 / ((1 / cuotas["1"]) + (1 / cuotas["X"])), 2)
                            cuotas["X2"] = round(1 / ((1 / cuotas["2"]) + (1 / cuotas["X"])), 2)
                            
                            partidos.append({"id": match["id"], "local": home_team, "visitante": away_team, "cuotas": cuotas})
            except requests.exceptions.RequestException:
                continue
                
        return partidos

    def escanear_valor_escalera(self) -> Dict[str, Any]:
        """
        Filtra apuestas de cuota baja con ventaja matemática,
        ajustadas por noticias (NLP), y las agrupa.
        """
        selecciones_seguras = []
        partidos = self.obtener_partidos_del_dia()

        for p in partidos:
            noticias_local = self.nlp.analizar_texto_partido(p["local"])
            noticias_visitante = self.nlp.analizar_texto_partido(p["visitante"])
            
            xg_local = round((3.0 / p["cuotas"]["1"]) * noticias_local["factor_ajuste"], 2)
            xg_visitante = round((3.0 / p["cuotas"]["2"]) * noticias_visitante["factor_ajuste"], 2)

            predictor = FutbolPredictor(xg_local=xg_local, xg_visitante=xg_visitante)
            probabilidades_reales = predictor.calcular_probabilidades_partido()

            mercados_dobles = ["1X", "X2"]
            for seleccion in mercados_dobles:
                cuota_mercado = p["cuotas"][seleccion]
                
                if 1.05 <= cuota_mercado <= 1.45:
                    prob_real = probabilidades_reales[seleccion]
                    ev = predictor.calcular_ev(prob_real, cuota_mercado)

                    if ev > 0.005: 
                        equipo_impactado = noticias_local["impacto"] if seleccion == "1X" else noticias_visitante["impacto"]
                        selecciones_seguras.append({
                            "partido": f"{p['local']} vs {p['visitante']}",
                            "pronostico": "Local o Empate (1X)" if seleccion == "1X" else "Visitante o Empate (X2)",
                            "cuota_oficial": cuota_mercado,
                            "prob_real_num": prob_real,
                            "probabilidad_real": f"{round(prob_real * 100, 1)}%",
                            "estado_equipo": equipo_impactado
                        })

        combinadas = self.builder.generar_combinada_segura(selecciones_seguras)

        return {
            "selecciones_simples_seguras": selecciones_seguras,
            "combinadas_reto_escalera": combinadas
        }

# ==========================================
# 5. INTERFAZ WEB / API (FastAPI)
# ==========================================

app = FastAPI(title="IA Apuestas - Reto Escalera", description="Escáner con NLP y Creador de Apuestas")
scanner = MarketScanner()

@app.get("/api/datos")
def get_datos_api():
    """Endpoint interno que devuelve los datos en formato JSON para que los lea la web."""
    if not scanner.api_key:
        return {"status": "error", "mensaje": "Falta la variable ODDS_API_KEY en Render."}
    
    resultados = scanner.escanear_valor_escalera()
    if not resultados["selecciones_simples_seguras"]:
        return {"status": "vacio", "mensaje": "Hoy no hay selecciones suficientemente seguras en las ligas monitorizadas."}
        
    return {"status": "ok", "resultados": resultados}


@app.get("/", response_class=HTMLResponse)
def home():
    """Endpoint principal que muestra el Dashboard visual (HTML/CSS/JS)."""
    html_content = """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Dashboard | Reto Escalera IA</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
        <style>
            body { font-family: 'Inter', sans-serif; background-color: #0f172a; color: #f8fafc; }
            .card { background-color: #1e293b; border: 1px solid #334155; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
            .loader { border: 3px solid #334155; border-top: 3px solid #10b981; border-radius: 50%; width: 24px; height: 24px; animation: spin 1s linear infinite; }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        </style>
    </head>
    <body class="p-4 md:p-10">
        <div class="max-w-4xl mx-auto">
            <header class="mb-10 text-center md:text-left border-b border-slate-700 pb-6">
                <h1 class="text-3xl md:text-4xl font-extrabold tracking-tight text-white mb-2">
                    Algoritmo <span class="text-emerald-500">Reto Escalera</span>
                </h1>
                <p class="text-slate-400">Escáner de Valor Esperado (EV) y Doble Oportunidad</p>
            </header>

            <div id="loading" class="flex flex-col items-center justify-center py-20">
                <div class="loader mb-4"></div>
                <p class="text-slate-400 font-semibold animate-pulse">Analizando mercados en vivo...</p>
            </div>

            <div id="error-container" class="hidden bg-red-900/30 border border-red-500/50 text-red-200 p-6 rounded-xl text-center">
            </div>

            <div id="content" class="hidden space-y-8">
                
                <!-- Sección de Combinada Recomendada -->
                <div>
                    <h2 class="text-xl font-bold text-slate-200 mb-4 flex items-center">
                        <svg class="w-6 h-6 mr-2 text-emerald-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
                        Combinada Sugerida (Paso Escalera)
                    </h2>
                    <div id="combinada-container" class="grid gap-4"></div>
                </div>

                <!-- Sección de Partidos Base -->
                <div>
                    <h2 class="text-xl font-bold text-slate-200 mb-4 flex items-center">
                        <svg class="w-6 h-6 mr-2 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                        Oportunidades de Cuota Baja (Base)
                    </h2>
                    <div id="simples-container" class="grid md:grid-cols-2 gap-4"></div>
                </div>

            </div>
        </div>

        <script>
            async function cargarDatos() {
                try {
                    const response = await fetch('/api/datos');
                    const data = await response.json();
                    
                    document.getElementById('loading').classList.add('hidden');

                    if (data.status === 'error' || data.status === 'vacio') {
                        const errContainer = document.getElementById('error-container');
                        errContainer.classList.remove('hidden');
                        errContainer.innerHTML = `<p class="font-semibold text-lg">${data.mensaje}</p>`;
                        return;
                    }

                    document.getElementById('content').classList.remove('hidden');

                    // Pintar Combinadas
                    const contCombinada = document.getElementById('combinada-container');
                    data.resultados.combinadas_reto_escalera.forEach(comb => {
                        let partidosHtml = '';
                        comb.partidos.forEach((partido, idx) => {
                            partidosHtml += `<div class="mb-2 bg-slate-900 p-3 rounded border border-slate-700">
                                <div class="text-sm text-slate-400">Partido ${idx+1}</div>
                                <div class="font-bold text-white">${partido}</div>
                                <div class="text-emerald-400 text-sm mt-1">Pronóstico: ${comb.pronosticos[idx]}</div>
                            </div>`;
                        });

                        contCombinada.innerHTML += `
                            <div class="card p-6 rounded-xl relative overflow-hidden">
                                <div class="absolute top-0 right-0 bg-emerald-600 text-white text-xs font-bold px-3 py-1 rounded-bl-lg">EV ${comb.ev_conjunto}</div>
                                <h3 class="text-lg font-bold mb-4 text-emerald-400">${comb.tipo}</h3>
                                ${partidosHtml}
                                <div class="mt-5 flex justify-between items-end border-t border-slate-700 pt-4">
                                    <div>
                                        <div class="text-slate-400 text-sm">Probabilidad Real</div>
                                        <div class="text-xl font-bold">${comb.probabilidad_conjunta}</div>
                                    </div>
                                    <div class="text-right">
                                        <div class="text-slate-400 text-sm">Cuota Bet365</div>
                                        <div class="text-3xl font-extrabold text-white">${comb.cuota_total}</div>
                                    </div>
                                </div>
                            </div>
                        `;
                    });

                    // Pintar Simples
                    const contSimples = document.getElementById('simples-container');
                    data.resultados.selecciones_simples_seguras.forEach(sel => {
                        contSimples.innerHTML += `
                            <div class="card p-5 rounded-xl">
                                <div class="font-bold text-white mb-1">${sel.partido}</div>
                                <div class="text-blue-400 text-sm mb-4 font-semibold">${sel.pronostico}</div>
                                <div class="flex justify-between text-sm mb-2">
                                    <span class="text-slate-400">Cuota:</span>
                                    <span class="font-bold text-white">${sel.cuota_oficial}</span>
                                </div>
                                <div class="flex justify-between text-sm mb-2">
                                    <span class="text-slate-400">Probabilidad:</span>
                                    <span class="font-bold text-emerald-400">${sel.probabilidad_real}</span>
                                </div>
                                <div class="flex justify-between text-sm">
                                    <span class="text-slate-400">Noticias/Impacto:</span>
                                    <span class="px-2 py-0.5 rounded text-xs font-bold ${sel.estado_equipo === 'Positivo' ? 'bg-emerald-900 text-emerald-300' : 'bg-slate-700 text-slate-300'}">${sel.estado_equipo}</span>
                                </div>
                            </div>
                        `;
                    });

                } catch (error) {
                    document.getElementById('loading').classList.add('hidden');
                    const errContainer = document.getElementById('error-container');
                    errContainer.classList.remove('hidden');
                    errContainer.innerHTML = `<p class="font-bold">Error de conexión.</p><p class="text-sm mt-2">No se pudo contactar con la API.</p>`;
                }
            }

            // Iniciar carga al abrir
            cargarDatos();
        </script>
    </body>
    </html>
    """
    return html_content

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)