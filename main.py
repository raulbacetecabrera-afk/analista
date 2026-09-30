import math
import os
import requests
import re
from typing import List, Dict, Any
from fastapi import FastAPI
import uvicorn

# ==========================================
# 1. CAPA DE MODELOS PREDICTIVOS
# ==========================================

class ApuestasAlgoritmo:
    def calcular_cuota_implicita(self, cuota_decimal: float) -> float:
        return 1.0 / cuota_decimal

    def calcular_ev(self, probabilidad_real: float, cuota_oficial: float) -> float:
        return (probabilidad_real * cuota_oficial) - 1.0

class FutbolPredictor(ApuestasAlgoritmo):
    def __init__(self, xg_local: float, xg_visitante: float):
        self.xg_local = xg_local
        self.xg_visitante = xg_visitante

    def _poisson_probability(self, lam: float, k: int) -> float:
        return (math.exp(-lam) * (lam ** k)) / math.factorial(k)

    def calcular_probabilidades_partido(self, max_goles: int = 5) -> Dict[str, float]:
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
    def __init__(self):
        self.keywords_bajas = ["lesionado", "baja", "sancionado", "descanso", "ausente", "suplente", "duda"]
        self.keywords_positivas = ["recupera", "titular", "en racha", "vuelve", "disponible", "entrena"]

    def scrapear_noticias_equipo(self, equipo: str) -> str:
        # Simulación de extracción. Aquí se integraría BeautifulSoup contra Marca o As.
        simulaciones = {
            "Real Madrid": "El delantero titular queda fuera por lesión y el portero titular descansará por rotación.",
            "Barcelona": "El equipo recupera a su estrella tras un mes de baja, entrena con normalidad."
        }
        return simulaciones.get(equipo, "Noticias estándar, sin bajas relevantes reportadas hoy.")

    def analizar_texto_partido(self, equipo: str) -> Dict[str, Any]:
        titular_o_texto = self.scrapear_noticias_equipo(equipo)
        texto_limpio = titular_o_texto.lower()
        
        impacto_negativo = sum(1 for kw in self.keywords_bajas if re.search(r'\b' + kw + r'\b', texto_limpio))
        impacto_positivo = sum(1 for kw in self.keywords_positivas if re.search(r'\b' + kw + r'\b', texto_limpio))
        
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
    def __init__(self, target_odds: float = 2.00):
        self.target_odds = target_odds

    def generar_combinada_segura(self, selecciones_bajas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        combinadas_sugeridas = []
        selecciones_bajas.sort(key=lambda x: x["prob_real_num"], reverse=True)
        
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
# 4. CAPA DE ESCANEO DE MERCADO (MULTI-LIGA Y DOBLE OPORTUNIDAD)
# ==========================================

class MarketScanner:
    def __init__(self):
        self.api_key = os.environ.get("ODDS_API_KEY", "")
        # Ampliamos a 3 grandes ligas europeas
        self.sport_keys = ["soccer_spain_la_liga", "soccer_epl", "soccer_italy_serie_a"]
        self.nlp = NewsSentimentExtractor()
        self.builder = BetBuilderOptimizer(target_odds=2.00)

    def obtener_partidos_del_dia(self) -> List[Dict[str, Any]]:
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
                            # Fórmula matemática para crear una doble oportunidad usando mercados 1X2 sin margen adicional
                            cuotas["1X"] = round(1 / ((1 / cuotas["1"]) + (1 / cuotas["X"])), 2)
                            cuotas["X2"] = round(1 / ((1 / cuotas["2"]) + (1 / cuotas["X"])), 2)
                            
                            partidos.append({"id": match["id"], "local": home_team, "visitante": away_team, "cuotas": cuotas})
            except requests.exceptions.RequestException:
                continue
                
        return partidos

    def escanear_valor_escalera(self) -> Dict[str, Any]:
        selecciones_seguras = []
        partidos = self.obtener_partidos_del_dia()

        for p in partidos:
            noticias_local = self.nlp.analizar_texto_partido(p["local"])
            noticias_visitante = self.nlp.analizar_texto_partido(p["visitante"])
            
            xg_local = round((3.0 / p["cuotas"]["1"]) * noticias_local["factor_ajuste"], 2)
            xg_visitante = round((3.0 / p["cuotas"]["2"]) * noticias_visitante["factor_ajuste"], 2)

            predictor = FutbolPredictor(xg_local=xg_local, xg_visitante=xg_visitante)
            probabilidades_reales = predictor.calcular_probabilidades_partido()

            # Solo evaluamos los mercados conservadores de Doble Oportunidad
            mercados_dobles = ["1X", "X2"]
            for seleccion in mercados_dobles:
                cuota_mercado = p["cuotas"][seleccion]
                
                # Ampliamos ligeramente el rango de cuota baja aceptable para dobles oportunidades (1.05 a 1.45)
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

@app.get("/")
def home():
    return {"mensaje": "IA de Apuestas Activa. Usa el endpoint /reto-escalera para ver las apuestas agrupadas."}

@app.get("/reto-escalera")
def get_escalera_del_dia():
    if not scanner.api_key:
        return {"error": "Falta la variable de entorno ODDS_API_KEY en Render."}
        
    resultados = scanner.escanear_valor_escalera()
    
    if not resultados["selecciones_simples_seguras"]:
        return {"status": "ok", "mensaje": "Hoy no hay selecciones suficientemente seguras en las ligas monitorizadas."}
    
    return {"status": "ok", "resultados": resultados}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)