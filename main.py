import math
import random
import os
import requests
import re
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from fastapi import FastAPI
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
            "2": round(prob_visitante, 4)
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
        # Ejemplo de implementación BeautifulSoup (comentado para evitar bloqueos por scraping real sin headers)
        # url = f"https://www.google.com/search?q=noticias+lesiones+{equipo.replace(' ', '+')}"
        # response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        # soup = BeautifulSoup(response.text, 'html.parser')
        # titulares = " ".join([h3.text for h3 in soup.find_all('h3')])
        
        # Simulación de titulares recientes extraídos de internet:
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
# 3. CAPA DE CONSTRUCCIÓN DE APUESTAS (RETO ESCALERA)
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
                "tipo": "Combinada Doble (2 partidos)",
                "partidos": [sel_1["partido"], sel_2["partido"]],
                "pronosticos": [sel_1["pronostico"], sel_2["pronostico"]],
                "cuota_total": cuota_comb,
                "probabilidad_conjunta": f"{round(prob_comb * 100, 1)}%",
                "ev_conjunto": f"+{round(((prob_comb * cuota_comb) - 1) * 100, 2)}%"
            })
            
        return combinadas_sugeridas

# ==========================================
# 4. CAPA DE ESCANEO DE MERCADO ONLINE (ACTUALIZADA)
# ==========================================

class MarketScanner:
    """
    Se conecta a The-Odds-API en tiempo real para obtener cuotas reales,
    aplica el NLP de noticias y filtra para el reto escalera.
    """
    def __init__(self):
        # Lee la clave desde las variables de entorno del servidor
        self.api_key = os.environ.get("ODDS_API_KEY", "")
        self.sport_key = "soccer_spain_la_liga" 
        self.nlp = NewsSentimentExtractor()
        self.builder = BetBuilderOptimizer(target_odds=2.00)

    def obtener_partidos_del_dia(self) -> List[Dict[str, Any]]:
        """Hace un GET a The-Odds-API para extraer partidos y cuotas en vivo."""
        if not self.api_key:
            # Fallback a datos simulados si no hay API Key para demostrar la lógica
            return [
                {"id": 1, "local": "Real Madrid", "visitante": "Getafe", "cuotas": {"1": 1.25, "X": 5.50, "2": 12.0}},
                {"id": 2, "local": "Barcelona", "visitante": "Las Palmas", "cuotas": {"1": 1.30, "X": 5.00, "2": 9.50}},
                {"id": 3, "local": "Athletic Club", "visitante": "Almeria", "cuotas": {"1": 1.45, "X": 4.20, "2": 7.00}}
            ]

        url = f"https://api.the-odds-api.com/v4/sports/{self.sport_key}/odds/"
        params = {"apiKey": self.api_key, "regions": "eu", "markets": "h2h", "oddsFormat": "decimal"}
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException:
            return []

        partidos = []
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
            
            if cuotas["1"] > 0 and cuotas["2"] > 0:
                partidos.append({"id": match["id"], "local": home_team, "visitante": away_team, "cuotas": cuotas})
                
        return partidos

    def escanear_valor_escalera(self) -> Dict[str, Any]:
        """
        Filtra apuestas de cuota baja (1.15 a 1.50) con ventaja matemática,
        ajustadas por noticias (NLP), y las agrupa.
        """
        selecciones_seguras = []
        partidos = self.obtener_partidos_del_dia()

        for p in partidos:
            # 1. Extracción de Noticias
            noticias_local = self.nlp.analizar_texto_partido(p["local"])
            noticias_visitante = self.nlp.analizar_texto_partido(p["visitante"])
            
            # 2. Estimación base de xG a partir de cuotas reales e impacto de noticias
            xg_local = round((3.0 / p["cuotas"]["1"]) * noticias_local["factor_ajuste"], 2)
            xg_visitante = round((3.0 / p["cuotas"]["2"]) * noticias_visitante["factor_ajuste"], 2)

            predictor = FutbolPredictor(xg_local=xg_local, xg_visitante=xg_visitante)
            probabilidades_reales = predictor.calcular_probabilidades_partido()

            # 3. Filtrar solo favoritos de cuota baja para "asegurar al máximo" (ej. cuotas menores a 1.50)
            for seleccion, cuota_mercado in p["cuotas"].items():
                if 1.10 <= cuota_mercado <= 1.50:
                    prob_real = probabilidades_reales[seleccion]
                    ev = predictor.calcular_ev(prob_real, cuota_mercado)

                    if ev > 0.01: # Buscamos EV positivo incluso en cuotas bajas
                        selecciones_seguras.append({
                            "partido": f"{p['local']} vs {p['visitante']}",
                            "pronostico": seleccion,
                            "cuota_oficial": cuota_mercado,
                            "prob_real_num": prob_real,
                            "probabilidad_real": f"{round(prob_real * 100, 1)}%",
                            "estado_equipo": noticias_local["impacto"] if seleccion == "1" else noticias_visitante["impacto"]
                        })

        # 4. Construir las combinadas para la escalera
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
    """Endpoint principal: devuelve las selecciones de cuota baja y las combinadas de 2-3 partidos."""
    resultados = scanner.escanear_valor_escalera()
    
    if not resultados["selecciones_simples_seguras"]:
        return {"status": "ok", "mensaje": "Hoy no hay selecciones suficientemente seguras."}
    
    return {"status": "ok", "resultados": resultados}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)