import re
from typing import Dict, List, Optional


def detectar_plataforma(url: str) -> str:
    url_lower = url.lower()

    if "instagram.com" in url_lower or "instagram" in url_lower:
        return "Instagram"
    if "tiktok.com" in url_lower or "tiktok" in url_lower:
        return "TikTok"
    if "youtube.com" in url_lower or "youtu.be" in url_lower or "youtube" in url_lower:
        return "YouTube"

    return "Desconhecida"


def extrair_estrelas(label: str) -> Optional[int]:
    match = re.search(r"(\d)", str(label))
    return int(match.group(1)) if match else None


def classificar_sentimento(label: str) -> str:
    estrelas = extrair_estrelas(label)

    if estrelas in [1, 2]:
        return "Negativo"
    if estrelas in [3, 4, 5]:
        return "Positivo/Neutro"
    return "Indefinido"


def simular_comentarios_por_link(url: str) -> List[str]:
    url_lower = url.lower()

    if "top" in url_lower or "good" in url_lower or "sucesso" in url_lower:
        return [
            "Amei essa campanha!",
            "Muito bom 👏",
            "Excelente conteúdo",
            "Sensacional",
            "Gostei bastante",
            "Muito profissional"
        ]

    if "fail" in url_lower or "bad" in url_lower or "crise" in url_lower:
        return [
            "Péssimo",
            "Não gostei",
            "Propaganda enganosa",
            "Horrível",
            "Muito ruim",
            "Não recomendo"
        ]

    return [
        "Amei essa campanha!",
        "Não gostei",
        "Achei ok",
        "Produto péssimo",
        "Sensacional",
        "Esperava mais"
    ]


def classificar_fake(comentario: str) -> Dict[str, str]:
    texto = comentario.lower()

    negativos = [
        "péssimo", "não gostei", "nao gostei", "propaganda enganosa",
        "ruim", "horrível", "horrivel", "não recomendo", "nao recomendo",
        "esperava mais", "decepcionante", "fraco"
    ]

    if any(termo in texto for termo in negativos):
        label = "1 star"
    else:
        label = "5 stars"

    return {
        "comentario": comentario,
        "label_modelo": label,
        "sentimento": classificar_sentimento(label)
    }


def analisar_link_simulado(url: str, plataforma: Optional[str] = None) -> dict:
    plataforma_detectada = detectar_plataforma(url)
    plataforma_final = plataforma or plataforma_detectada

    comentarios = simular_comentarios_por_link(url)
    classificados = [classificar_fake(c) for c in comentarios]

    positivos = sum(1 for item in classificados if item["sentimento"] == "Positivo/Neutro")
    negativos = sum(1 for item in classificados if item["sentimento"] == "Negativo")

    return {
        "url": url,
        "plataforma": plataforma_final,
        "origem": "backend_simulado",
        "total_comentarios": len(classificados),
        "positivos_neutros": positivos,
        "negativos": negativos,
        "comentarios": classificados
    }


def analisar_multiplos_links_simulado(urls: List[str], plataforma: Optional[str] = None) -> dict:
    todos_comentarios = []
    plataformas_detectadas = []

    for url in urls:
        resultado_link = analisar_link_simulado(url=url, plataforma=plataforma)
        plataformas_detectadas.append(resultado_link["plataforma"])

        for comentario in resultado_link["comentarios"]:
            comentario_com_origem = dict(comentario)
            comentario_com_origem["url_origem"] = url
            todos_comentarios.append(comentario_com_origem)

    positivos = sum(1 for item in todos_comentarios if item["sentimento"] == "Positivo/Neutro")
    negativos = sum(1 for item in todos_comentarios if item["sentimento"] == "Negativo")

    plataforma_final = plataforma or (plataformas_detectadas[0] if plataformas_detectadas else "Desconhecida")

    return {
        "urls": urls,
        "plataforma": plataforma_final,
        "origem": "backend_simulado_multilinks",
        "total_links": len(urls),
        "total_comentarios": len(todos_comentarios),
        "positivos_neutros": positivos,
        "negativos": negativos,
        "comentarios": todos_comentarios
    }
