import re
import base64
import time
import textwrap
from pathlib import Path
from io import BytesIO
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
import streamlit as st
import requests
from transformers import pipeline

# =========================================================
# CONFIGURAÇÃO DA PÁGINA
# =========================================================
st.set_page_config(
    page_title="SENTIMENTAÇÃO",
    page_icon="📊",
    layout="wide"
)

BACKEND_URL = "http://127.0.0.1:8000"

# =========================================================
# CSS GLOBAL
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"]  {
    font-family: 'Montserrat', sans-serif;
}

.stApp {
    background-color: #FFFFFF;
}

.block-container {
    padding-top: 2.6rem;
    padding-bottom: 1.2rem;
    max-width: 1450px;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* =========================================================
   CABEÇALHO GLOBAL
   ========================================================= */
.header-wrap {
    background: #FFFFFF;
    border: 1px solid #E9E5DC;
    border-radius: 18px;
    padding: 22px 26px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    margin-top: 10px;
    margin-bottom: 24px;
}

.brand-line {
    display: flex;
    align-items: center;
    gap: 18px;
}

.logo-fhits {
    font-size: 34px;
    font-weight: 700;
    color: #1F2A44;
    line-height: 1;
    white-space: nowrap;
}

.logo-star {
    color: #B08D57;
}

.logo-divider {
    width: 1px;
    height: 38px;
    background-color: #E5E7EB;
}

.title-area {
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.main-title {
    font-size: 30px;
    font-weight: 800;
    color: #1F2A44;
    margin: 0;
    line-height: 1.1;
}

.sub-title {
    font-size: 15px;
    color: #6B7280;
    margin-top: 4px;
}

/* =========================================================
   COMPONENTES GERAIS
   ========================================================= */
.side-box {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 16px;
    padding: 18px;
    margin-bottom: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

.card-box {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 16px;
    padding: 16px 18px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    min-height: 120px;
}

.chart-box {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    height: 100%;
}

.summary-box {
    background: linear-gradient(135deg, #FFFFFF 0%, #FCFBF8 100%);
    border: 1px solid #E8E2D7;
    border-radius: 18px;
    padding: 22px;
    color: #374151;
    font-size: 15px;
    line-height: 1.85;
    box-shadow: 0 3px 12px rgba(0,0,0,0.04);
    height: 100%;
}

.summary-tag {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    background: #F5EFE3;
    color: #9A7A43;
    font-size: 12px;
    font-weight: 700;
    margin-bottom: 12px;
}

.card-icon {
    font-size: 22px;
    margin-bottom: 8px;
    line-height: 1;
}

.card-title {
    font-size: 14px;
    font-weight: 600;
    color: #6B7280;
    margin-bottom: 8px;
}

.card-number {
    font-size: 34px;
    font-weight: 800;
    color: #111827;
    line-height: 1.1;
}

.card-sub {
    font-size: 14px;
    margin-top: 6px;
    font-weight: 700;
}

.section-title {
    font-size: 18px;
    font-weight: 800;
    color: #1F2A44;
    margin-bottom: 14px;
}

.green-text {
    color: #4C9A57;
    font-weight: 700;
}

.red-text {
    color: #D94B4B;
    font-weight: 700;
}

.neg-item {
    background: #FFF7F7;
    border: 1px solid #F1CFCF;
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 12px;
    color: #374151;
    box-shadow: 0 1px 5px rgba(0,0,0,0.03);
}

.mode-badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    background: #EEF2F7;
    color: #1F2A44;
    font-size: 12px;
    font-weight: 700;
    margin-top: 8px;
}

.fake-status-box {
    background: #FFFDF8;
    border: 1px solid #E8E2D7;
    border-radius: 14px;
    padding: 14px 16px;
    font-size: 14px;
    color: #5B6470;
    margin-top: 10px;
    line-height: 1.7;
}

/* =========================================================
   FHITS INTERNO - CONTROLES
   ========================================================= */
.fhits-controls {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 16px;
    padding: 18px;
    margin-bottom: 18px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

/* =========================================================
   FHITS INTERNO - RELATÓRIO
   ========================================================= */
.fhits-canvas-bg {
    background: #FFFFFF;
    padding: 22px;
    border-radius: 10px;
}

.fhits-page {
    background: #FFFFFF;
    padding: 34px 34px 22px 34px;
    border: 1px solid #E3DED4;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}

.fhits-title {
    font-size: 28px;
    font-weight: 300;
    color: #111111;
    letter-spacing: 0.3px;
    margin-bottom: 20px;
    line-height: 1.1;
}

.fhits-text {
    font-size: 28px;
    color: #1F1F1F;
    line-height: 1.45;
    margin-bottom: 18px;
    max-width: 100%;
}

.fhits-green {
    color: #2FB04A;
    font-weight: 700;
}

.fhits-red {
    color: #EF3F3F;
    font-weight: 700;
}

.fhits-big {
    font-family: Impact, Haettenschweiler, 'Arial Narrow Bold', sans-serif;
    font-size: 48px;
    font-weight: 700;
    color: #000000;
    letter-spacing: 0.2px;
}

/* Títulos das colunas de comentários */
.fhits-comments-title-green {
    font-size: 13px;
    font-weight: 800;
    color: #2F8F44;
    margin-bottom: 10px;
    text-transform: uppercase;
}

.fhits-comments-title-red {
    font-size: 13px;
    font-weight: 800;
    color: #EF3F3F;
    margin-bottom: 10px;
    text-transform: uppercase;
}

/* Card do comentário */
.fhits-comment-card {
    background: #FFFFFF;
    border: 1px solid #E9E9E9;
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 12px;
    min-height: 96px;
}

.fhits-comment-row {
    display: flex;
    align-items: flex-start;
    gap: 12px;
}

.fhits-avatar {
    width: 42px;
    height: 42px;
    min-width: 42px;
    border-radius: 50%;
    background: linear-gradient(135deg, #D9D9D9 0%, #BFBFBF 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #3A3A3A;
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
}

.fhits-avatar.neg {
    background: linear-gradient(135deg, #D8D8D8 0%, #B9B9B9 100%);
}

.fhits-avatar-img {
    width: 42px;
    height: 42px;
    min-width: 42px;
    border-radius: 50%;
    object-fit: cover;
    border: 1px solid #D7D7D7;
}

.fhits-comment-body {
    flex: 1;
}

.fhits-comment-user {
    font-size: 14px;
    font-weight: 700;
    color: #222222;
    margin-bottom: 6px;
}

.fhits-comment-text {
    font-size: 14px;
    color: #444444;
    line-height: 1.45;
}

.fhits-comment-like {
    font-size: 12px;
    color: #7A7A7A;
    margin-top: 8px;
}

.fhits-logo-bottom {
    text-align: right;
    margin-top: 14px;
    font-size: 30px;
    font-weight: 700;
    color: #1F1F1F;
}

.fhits-logo-bottom-star {
    color: #B08D57;
}

/* =========================================================
   BOTÕES
   ========================================================= */
div.stButton > button {
    background-color: #1F2A44;
    color: white;
    border-radius: 10px;
    border: none;
    font-weight: 600;
    padding: 0.70rem 1rem;
    width: 100%;
}

div.stButton > button:hover {
    background-color: #2A3756;
    color: white;
}

div.stDownloadButton > button {
    background-color: #FFFFFF;
    color: #1F2A44;
    border-radius: 10px;
    border: 1px solid #D1D5DB;
    font-weight: 600;
    padding: 0.70rem 1rem;
    width: 100%;
}

div.stDownloadButton > button:hover {
    border: 1px solid #B08D57;
    color: #B08D57;
}

section[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 16px;
    padding: 14px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

[data-testid="stInfo"] {
    border-radius: 14px;
}

.footer-note {
    color: #8A8A8A;
    font-size: 13px;
    text-align: center;
    margin-top: 16px;
}

.element-container {
    margin-bottom: 0.45rem;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# FUNÇÕES
# =========================================================
@st.cache_resource
def carregar_modelo():
    return pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment"
    )


def limpar_texto(texto):
    if pd.isna(texto):
        return ""
    texto = str(texto).strip()
    texto = re.sub(r"\s+", " ", texto)
    return texto


def extrair_estrelas(label):
    match = re.search(r"(\d)", str(label))
    return int(match.group(1)) if match else None


def classificar_sentimento(label):
    estrelas = extrair_estrelas(label)
    if estrelas in [1, 2]:
        return "Negativo"
    elif estrelas in [3, 4, 5]:
        return "Positivo/Neutro"
    return "Indefinido"


def gerar_resumo(total, positivos, negativos, percentual_negativo):
    if total == 0:
        return "Nenhum comentário foi analisado."

    if percentual_negativo < 10:
        leitura = "predominantemente positiva"
        destaque = "baixo volume de críticas"
    elif percentual_negativo < 25:
        leitura = "majoritariamente positiva, com pontos de atenção"
        destaque = "algumas manifestações críticas relevantes"
    else:
        leitura = "dividida, com atenção importante para comentários negativos"
        destaque = "volume expressivo de críticas"

    return f"""
    <div class="summary-tag">Resumo automático</div>
    <div>
        Foram analisados <b>{total}</b> comentários no total. Desse volume,
        <span class="green-text">{positivos}</span> foram classificados como
        <b>Positivo/Neutro</b> e <span class="red-text">{negativos}</span> como
        <b>Negativo</b>.
        <br><br>
        Os comentários negativos representam
        <span class="red-text">{percentual_negativo:.1f}%</span> da base analisada,
        indicando uma percepção geral <b>{leitura}</b>.
        <br><br>
        Em termos executivos, o material aponta <b>{destaque}</b>, e o painel ajuda
        a identificar rapidamente os comentários negativos mais sensíveis para leitura qualitativa.
    </div>
    """


def contar_termos_criticos(texto, lista_termos):
    texto = texto.lower()
    return sum(1 for termo in lista_termos if termo in texto)


def score_relevancia_negativo(comentario, score_modelo, termos_criticos):
    tamanho = len(comentario)
    score = (
        (score_modelo * 0.6) +
        (min(tamanho / 200, 1) * 0.2) +
        (min(termos_criticos, 3) * 0.2)
    )
    return round(score, 4)


def analisar_comentarios(df, coluna_comentario, modelo):
    comentarios = df[coluna_comentario].fillna("").astype(str).apply(limpar_texto)
    comentarios = comentarios[comentarios != ""].reset_index(drop=True)

    if len(comentarios) == 0:
        return pd.DataFrame(), 0, 0

    resultados = []
    barra = st.progress(0, text="Iniciando análise...")
    inicio_total = time.time()

    for i, comentario in enumerate(comentarios, start=1):
        resultado = modelo(comentario)[0]
        label_modelo = resultado["label"]
        score_modelo = float(resultado["score"])
        sentimento = classificar_sentimento(label_modelo)

        resultados.append({
            "comentario": comentario,
            "label_modelo": label_modelo,
            "sentimento": sentimento,
            "confianca": round(score_modelo, 4)
        })

        progresso = i / len(comentarios)
        barra.progress(progresso, text=f"Analisando comentário {i} de {len(comentarios)}...")

    fim_total = time.time()
    tempo_total = fim_total - inicio_total
    tempo_medio = tempo_total / len(comentarios)
    barra.progress(1.0, text="Análise concluída com sucesso.")

    return pd.DataFrame(resultados), tempo_total, tempo_medio


def preparar_top_negativos(df, quantidade_top=4):
    termos_criticos_base = [
        "péssimo", "horrível", "ridículo", "lixo", "nunca mais",
        "decepção", "terrível", "odiei", "enganado", "enganação",
        "não recomendo", "nao recomendo", "vergonha", "fracasso",
        "muito ruim", "absurdo", "ódio", "odio", "cancelar",
        "problema", "demora", "atraso", "mal atendimento"
    ]

    negativos = df[df["sentimento"] == "Negativo"].copy()

    if negativos.empty:
        return negativos

    negativos["termos_criticos"] = negativos["comentario"].apply(
        lambda x: contar_termos_criticos(x, termos_criticos_base)
    )

    negativos["score_relevancia"] = negativos.apply(
        lambda row: score_relevancia_negativo(
            comentario=row["comentario"],
            score_modelo=row["confianca"],
            termos_criticos=row["termos_criticos"]
        ),
        axis=1
    )

    negativos = negativos.sort_values(
        by=["score_relevancia", "confianca", "termos_criticos"],
        ascending=False
    ).reset_index(drop=True)

    return negativos.head(quantidade_top)


def preparar_top_positivos(df, quantidade_top=4):
    positivos = df[df["sentimento"] == "Positivo/Neutro"].copy()

    if positivos.empty:
        return positivos

    positivos["tamanho"] = positivos["comentario"].apply(len)
    positivos = positivos.sort_values(by=["tamanho"], ascending=False).reset_index(drop=True)

    return positivos.head(quantidade_top)


def gerar_grafico_donut(positivos, negativos):
    fig, ax = plt.subplots(figsize=(3.2, 3.2), facecolor="#FFFFFF")
    valores = [positivos, negativos]
    cores = ["#67B26F", "#E25555"]

    ax.pie(
        valores,
        labels=None,
        colors=cores,
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.42, edgecolor="white"),
        pctdistance=0.78,
        textprops=dict(color="white", fontsize=10, fontweight="bold")
    )

    ax.axis("equal")
    plt.tight_layout()
    return fig


def gerar_grafico_pizza_relatorio(positivos, negativos):
    fig, ax = plt.subplots(figsize=(5.8, 5.8), facecolor="#FFFFFF")
    valores = [positivos, negativos]
    cores = ["#34C759", "#EF3F3F"]

    if positivos + negativos == 0:
        valores = [1]
        cores = ["#D9D9D9"]

    wedges, _, _ = ax.pie(
        valores,
        labels=None,
        colors=cores,
        autopct="%1.1f%%" if positivos + negativos > 0 else "",
        startangle=90,
        counterclock=False,
        textprops=dict(color="white", fontsize=14, fontweight="bold"),
        wedgeprops=dict(edgecolor="white", linewidth=1.2)
    )

    ax.axis("equal")

    legend_labels = ["Positivos / Neutros", "Negativos"] if positivos + negativos > 0 else ["Sem dados"]

    ax.legend(
        wedges,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        fontsize=11,
        handlelength=1.1,
        handletextpad=0.4,
        columnspacing=1.0
    )

    plt.tight_layout()
    return fig


def validar_url_publicacao(url):
    url = url.strip()
    return bool(re.match(r"^https?://", url))



def analisar_link_backend(url, plataforma):
    try:
        response = requests.post(
            f"{BACKEND_URL}/analisar-link",
            json={
                "url": url,
                "plataforma": plataforma
            },
            timeout=60
        )
        response.raise_for_status()
        payload = response.json()
        comentarios = payload.get("comentarios", [])
        if not comentarios:
            return pd.DataFrame(columns=["comentario", "label_modelo", "sentimento"]), payload

        df = pd.DataFrame(comentarios)
        return df, payload
    except requests.exceptions.RequestException as e:
        st.error(f"Erro de conexão com o backend: {e}")
        return None, None
    except Exception as e:
        st.error(f"Erro ao processar resposta do backend: {e}")
        return None, None


def analisar_multiplos_links_backend(lista_links, plataforma):
    try:
        response = requests.post(
            f"{BACKEND_URL}/analisar-links",
            json={
                "urls": lista_links,
                "plataforma": plataforma
            },
            timeout=120
        )
        response.raise_for_status()
        payload = response.json()

        comentarios = payload.get("comentarios", [])
        if not comentarios:
            return pd.DataFrame(columns=["comentario", "label_modelo", "sentimento"]), payload

        df = pd.DataFrame(comentarios)
        return df, payload

    except requests.exceptions.RequestException as e:
        st.error(f"Erro de conexão com o backend: {e}")
        return None, None
    except Exception as e:
        st.error(f"Erro ao processar resposta do backend: {e}")
        return None, None


def simular_comentarios_por_link(url):
    url_lower = url.lower()

    if "top" in url_lower or "good" in url_lower or "sucesso" in url_lower:
        comentarios = [
            "Amei essa campanha, ficou incrível!",
            "Muito bom, parabéns pelo trabalho 👏",
            "Sensacional, comunicação perfeita!",
            "Gostei demais, continuem assim!",
            "Marca mandou muito bem dessa vez",
            "Excelente conteúdo!",
            "Criativo e bem executado",
            "Muito acima da média",
            "Achei muito profissional",
            "Uma das melhores campanhas que já vi"
        ]

    elif "fail" in url_lower or "bad" in url_lower or "crise" in url_lower:
        comentarios = [
            "Péssimo, não compro mais.",
            "Propaganda enganosa.",
            "Não gostei, achei fraco.",
            "Muito ruim, esperava mais.",
            "Decepcionante.",
            "Horrível, comunicação confusa.",
            "Não recomendo.",
            "Erro feio da marca.",
            "Achei bem mal feito.",
            "Campanha sem sentido."
        ]

    else:
        comentarios = [
            "Amei essa campanha, ficou incrível!",
            "Não gostei, propaganda enganosa.",
            "Achei ok, nada demais.",
            "Produto péssimo, não compro mais.",
            "Sensacional, estão de parabéns 👏👏",
            "Esperava mais pelo preço.",
            "Muito bom, gostei bastante da proposta.",
            "Não recomendo, achei fraco.",
            "Excelente publicação, bem feita.",
            "Horrível, comunicação confusa."
        ]

    return pd.DataFrame({"comentario": comentarios})


def consolidar_multiplos_links(lista_links):
    dfs = [simular_comentarios_por_link(link) for link in lista_links]
    if not dfs:
        return pd.DataFrame(columns=["comentario"])
    return pd.concat(dfs, ignore_index=True)


def gerar_nome_usuario_fake(indice, sentimento):
    positivos = [
        "larissamartins", "gustavo.souza", "marianacosta", "joaovictor.r"
    ]
    negativos = [
        "rodrigues83", "suellen.moraes", "felipe_santos", "bia.almeida"
    ]
    base = positivos if sentimento == "positivo" else negativos
    return base[indice % len(base)]


def gerar_curtidas_fake(indice, sentimento):
    if sentimento == "positivo":
        base = [28, 15, 31, 12]
    else:
        base = [3, 2, 4, 1]
    return base[indice % len(base)]


def gerar_avatar_iniciais(usuario):
    partes = re.split(r"[._\\-]+", usuario)
    iniciais = "".join([p[0] for p in partes if p])[:2]
    return iniciais.upper() if iniciais else "US"


def converter_imagem_base64(caminho_imagem):
    try:
        with open(caminho_imagem, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception:
        return ""


def obter_avatar_path_fake(indice, sentimento):
    if sentimento == "positivo":
        nomes = ["mulher_1.png", "mulher_2.png", "mulher_3.png", "mulher_4.png"]
    else:
        nomes = ["homem_1.png", "homem_2.png", "homem_3.png", "homem_4.png"]

    avatar_dir = Path(__file__).resolve().parent / "avatars"
    avatar_path = avatar_dir / nomes[indice % len(nomes)]
    return avatar_path if avatar_path.exists() else None


def _desenhar_card_comentario_png(
    ax,
    x,
    y,
    largura,
    altura,
    usuario,
    comentario,
    likes,
    avatar_path=None,
    negativo=False
):
    card = FancyBboxPatch(
        (x, y - altura),
        largura,
        altura,
        boxstyle="round,pad=0.006,rounding_size=0.01",
        linewidth=1,
        edgecolor="#E9E9E9",
        facecolor="#FFFFFF"
    )
    ax.add_patch(card)

    avatar_centro_x = x + 0.03
    avatar_centro_y = y - 0.04
    avatar_raio = 0.018

    avatar_desenhado = False
    if avatar_path is not None:
        try:
            avatar_img = plt.imread(avatar_path)
            avatar_plot = ax.imshow(
                avatar_img,
                extent=(
                    avatar_centro_x - avatar_raio,
                    avatar_centro_x + avatar_raio,
                    avatar_centro_y - avatar_raio,
                    avatar_centro_y + avatar_raio
                ),
                zorder=3
            )
            clip_circle = Circle((avatar_centro_x, avatar_centro_y), avatar_raio, transform=ax.transData)
            avatar_plot.set_clip_path(clip_circle)
            avatar_desenhado = True
        except Exception:
            avatar_desenhado = False

    if not avatar_desenhado:
        avatar = Circle(
            (avatar_centro_x, avatar_centro_y),
            radius=avatar_raio,
            facecolor="#D8D8D8" if negativo else "#D9D9D9",
            edgecolor="#B9B9B9" if negativo else "#BFBFBF",
            linewidth=1.4
        )
        ax.add_patch(avatar)
        ax.text(
            avatar_centro_x,
            avatar_centro_y,
            gerar_avatar_iniciais(usuario),
            ha="center",
            va="center",
            fontsize=8.2,
            color="#3A3A3A",
            fontweight="bold"
        )

    avatar_borda = Circle(
        (avatar_centro_x, avatar_centro_y),
        radius=avatar_raio,
        facecolor=(0, 0, 0, 0),
        edgecolor="#B9B9B9" if negativo else "#BFBFBF",
        linewidth=1.2,
        zorder=4
    )
    ax.add_patch(avatar_borda)

    texto_x = x + 0.058
    ax.text(
        texto_x,
        y - 0.02,
        usuario,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#222222",
        fontweight="bold"
    )

    comentario_wrap = textwrap.fill(comentario, width=35)
    ax.text(
        texto_x,
        y - 0.045,
        comentario_wrap,
        ha="left",
        va="top",
        fontsize=8,
        color="#444444",
        linespacing=1.3
    )

    ax.text(
        texto_x,
        y - altura + 0.018,
        f"\u2661 {likes}",
        ha="left",
        va="bottom",
        fontsize=7.5,
        color="#7A7A7A"
    )


def gerar_texto_relatorio_fhits(df, qtd_posts, plataforma):
    total = len(df)
    positivos = len(df[df["sentimento"] == "Positivo/Neutro"])
    negativos = len(df[df["sentimento"] == "Negativo"])

    pct_pos = (positivos / total * 100) if total else 0
    pct_neg = (negativos / total * 100) if total else 0

    causas = []
    termos_map = {
        "atrasos no prazo de entrega": ["atraso", "demora", "demorou", "entrega", "prazo"],
        "problemas com cobranças indevidas": ["cobrança", "cobranca", "cobrado", "indevida", "cobranças"],
        "questões de qualidade": ["ruim", "péssimo", "horrível", "baixa qualidade", "qualidade"],
        "problemas de atendimento": ["atendimento", "suporte", "resposta"],
    }

    comentarios_neg = df[df["sentimento"] == "Negativo"]["comentario"].astype(str).str.lower()
    for causa, termos in termos_map.items():
        for termo in termos:
            if comentarios_neg.str.contains(termo, regex=False).any():
                causas.append(causa)
                break

    if causas:
        if len(causas) == 1:
            texto_causas = causas[0]
        elif len(causas) == 2:
            texto_causas = f"{causas[0]} e {causas[1]}"
        else:
            texto_causas = ", ".join(causas[:-1]) + f" e {causas[-1]}"
    else:
        texto_causas = "pontos específicos identificados na leitura qualitativa"

    if negativos == 0:
        texto = f"""Ao todo foram entregues <span class="fhits-big">{qtd_posts}</span> conteúdos no feed da plataforma <b>{plataforma}</b>, com um total de <span class="fhits-big">{total}</span> comentários.<br><br>Realizamos uma análise de sentimento e constatamos <span class="fhits-green">{pct_pos:.0f}%</span> dos comentários foram positivos ou neutros.<br><br>Destacamos ainda, que o público respondeu positivamente às dicas e temas apresentados e demonstrou afeição positiva pela marca e seus produtos."""
    else:
        texto = f"""Ao todo foram entregues <span class="fhits-big">{qtd_posts}</span> conteúdos no feed da plataforma <b>{plataforma}</b>, com um total de <span class="fhits-big">{total}</span> comentários.<br><br>Realizamos uma análise de sentimento e constatamos <span class="fhits-green">{pct_pos:.0f}%</span> dos comentários foram positivos ou neutros e <span class="fhits-red">{pct_neg:.0f}%</span> foram considerados negativos.<br><br>Os <span class="fhits-red">{negativos}</span> comentários negativos foram devido a {texto_causas}.<br><br>Destacamos ainda, que o público respondeu positivamente às dicas e temas apresentados e demonstrou afeição positiva pela marca e seus produtos."""

    return texto


def gerar_png_relatorio_fhits(resultado_df, qtd_links, plataforma="Instagram"):
    total = len(resultado_df)
    positivos = len(resultado_df[resultado_df["sentimento"] == "Positivo/Neutro"])
    negativos = len(resultado_df[resultado_df["sentimento"] == "Negativo"])

    top_negativos_df = preparar_top_negativos(resultado_df, quantidade_top=4)
    top_positivos_df = preparar_top_positivos(resultado_df, quantidade_top=4)

    fig = plt.figure(figsize=(16, 9), dpi=150, facecolor="#FFFFFF")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    pagina = FancyBboxPatch(
        (0.03, 0.04),
        0.94,
        0.92,
        boxstyle="round,pad=0.0,rounding_size=0.005",
        linewidth=1.0,
        edgecolor="#E3DED4",
        facecolor="#FFFFFF"
    )
    ax.add_patch(pagina)

    ax.text(
        0.06,
        0.92,
        "SENTIMENTACAO",
        ha="left",
        va="top",
        fontsize=26,
        fontweight="light",
        color="#111111"
    )

    texto_resumo_html = gerar_texto_relatorio_fhits(
        df=resultado_df,
        qtd_posts=qtd_links,
        plataforma=plataforma
    )
    texto_resumo = re.sub(r"<[^>]+>", "", texto_resumo_html).replace("<br><br>", "\n\n")
    ax.text(
        0.06,
        0.84,
        texto_resumo,
        ha="left",
        va="top",
        fontsize=11.4,
        color="#1F1F1F",
        linespacing=1.55
    )

    pie_ax = fig.add_axes([0.07, 0.22, 0.26, 0.36])
    pie_ax.set_facecolor("#FFFFFF")
    if positivos + negativos == 0:
        valores = [1]
        cores = ["#D9D9D9"]
        labels = ["Sem dados"]
        autopct = ""
    else:
        valores = [positivos, negativos]
        cores = ["#34C759", "#EF3F3F"]
        labels = ["Positivos / Neutros", "Negativos"]
        autopct = "%1.1f%%"

    wedges, _, _ = pie_ax.pie(
        valores,
        labels=None,
        colors=cores,
        autopct=autopct,
        startangle=90,
        counterclock=False,
        textprops=dict(color="white", fontsize=10.5, fontweight="bold"),
        wedgeprops=dict(edgecolor="white", linewidth=1.2)
    )
    pie_ax.axis("equal")
    pie_ax.legend(
        wedges,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=2,
        frameon=False,
        fontsize=8.6,
        handlelength=1.0,
        handletextpad=0.4,
        columnspacing=1.0
    )

    total_sentimentos = positivos + negativos
    pct_pos = (positivos / total_sentimentos * 100) if total_sentimentos else 0
    pct_neg = (negativos / total_sentimentos * 100) if total_sentimentos else 0
    ax.text(0.10, 0.155, "● Positivos / Neutros", ha="left", va="center", fontsize=11, color="#34C759", fontweight="bold")
    ax.text(0.10, 0.125, f"{positivos} ({pct_pos:.1f}%)", ha="left", va="center", fontsize=16, color="#34C759", fontweight="bold")
    ax.text(0.24, 0.155, "● Negativos", ha="left", va="center", fontsize=11, color="#EF3F3F", fontweight="bold")
    ax.text(0.24, 0.125, f"{negativos} ({pct_neg:.1f}%)", ha="left", va="center", fontsize=16, color="#EF3F3F", fontweight="bold")

    ax.text(
        0.38,
        0.66,
        "EXEMPLOS DE COMENTARIOS POSITIVOS",
        ha="left",
        va="center",
        fontsize=9.5,
        color="#2F8F44",
        fontweight="bold"
    )
    ax.text(
        0.67,
        0.66,
        "EXEMPLOS DE COMENTARIOS NEGATIVOS",
        ha="left",
        va="center",
        fontsize=9.5,
        color="#EF3F3F",
        fontweight="bold"
    )

    card_largura = 0.26
    card_altura = 0.108
    espacamento = 0.02
    y_inicial = 0.63

    if top_positivos_df.empty:
        top_positivos_df = pd.DataFrame([{"comentario": "Nenhum comentario positivo/neutro encontrado."}])

    if top_negativos_df.empty:
        top_negativos_df = pd.DataFrame([{"comentario": "Nenhum comentario negativo encontrado."}])

    for i in range(4):
        pos_y = y_inicial - i * (card_altura + espacamento)

        if i < len(top_positivos_df):
            comentario_pos = top_positivos_df.iloc[i]["comentario"]
            user_pos = gerar_nome_usuario_fake(i, "positivo")
            likes_pos = gerar_curtidas_fake(i, "positivo")
            avatar_pos = obter_avatar_path_fake(i, "positivo")
            _desenhar_card_comentario_png(
                ax=ax,
                x=0.38,
                y=pos_y,
                largura=card_largura,
                altura=card_altura,
                usuario=user_pos,
                comentario=str(comentario_pos),
                likes=likes_pos,
                avatar_path=avatar_pos,
                negativo=False
            )

        if i < len(top_negativos_df):
            comentario_neg = top_negativos_df.iloc[i]["comentario"]
            user_neg = gerar_nome_usuario_fake(i, "negativo")
            likes_neg = gerar_curtidas_fake(i, "negativo")
            avatar_neg = obter_avatar_path_fake(i, "negativo")
            _desenhar_card_comentario_png(
                ax=ax,
                x=0.67,
                y=pos_y,
                largura=card_largura,
                altura=card_altura,
                usuario=user_neg,
                comentario=str(comentario_neg),
                likes=likes_neg,
                avatar_path=avatar_neg,
                negativo=True
            )

    ax.text(
        0.92,
        0.08,
        "F",
        ha="right",
        va="center",
        fontsize=24,
        color="#1F1F1F",
        fontweight="bold"
    )
    ax.text(
        0.923,
        0.08,
        "\u2605",
        ha="left",
        va="center",
        fontsize=15,
        color="#B08D57",
        fontweight="bold"
    )
    ax.text(
        0.94,
        0.08,
        "hits",
        ha="left",
        va="center",
        fontsize=24,
        color="#1F1F1F",
        fontweight="bold"
    )

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def renderizar_dashboard_cliente(resultado_df, tempo_total, tempo_medio, quantidade_top_negativos=5):
    total = len(resultado_df)
    positivos = len(resultado_df[resultado_df["sentimento"] == "Positivo/Neutro"])
    negativos = len(resultado_df[resultado_df["sentimento"] == "Negativo"])

    percentual_positivo = (positivos / total) * 100 if total > 0 else 0
    percentual_negativo = (negativos / total) * 100 if total > 0 else 0

    resumo = gerar_resumo(total, positivos, negativos, percentual_negativo)
    top_negativos_df = preparar_top_negativos(resultado_df, quantidade_top=quantidade_top_negativos)

    csv_exportacao = resultado_df[["comentario", "label_modelo", "sentimento"]].to_csv(
        index=False
    ).encode("utf-8-sig")

    c1, c2, c3, c4 = st.columns(4, gap="medium")

    with c1:
        st.markdown(f"""
        <div class="card-box">
            <div class="card-icon">💬</div>
            <div class="card-title">Total de comentários</div>
            <div class="card-number">{total}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="card-box">
            <div class="card-icon">🟢</div>
            <div class="card-title">Positivo/Neutro</div>
            <div class="card-number green-text">{positivos}</div>
            <div class="card-sub green-text">{percentual_positivo:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="card-box">
            <div class="card-icon">🔴</div>
            <div class="card-title">Negativo</div>
            <div class="card-number red-text">{negativos}</div>
            <div class="card-sub red-text">{percentual_negativo:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="card-box">
            <div class="card-icon">⏱️</div>
            <div class="card-title">Tempo médio</div>
            <div class="card-number">{tempo_medio:.2f}s</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    g1, g2 = st.columns([1, 1.7], gap="medium")

    with g1:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Distribuição dos sentimentos</div>', unsafe_allow_html=True)
        fig = gerar_grafico_donut(positivos, negativos)
        st.pyplot(fig)
        st.markdown(
            f"""
            <div style="margin-top:8px; font-size:14px; line-height:1.8;">
                <span style="color:#67B26F; font-weight:700;">● Positivo/Neutro</span>: {positivos} ({percentual_positivo:.1f}%)<br>
                <span style="color:#E25555; font-weight:700;">● Negativo</span>: {negativos} ({percentual_negativo:.1f}%)
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with g2:
        st.markdown('<div class="summary-box">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Resumo executivo</div>', unsafe_allow_html=True)
        st.markdown(resumo, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    b1, b2 = st.columns([2.2, 1], gap="medium")

    with b1:
        st.markdown('<div class="side-box">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Top comentários negativos</div>', unsafe_allow_html=True)

        if top_negativos_df.empty:
            st.success("Nenhum comentário negativo encontrado.")
        else:
            for i, row in top_negativos_df.iterrows():
                st.markdown(
                    f"""
                    <div class="neg-item">
                        <div><b>#{i+1}</b></div>
                        <div><b>Comentário:</b> {row['comentario']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        st.markdown('</div>', unsafe_allow_html=True)

    with b2:
        st.markdown('<div class="side-box">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Exportação</div>', unsafe_allow_html=True)
        st.write("Baixe a base completa classificada para uso interno.")
        st.download_button(
            label="⬇️ Exportar CSV",
            data=csv_exportacao,
            file_name="resultado_sentimentacao.csv",
            mime="text/csv",
            key="download_cliente"
        )
        st.markdown('</div>', unsafe_allow_html=True)


def renderizar_relatorio_fhits(resultado_df, qtd_links, plataforma="Instagram"):
    total = len(resultado_df)
    positivos = len(resultado_df[resultado_df["sentimento"] == "Positivo/Neutro"])
    negativos = len(resultado_df[resultado_df["sentimento"] == "Negativo"])

    top_negativos_df = preparar_top_negativos(resultado_df, quantidade_top=4)
    top_positivos_df = preparar_top_positivos(resultado_df, quantidade_top=4)

    texto_topo = gerar_texto_relatorio_fhits(
        df=resultado_df,
        qtd_posts=qtd_links,
        plataforma=plataforma
    )

    st.markdown('<div class="fhits-canvas-bg">', unsafe_allow_html=True)
    st.markdown('<div class="fhits-page">', unsafe_allow_html=True)

    st.markdown('<div class="fhits-title">SENTIMENTAÇÃO</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="fhits-text">{texto_topo}</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    main_left, main_right = st.columns([1.05, 2.15], gap="large")

    with main_left:
        fig_relatorio = gerar_grafico_pizza_relatorio(positivos, negativos)
        st.pyplot(fig_relatorio, use_container_width=True)
        total_sentimentos = positivos + negativos
        pct_pos = (positivos / total_sentimentos * 100) if total_sentimentos else 0
        pct_neg = (negativos / total_sentimentos * 100) if total_sentimentos else 0
        st.markdown(
            f"""
            <div style="margin-top: -6px; font-size: 18px; line-height: 1.9; text-align: left; padding-left: 28px;">
                <span style="color:#34C759; font-weight:700;">● Positivos / Neutros</span><br>
                <span style="color:#34C759; font-family: Impact, Haettenschweiler, 'Arial Narrow Bold', sans-serif; font-size: 30px;">
                    {positivos} ({pct_pos:.1f}%)
                </span>
                <br><br>
                <span style="color:#EF3F3F; font-weight:700;">● Negativos</span><br>
                <span style="color:#EF3F3F; font-family: Impact, Haettenschweiler, 'Arial Narrow Bold', sans-serif; font-size: 30px;">
                    {negativos} ({pct_neg:.1f}%)
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    with main_right:
        pos_col, neg_col = st.columns(2, gap="medium")

        with pos_col:
            st.markdown('<div class="fhits-comments-title-green">Exemplos de comentários positivos</div>', unsafe_allow_html=True)
            if top_positivos_df.empty:
                st.markdown(
                    '<div class="fhits-comment-card"><div class="fhits-comment-text">Nenhum comentário positivo/neutro encontrado.</div></div>',
                    unsafe_allow_html=True
                )
            else:
                for i, row in top_positivos_df.iterrows():
                    user = gerar_nome_usuario_fake(i, "positivo")
                    likes = gerar_curtidas_fake(i, "positivo")
                    avatar_path = obter_avatar_path_fake(i, "positivo")
                    avatar_base64 = converter_imagem_base64(str(avatar_path)) if avatar_path else ""
                    avatar_html = f'<img src="data:image/png;base64,{avatar_base64}" class="fhits-avatar-img">' if avatar_base64 else f'<div class="fhits-avatar">{gerar_avatar_iniciais(user)}</div>'
                    st.markdown(
                        f"""
                        <div class="fhits-comment-card">
                            <div class="fhits-comment-row">
                                {avatar_html}
                                <div class="fhits-comment-body">
                                    <div class="fhits-comment-user">{user}</div>
                                    <div class="fhits-comment-text">{row['comentario']}</div>
                                    <div class="fhits-comment-like">♡ {likes}</div>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        with neg_col:
            st.markdown('<div class="fhits-comments-title-red">Exemplos de comentários negativos</div>', unsafe_allow_html=True)
            if top_negativos_df.empty:
                st.markdown(
                    '<div class="fhits-comment-card"><div class="fhits-comment-text">Nenhum comentário negativo encontrado.</div></div>',
                    unsafe_allow_html=True
                )
            else:
                for i, row in top_negativos_df.iterrows():
                    user = gerar_nome_usuario_fake(i, "negativo")
                    likes = gerar_curtidas_fake(i, "negativo")
                    avatar_path = obter_avatar_path_fake(i, "negativo")
                    avatar_base64 = converter_imagem_base64(str(avatar_path)) if avatar_path else ""
                    avatar_html = f'<img src="data:image/png;base64,{avatar_base64}" class="fhits-avatar-img">' if avatar_base64 else f'<div class="fhits-avatar neg">{gerar_avatar_iniciais(user)}</div>'
                    st.markdown(
                        f"""
                        <div class="fhits-comment-card">
                            <div class="fhits-comment-row">
                                {avatar_html}
                                <div class="fhits-comment-body">
                                    <div class="fhits-comment-user">{user}</div>
                                    <div class="fhits-comment-text">{row['comentario']}</div>
                                    <div class="fhits-comment-like">♡ {likes}</div>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    st.markdown(
        '<div class="fhits-logo-bottom">F<span class="fhits-logo-bottom-star">★</span>hits</div>',
        unsafe_allow_html=True
    )

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    csv_exportacao = resultado_df[["comentario", "label_modelo", "sentimento"]].to_csv(
        index=False
    ).encode("utf-8-sig")
    png_exportacao = gerar_png_relatorio_fhits(
        resultado_df=resultado_df,
        qtd_links=qtd_links,
        plataforma=plataforma
    )

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    st.download_button(
        label="⬇️ Exportar CSV consolidado",
        data=csv_exportacao,
        file_name="resultado_sentimentacao_fhits.csv",
        mime="text/csv",
        key="download_fhits_csv"
    )

    st.download_button(
        label="Exportar relatorio PNG",
        data=png_exportacao,
        file_name=f"relatorio_sentimentacao_fhits_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
        mime="image/png",
        key="download_fhits_png"
    )


# =========================================================
# CABEÇALHO
# =========================================================
st.markdown("""
<div class="header-wrap">
    <div class="brand-line">
        <div class="logo-fhits">F<span class="logo-star">★</span>hits</div>
        <div class="logo-divider"></div>
        <div class="title-area">
            <div class="main-title">SENTIMENTAÇÃO</div>
            <div class="sub-title">Análise de sentimento de comentários</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# ABAS
# =========================================================
aba_cliente, aba_fhits = st.tabs(["Cliente", "FHITS Interno"])

# =========================================================
# ABA CLIENTE
# =========================================================
with aba_cliente:
    modo_entrada = st.radio(
        "Modo de entrada",
        options=["CSV", "Link"],
        horizontal=True,
        key="modo_entrada_cliente"
    )

    col_left, col_right = st.columns([1.05, 3.2], gap="large")

    with col_left:
        st.markdown('<div class="side-box">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Configurações</div>', unsafe_allow_html=True)

        if modo_entrada == "CSV":
            st.markdown('<div class="mode-badge">Modo CSV</div>', unsafe_allow_html=True)
            arquivo = st.file_uploader("Upload do CSV", type=["csv"], key="upload_csv_cliente")
            quantidade_top_negativos = st.selectbox(
                "Qtd. top negativos",
                options=[3, 5, 7, 10],
                index=1,
                key="top_neg_csv_cliente"
            )
            iniciar_csv = st.button("🚀 Iniciar análise CSV", key="btn_csv_cliente")
            iniciar_link = False
            url_publicacao = ""
        else:
            st.markdown('<div class="mode-badge">Modo Link</div>', unsafe_allow_html=True)
            st.selectbox(
                "Plataforma",
                options=["Instagram", "TikTok", "YouTube"],
                key="plataforma_cliente"
            )
            url_publicacao = st.text_input(
                "Cole a URL da publicação",
                placeholder="https://www.instagram.com/p/xxxxxxxx/",
                key="url_cliente"
            )
            quantidade_top_negativos = st.selectbox(
                "Qtd. top negativos",
                options=[3, 5, 7, 10],
                index=1,
                key="top_neg_link_cliente"
            )
            iniciar_link = st.button("🔗 Analisar publicação", key="btn_link_cliente")
            iniciar_csv = False
            arquivo = None

        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        if modo_entrada == "CSV":
            if iniciar_csv:
                if arquivo is None:
                    st.warning("Envie um CSV antes de iniciar a análise.")
                    st.stop()

                try:
                    df = pd.read_csv(arquivo)
                except UnicodeDecodeError:
                    df = pd.read_csv(arquivo, encoding="latin-1")
                except pd.errors.EmptyDataError:
                    st.error("O CSV está vazio.")
                    st.stop()
                except Exception as e:
                    st.error(f"Erro ao ler o arquivo: {e}")
                    st.stop()

                if df.empty:
                    st.error("O CSV está vazio.")
                    st.stop()

                if "comentario" not in df.columns:
                    st.error("A coluna 'comentario' não foi encontrada no CSV.")
                    st.stop()

                with st.spinner("Carregando modelo..."):
                    modelo = carregar_modelo()

                resultado_df, tempo_total, tempo_medio = analisar_comentarios(df, "comentario", modelo)

                if resultado_df.empty:
                    st.warning("Nenhum comentário válido foi encontrado.")
                    st.stop()

                renderizar_dashboard_cliente(
                    resultado_df=resultado_df,
                    tempo_total=tempo_total,
                    tempo_medio=tempo_medio,
                    quantidade_top_negativos=quantidade_top_negativos
                )
            else:
                st.info("Envie um CSV e clique em **Iniciar análise CSV**.")
        else:
            if iniciar_link:
                if not url_publicacao.strip():
                    st.warning("Cole uma URL da publicação para continuar.")
                    st.stop()

                if not validar_url_publicacao(url_publicacao):
                    st.error("A URL informada parece inválida. Verifique e tente novamente.")
                    st.stop()

                with st.spinner("Consultando o backend e analisando a publicação..."):
                    inicio_backend = time.time()
                    resultado_df, resposta_backend = analisar_link_backend(
                        url=url_publicacao,
                        plataforma=st.session_state.get("plataforma_cliente", "Instagram")
                    )
                    tempo_total = time.time() - inicio_backend

                if resultado_df is None:
                    st.stop()

                if resultado_df.empty:
                    st.warning("Nenhum comentário válido foi encontrado.")
                    st.stop()

                tempo_medio = tempo_total / len(resultado_df) if len(resultado_df) > 0 else 0

                renderizar_dashboard_cliente(
                    resultado_df=resultado_df,
                    tempo_total=tempo_total,
                    tempo_medio=tempo_medio,
                    quantidade_top_negativos=quantidade_top_negativos
                )
            else:
                st.info("Cole a URL da publicação e clique em **Analisar publicação**.")

# =========================================================
# ABA FHITS INTERNO
# =========================================================
with aba_fhits:
    st.markdown('<div class="fhits-controls">', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 2.6, 1])

    with c1:
        plataforma_fhits = st.selectbox(
            "Plataforma",
            options=["Instagram", "TikTok", "YouTube"],
            key="plataforma_fhits"
        )

    with c2:
        links_multiplos = st.text_area(
            "Cole os links das publicações (1 por linha)",
            placeholder="https://www.instagram.com/p/top-campaign\nhttps://www.instagram.com/p/fail-campaign\nhttps://www.instagram.com/p/campanha-normal",
            height=110,
            key="links_multiplos_fhits"
        )

    with c3:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        iniciar_fhits = st.button("📄 Gerar relatório FHITS", key="btn_fhits_relatorio")

    st.markdown('</div>', unsafe_allow_html=True)

    if iniciar_fhits:
        lista_links = [linha.strip() for linha in links_multiplos.split("\n") if linha.strip()]

        if not lista_links:
            st.warning("Cole pelo menos um link para continuar.")
            st.stop()

        urls_invalidas = [url for url in lista_links if not validar_url_publicacao(url)]
        if urls_invalidas:
            st.error("Uma ou mais URLs parecem inválidas. Revise os links informados.")
            st.stop()

        with st.spinner("Consultando o backend e consolidando as publicações..."):
            resultado_df_fhits, respostas_backend = analisar_multiplos_links_backend(
                lista_links=lista_links,
                plataforma=plataforma_fhits
            )

        if resultado_df_fhits is None:
            st.stop()

        if resultado_df_fhits.empty:
            st.warning("Nenhum comentário válido foi encontrado.")
            st.stop()

        renderizar_relatorio_fhits(
            resultado_df=resultado_df_fhits,
            qtd_links=len(lista_links),
            plataforma=plataforma_fhits
        )
    else:
        st.info("Cole vários links, um por linha, e clique em **Gerar relatório FHITS**.")
