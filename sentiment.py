import pandas as pd
from transformers import pipeline
import time

ARQUIVO_ENTRADA = "comentarios_reais.csv"
ARQUIVO_RESULTADO = "resultado_real.csv"
ARQUIVO_RELATORIO = "relatorio_sentimentacao.txt"

print("🔍 Carregando modelo de sentimento em português...")

sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

# Ler CSV real
df = pd.read_csv(ARQUIVO_ENTRADA, encoding="utf-8")

# Limpar comentários vazios
df["comentario"] = df["comentario"].astype(str).str.strip()
df = df[df["comentario"] != ""]
df = df[df["comentario"].str.lower() != "nan"]

comentarios = df["comentario"].tolist()
total_comentarios = len(comentarios)

print(f"\n📊 Total de comentários reais: {total_comentarios}\n")

inicio = time.time()

# Processamento em lote
resultados_modelo = sentiment_analyzer(comentarios, batch_size=32)

resultados = []

positivos = 0
neutros = 0
negativos = 0

for comentario, resultado in zip(comentarios, resultados_modelo):
    estrelas = int(resultado["label"][0])
    confianca = round(resultado["score"], 2)

    if estrelas >= 4:
        classificacao = "Positivo"
        positivos += 1
    elif estrelas == 3:
        classificacao = "Neutro"
        neutros += 1
    else:
        classificacao = "Negativo"
        negativos += 1

    resultados.append({
        "comentario": comentario,
        "estrelas": estrelas,
        "confianca": confianca,
        "classificacao": classificacao
    })

# Salvar CSV final
df_resultado = pd.DataFrame(resultados)
df_resultado.to_csv(ARQUIVO_RESULTADO, index=False, encoding="utf-8-sig")

fim = time.time()
tempo_total = round(fim - inicio, 2)
tempo_medio = round(tempo_total / total_comentarios, 4) if total_comentarios > 0 else 0

# Percentuais
percentual_positivos = round((positivos / total_comentarios) * 100, 2) if total_comentarios > 0 else 0
percentual_neutros = round((neutros / total_comentarios) * 100, 2) if total_comentarios > 0 else 0
percentual_negativos = round((negativos / total_comentarios) * 100, 2) if total_comentarios > 0 else 0

# Risco
if percentual_negativos > 50:
    risco = "ALTO 🚨"
elif percentual_negativos > 30:
    risco = "MÉDIO ⚠️"
else:
    risco = "BAIXO ✅"

# Resumo executivo
if percentual_positivos > percentual_negativos + 5:
    resumo = "A percepção do público é majoritariamente positiva, indicando boa aceitação da campanha."
elif percentual_negativos > percentual_positivos + 5:
    resumo = "A percepção do público apresenta viés negativo, indicando possíveis pontos de atenção na campanha."
else:
    resumo = "A percepção do público está equilibrada, sem predominância clara de sentimento, indicando recepção mista da campanha."

# Terminal
print("\n📈 RELATÓRIO FINAL")
print("=" * 40)
print(f"Total de comentários analisados: {total_comentarios}")
print(f"Positivos: {positivos} ({percentual_positivos}%)")
print(f"Neutros: {neutros} ({percentual_neutros}%)")
print(f"Negativos: {negativos} ({percentual_negativos}%)")
print(f"\n🚨 Classificação de risco: {risco}")
print(f"\n🧠 Resumo executivo:")
print(resumo)
print(f"\n⏱ Tempo total: {tempo_total} segundos")
print(f"⚡ Tempo médio por comentário: {tempo_medio} segundos")

# Relatório TXT
with open(ARQUIVO_RELATORIO, "w", encoding="utf-8") as arquivo:
    arquivo.write("RELATÓRIO DE SENTIMENTAÇÃO\n")
    arquivo.write("=" * 50 + "\n\n")
    arquivo.write(f"Arquivo analisado: {ARQUIVO_ENTRADA}\n")
    arquivo.write(f"Total de comentários analisados: {total_comentarios}\n\n")

    arquivo.write("DISTRIBUIÇÃO DE SENTIMENTOS\n")
    arquivo.write("-" * 50 + "\n")
    arquivo.write(f"Positivos: {positivos} ({percentual_positivos}%)\n")
    arquivo.write(f"Neutros: {neutros} ({percentual_neutros}%)\n")
    arquivo.write(f"Negativos: {negativos} ({percentual_negativos}%)\n\n")

    arquivo.write(f"CLASSIFICAÇÃO DE RISCO: {risco}\n\n")

    arquivo.write("RESUMO EXECUTIVO\n")
    arquivo.write("-" * 50 + "\n")
    arquivo.write(resumo + "\n\n")

    arquivo.write("PERFORMANCE\n")
    arquivo.write("-" * 50 + "\n")
    arquivo.write(f"Tempo total de execução: {tempo_total} segundos\n")
    arquivo.write(f"Tempo médio por comentário: {tempo_medio} segundos\n")