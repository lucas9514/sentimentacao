from transformers import pipeline

print("🔍 Carregando modelo de sentimento em português...")

analisador = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

comentarios = [
    "Amei essa campanha, ficou incrível!",
    "Não gostei, propaganda enganosa.",
    "Achei ok, nada demais.",
    "Produto péssimo, não compro mais.",
    "Sensacional, estão de parabéns 👏👏",
    "Esperava mais pelo preço."
]

resultado_final = {
    "positivo": 0,
    "neutro": 0,
    "negativo": 0
}

print("\n📊 Analisando comentários...\n")

for comentario in comentarios:
    resultado = analisador(comentario)[0]
    estrelas = int(resultado["label"][0])

    if estrelas <= 2:
        resultado_final["negativo"] += 1
    elif estrelas == 3:
        resultado_final["neutro"] += 1
    else:
        resultado_final["positivo"] += 1

    print(f"Comentário: {comentario}")
    print(f"Sentimento: {estrelas} estrelas | Confiança: {resultado['score']:.2f}")
    print("-" * 50)

print("\n📈 RESUMO FINAL")
print(f"Positivos: {resultado_final['positivo']}")
print(f"Neutros: {resultado_final['neutro']}")
print(f"Negativos: {resultado_final['negativo']}")