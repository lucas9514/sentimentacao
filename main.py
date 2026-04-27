print("Sentimentação iniciada com sucesso 🚀")
from textblob import TextBlob

comentarios = [
    "I love this product",
    "This is amazing",
    "I hate this experience",
    "This is terrible",
    "It's okay",
    "Nothing special"
]

positivos = 0
negativos = 0
neutros = 0

for comentario in comentarios:
    analise = TextBlob(comentario)
    polaridade = analise.sentiment.polarity

    if polaridade > 0:
        positivos += 1
    elif polaridade < 0:
        negativos += 1
    else:
        neutros += 1

print("Resultado da análise de sentimento:")
print("Positivos:", positivos)
print("Negativos:", negativos)
print("Neutros:", neutros)

