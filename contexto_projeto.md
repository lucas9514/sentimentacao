# PROJETO: SENTIMENTAÇÃO

## Objetivo

Ferramenta interna para análise de sentimento de comentários de campanhas de influenciadores.

## Stack

- Python
- Streamlit
- Transformers (nlptown/bert-base-multilingual-uncased-sentiment)

## Entrada de dados

- CSV com coluna "comentario"
- Simulação via links (futuro: integração real com Instagram/TikTok)

## Classificação

- 1 e 2 estrelas → Negativo
- 3, 4 e 5 estrelas → Positivo/Neutro

## Funcionalidades atuais

- Upload de CSV
- Análise de sentimento
- Dashboard com:
  - Cards
  - Gráfico de pizza
  - Resumo executivo
  - Top comentários negativos
- Aba FHITS Interno:
  - Layout estilo relatório
  - 4 comentários positivos e 4 negativos
  - Avatares locais (pasta /avatars)
  - Texto padrão de relatório
  - Gráfico grande estilo apresentação

## Regras importantes

- NÃO usar score/confiança no output
- Foco em simplicidade para uso interno
- Layout precisa ser copiável para PPT/PDF
- Visual inspirado em relatórios FHITS

## Estrutura de pastas

/avatars

- mulher_1.png
- mulher_2.png
- mulher_3.png
- mulher_4.png
- homem_1.png
- homem_2.png
- homem_3.png
- homem_4.png

## Próximos passos

- Exportar relatório como PNG
- Integrar com links reais
- Usar curtidas como relevância
