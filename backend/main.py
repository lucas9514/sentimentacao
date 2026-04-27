from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import LinkRequest, LinkResponse, MultiLinkRequest, MultiLinkResponse
from backend.services import analisar_link_simulado, analisar_multiplos_links_simulado

app = FastAPI(
    title="SENTIMENTAÇÃO API",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analisar-link", response_model=LinkResponse)
def analisar_link(payload: LinkRequest):
    return analisar_link_simulado(
        url=str(payload.url),
        plataforma=payload.plataforma
    )


@app.post("/analisar-links", response_model=MultiLinkResponse)
def analisar_links(payload: MultiLinkRequest):
    urls = [str(url) for url in payload.urls]
    return analisar_multiplos_links_simulado(
        urls=urls,
        plataforma=payload.plataforma
    )
