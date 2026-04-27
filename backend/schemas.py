from pydantic import BaseModel, HttpUrl
from typing import List, Optional


class LinkRequest(BaseModel):
    url: HttpUrl
    plataforma: Optional[str] = None


class MultiLinkRequest(BaseModel):
    urls: List[HttpUrl]
    plataforma: Optional[str] = None


class ComentarioItem(BaseModel):
    comentario: str
    sentimento: str
    label_modelo: str


class LinkResponse(BaseModel):
    url: str
    plataforma: str
    origem: str
    total_comentarios: int
    positivos_neutros: int
    negativos: int
    comentarios: List[ComentarioItem]


class MultiLinkResponse(BaseModel):
    urls: List[str]
    plataforma: str
    origem: str
    total_links: int
    total_comentarios: int
    positivos_neutros: int
    negativos: int
    comentarios: List[ComentarioItem]
