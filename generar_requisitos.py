"""Genera requisitos.js: para cada fundamento, qué fundamentos necesita antes y de
cuáles es requisito.

Los requisitos salen del contenido, no de una lista escrita a mano: si la página B
enlaza a la página A y A va ANTES en el orden de estudio, A es requisito de B. Los
enlaces hacia páginas posteriores ("lo verás más adelante") no cuentan. El orden de
estudio es el de index.html (módulo, lección y número de fundamento).

Uso, desde esta carpeta, cada vez que se agregan o cambian fundamentos:

    python generar_requisitos.py

También agrega a cada fundamento la línea <script src="requisitos.js" defer>, si
todavía no la tiene.
"""

import json
import re
from html import unescape
from pathlib import Path

CARPETA = Path(__file__).resolve().parent
SCRIPT_TAG = '<script src="requisitos.js" defer></script>'


def texto(html: str) -> str:
    return unescape(re.sub(r"<[^>]+>", "", html)).strip()


def leer_orden() -> tuple[list[str], dict, dict]:
    """Fundamentos en orden de estudio, sus datos y los datos de cada lección."""
    index = (CARPETA / "index.html").read_text(encoding="utf-8")
    orden, fundamentos, lecciones = [], {}, {}
    tarjetas = re.findall(r'<div class="lesson-card.*?(?=<div class="lesson-card|</section>)', index, re.S)
    for tarjeta in tarjetas:
        num = re.search(r"LECCIÓN ([\d.]+)", tarjeta).group(1)
        lecciones[num] = texto(re.search(r'lesson-title">(.*?)</h3>', tarjeta, re.S).group(1))
        for archivo, nombre in re.findall(r'<a href="(\d\d_fundamento_[^"]+\.html)">(.*?)</a>', tarjeta, re.S):
            nombre = re.sub(r"^\d+\.\s*", "", texto(nombre))
            nombre = re.sub(r"\s*\(complemento\)\s*$", "", nombre)
            orden.append(archivo)
            fundamentos[archivo] = {"t": nombre, "l": num}
    return orden, fundamentos, lecciones


def enlaces_del_contenido(archivo: str) -> list[str]:
    """Fundamentos que la página enlaza en su contenido (sin el pie ni el recuadro)."""
    html = (CARPETA / archivo).read_text(encoding="utf-8")
    cuerpo = html.split("</style>", 1)[-1].split("<footer", 1)[0]
    # Los enlaces de navegación ("Siguiente: …", "Lección anterior: …") no son requisitos
    cuerpo = re.sub(r"(Siguiente(\s+lección)?|Lección anterior)\s*:\s*<a\b[^>]*>.*?</a>", "", cuerpo, flags=re.S)
    vistos = []
    for destino in re.findall(r'href="(\d\d_fundamento_[^"#]+\.html)', cuerpo):
        if destino != archivo and destino not in vistos:
            vistos.append(destino)
    return vistos


def agregar_script(archivo: str) -> bool:
    ruta = CARPETA / archivo
    html = ruta.read_text(encoding="utf-8")
    if SCRIPT_TAG in html:
        return False
    # Conserva los saltos de línea del archivo (algunos tienen CRLF)
    salto = "\r\n" if "\r\n" in html else "\n"
    html = html.replace("</body>", SCRIPT_TAG + salto + "</body>", 1)
    ruta.write_bytes(html.encode("utf-8"))
    return True


def main() -> None:
    orden, fundamentos, lecciones = leer_orden()
    posicion = {archivo: i for i, archivo in enumerate(orden)}
    for archivo in orden:
        fundamentos[archivo]["r"] = [
            destino for destino in enlaces_del_contenido(archivo)
            if destino in posicion and posicion[destino] < posicion[archivo]
        ]
    datos = {"lecciones": lecciones, "orden": orden, "fundamentos": fundamentos}
    plantilla = (CARPETA / "requisitos_plantilla.js").read_text(encoding="utf-8")
    salida = plantilla.replace("/*DATOS*/null", json.dumps(datos, ensure_ascii=False, separators=(",", ":")))
    (CARPETA / "requisitos.js").write_text(salida, encoding="utf-8", newline="\n")

    agregados = sum(agregar_script(archivo) for archivo in orden)
    total = sum(len(f["r"]) for f in fundamentos.values())
    sin_nada = [a for a in orden if not fundamentos[a]["r"] and not any(a in f["r"] for f in fundamentos.values())]
    print(f"{len(orden)} fundamentos, {total} relaciones de requisito, script agregado a {agregados} páginas")
    if sin_nada:
        print(f"{len(sin_nada)} fundamentos sin requisitos ni dependientes: {', '.join(sin_nada)}")


if __name__ == "__main__":
    main()
