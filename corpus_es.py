"""
CORPUS EN ESPANOL PARA EL MODULO DE PLN (lecciones 3.1 a 3.6)
=============================================================

Dos fuentes de texto:
  1. "Don Quijote de la Mancha" (Cervantes, dominio publico), del Proyecto
     Gutenberg. Se descarga una vez a data/ (~2 MB, excluida del repositorio).
  2. RESENAS: 80 resenas cortas escritas para este curso, la mitad positivas
     (etiqueta 1) y la mitad negativas (etiqueta 0), para la leccion 3.2.
"""

import os
import urllib.request

URL_QUIJOTE = "https://www.gutenberg.org/cache/epub/2000/pg2000.txt"


def cargar_quijote(carpeta="data"):
    """Devuelve el texto completo del Quijote, sin la cabecera ni el pie de Gutenberg."""
    os.makedirs(carpeta, exist_ok=True)
    ruta = os.path.join(carpeta, "quijote.txt")
    if not os.path.exists(ruta):
        print("Descargando Don Quijote (Proyecto Gutenberg)...")
        urllib.request.urlretrieve(URL_QUIJOTE, ruta)
    with open(ruta, encoding="utf-8") as f:
        texto = f.read()
    inicio = texto.index("*** START OF")
    inicio = texto.index("\n", inicio) + 1
    fin = texto.index("*** END OF")
    return texto[inicio:fin].strip()


RESENAS = [
    # --- positivas (1) ---
    ("La comida estaba deliciosa y el servicio fue muy amable", 1),
    ("Excelente producto, llego rapido y funciona perfecto", 1),
    ("Me encanto la pelicula, los actores son increibles", 1),
    ("El hotel es comodo, limpio y con una vista hermosa", 1),
    ("Muy buena calidad por el precio, lo recomiendo", 1),
    ("El libro es fascinante, no pude dejar de leerlo", 1),
    ("Atencion rapida y amable, volveria sin dudarlo", 1),
    ("Una experiencia maravillosa de principio a fin", 1),
    ("El telefono tiene una bateria excelente y una camara genial", 1),
    ("Los postres son espectaculares y el cafe muy rico", 1),
    ("Funciona muy bien, facil de usar y muy practico", 1),
    ("El concierto fue increible, la banda sono perfecta", 1),
    ("Me gusto mucho el curso, el profesor explica muy claro", 1),
    ("Gran ambiente, buena musica y precios justos", 1),
    ("El envio fue rapido y el paquete llego en perfecto estado", 1),
    ("Recomiendo este restaurante, todo estaba muy sabroso", 1),
    ("La aplicacion es intuitiva y muy util para organizarme", 1),
    ("Excelente atencion al cliente, resolvieron todo enseguida", 1),
    ("Los zapatos son comodos y de muy buena calidad", 1),
    ("Una serie brillante, con una historia muy bien contada", 1),
    ("El personal fue amable y la habitacion estaba impecable", 1),
    ("Muy contento con la compra, supero mis expectativas", 1),
    ("La pizza estaba exquisita y llego caliente", 1),
    ("Un juego divertido, con graficos hermosos", 1),
    ("El auto es comodo, economico y muy seguro", 1),
    ("Me encanto el lugar, tranquilo y con buena atencion", 1),
    ("La musica del disco es preciosa, lo escucho todos los dias", 1),
    ("Producto excelente, tal como en la descripcion", 1),
    ("La clase de yoga fue relajante y muy bien guiada", 1),
    ("Volveria a comprar, todo funciona de maravilla", 1),
    ("El paisaje es impresionante y el guia muy simpatico", 1),
    ("La cafetera es rapida y el cafe sale muy rico", 1),
    ("Muy buena obra de teatro, actuaciones excelentes", 1),
    ("El vendedor fue honesto y muy atento", 1),
    ("La bicicleta es liviana y muy facil de manejar", 1),
    ("Gran libro, lo recomiendo a todos mis amigos", 1),
    ("El spa fue relajante, una experiencia perfecta", 1),
    ("La tienda tiene precios buenos y mucha variedad", 1),
    ("Los auriculares suenan muy bien y son comodos", 1),
    ("Todo salio perfecto, gracias por la buena atencion", 1),
    # --- negativas (0) ---
    ("La comida estaba fria y el servicio fue muy lento", 0),
    ("Pesimo producto, se rompio a los dos dias", 0),
    ("La pelicula es aburrida y demasiado larga", 0),
    ("El hotel estaba sucio y la habitacion olia mal", 0),
    ("Mala calidad, no vale lo que cuesta", 0),
    ("El libro es confuso y muy aburrido", 0),
    ("Atencion grosera y lenta, no vuelvo mas", 0),
    ("Una experiencia horrible de principio a fin", 0),
    ("El telefono se calienta y la bateria dura muy poco", 0),
    ("Los postres estaban secos y el cafe frio", 0),
    ("No funciona bien, es complicado y poco practico", 0),
    ("El concierto fue un desastre, el sonido era terrible", 0),
    ("No me gusto el curso, el profesor explica muy mal", 0),
    ("Ambiente ruidoso, musica fea y precios abusivos", 0),
    ("El envio tardo semanas y el paquete llego roto", 0),
    ("No recomiendo este restaurante, todo estaba salado", 0),
    ("La aplicacion se cuelga todo el tiempo y es inutil", 0),
    ("Pesima atencion al cliente, nunca respondieron", 0),
    ("Los zapatos son incomodos y se rompieron rapido", 0),
    ("Una serie mala, con una historia sin sentido", 0),
    ("El personal fue grosero y la habitacion estaba sucia", 0),
    ("Muy decepcionado con la compra, una perdida de dinero", 0),
    ("La pizza estaba quemada y llego fria", 0),
    ("Un juego aburrido, con graficos horribles", 0),
    ("El auto consume mucho y se descompone seguido", 0),
    ("No me gusto el lugar, ruidoso y con mala atencion", 0),
    ("La musica del disco es aburrida, no la volvi a escuchar", 0),
    ("Producto defectuoso, nada que ver con la descripcion", 0),
    ("La clase fue estresante y el instructor no explico nada", 0),
    ("No vuelvo a comprar, nada funciona bien", 0),
    ("El paisaje no tiene nada especial y el guia fue antipatico", 0),
    ("La cafetera pierde agua y el cafe sale horrible", 0),
    ("Mala obra de teatro, actuaciones flojas", 0),
    ("El vendedor fue deshonesto y nada atento", 0),
    ("La bicicleta es pesada y dificil de manejar", 0),
    ("Mal libro, no se lo recomiendo a nadie", 0),
    ("El spa fue un desastre, una experiencia terrible", 0),
    ("La tienda es cara y tiene poca variedad", 0),
    ("Los auriculares suenan mal y lastiman las orejas", 0),
    ("Todo salio mal, una pesima atencion", 0),
]
