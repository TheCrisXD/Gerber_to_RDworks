#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ares_a_laser.py  -  Proteus ARES  ->  negativo del cobre para la cortadora laser
=================================================================================

Toma el Gerber (ZIP) que entrega Proteus ARES y genera el NEGATIVO del cobre: un rectangulo
que cubre la placa con "huecos" donde van pistas y pads. Es la zona donde el laser debe
quitar el sticker para que el cloruro ferrico ataque solo ahi.

  * Las PERFORACIONES (archivo Drill/Excellon del ZIP) se dejan como circulos EXPUESTOS dentro
    de cada pad: el acido se come ese punto de cobre y luego el laser puede perforar o marcar
    la entrada del taladro.
  * Convierte cada pista (trazo con grosor) y cada pad en un poligono real, une todo y lo resta
    de un rectangulo con margen: lo que Inkscape no logra con cientos de trazos.
  * Entrega PNG (blanco y negro a escala real, el mas seguro para RDWorks en modo Scan), DXF,
    PLT y SVG. Todo en milimetros, con origen en 0,0.
  * Opcional (el programa pregunta): archivos SOLO de las perforaciones, en el mismo sistema de
    coordenadas, para volver a pasar la placa ya sin cobre y cortar los agujeros con el laser.
  * Tambien lee PDF, SVG y DXF exportados de ARES (opciones secundarias).

Instalacion (una sola vez):
    pip install shapely ezdxf pymupdf svgelements pillow
    (para leer solo Gerber basta: pip install shapely pillow ezdxf)

Uso:
    python ares_a_laser.py "Proyecto - CADCAM.zip"     (ZIP o carpeta de Gerber de ARES)
    python ares_a_laser.py                             (modo asistente: te va preguntando)
    python ares_a_laser.py <archivo> --listar          (solo diagnostico, no escribe nada)

En ARES: Output > Generate Gerber/Excellon Files, formato RS274X, con Bottom Copper y Drill
marcados, y DESMARCA "Edge (will appear on all layers)". Deja la reflexion en Normal: esta
herramienta espeja sola el cobre inferior (cobre hacia arriba en la maquina). Si ya espejaste
en ARES usa --sin-espejo.
"""

import argparse
import colorsys
import math
import os
import re
import sys
from collections import Counter, defaultdict

VERSION = "1.3"

try:
    import numpy as np
    import shapely
    from shapely import affinity
    from shapely.geometry import LinearRing, LineString, Point, Polygon, box
    from shapely.geometry.polygon import orient
    from shapely.ops import unary_union
    if int(shapely.__version__.split(".")[0]) < 2:
        raise ImportError("shapely demasiado antiguo")
except ImportError:
    print("Faltan librerias (o shapely es muy antiguo). Instala con:\n"
          "    pip install -U shapely ezdxf pymupdf svgelements pillow")
    if sys.stdin and sys.stdin.isatty():
        input("Pulsa Enter para salir...")
    sys.exit(1)

# --------------------------------------------------------------------------
# Constantes
# --------------------------------------------------------------------------
TOL = 0.004          # mm: error maximo al aproximar curvas con segmentos
MM_PT = 25.4 / 72.0  # PDF: 1 punto en mm
MM_PX = 25.4 / 96.0  # SVG: 1 px (96 ppp) en mm
CIERRE = 0.01        # mm: tapa rendijas numericas entre formas que se tocan
AREA_MIN = 0.0005    # mm2: descarta motas diminutas del resultado
SIMPLIF = 0.003      # mm: simplificacion final (reduce vertices sin afectar la precision)


class ErrorUsuario(Exception):
    """Error con mensaje claro para el usuario (no es un fallo del programa)."""


# --------------------------------------------------------------------------
# Utilidades generales
# --------------------------------------------------------------------------
def importar(*nombres, paquete):
    for n in nombres:
        try:
            return __import__(n)
        except ImportError:
            continue
    raise ErrorUsuario("Falta la libreria '%s'. Instalala con:  pip install %s" % (paquete, paquete))


def segs_cuarto(r, tol=TOL):
    """Segmentos por cuarto de circunferencia para que el error sea < tol."""
    x = 1.0 - tol / max(r, 1e-9)
    if x <= 0:
        return 3
    a = 2.0 * math.acos(min(x, 1.0))
    if a <= 1e-9:
        return 48
    return int(max(3, min(48, math.ceil((math.pi / 2) / a))))


def n_curva(longitud_mm):
    """Cantidad de segmentos para aproximar una curva de esa longitud."""
    return int(min(200, max(4, math.ceil(math.sqrt(max(longitud_mm, 1e-9) / (5 * TOL))))))


def bezier(p0, p1, p2, p3, n):
    t = np.linspace(0.0, 1.0, n + 1)[:, None]
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * (t ** 2) * p2 + (t ** 3) * p3


def poligonos(g):
    """Lista de Polygon que hay dentro de cualquier geometria."""
    if g is None or g.is_empty:
        return []
    if g.geom_type == "Polygon":
        return [g]
    if hasattr(g, "geoms"):
        out = []
        for h in g.geoms:
            out.extend(poligonos(h))
        return out
    return []


def arreglar(g):
    """Devuelve una geometria valida (corrige autointersecciones)."""
    if g.is_valid:
        return g
    return unary_union(poligonos(shapely.make_valid(g)))


def area_firmada(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


# ---- colores ----
def a_hex(c):
    """Color de PDF/SVG (gris, RGB o CMYK en 0..1) -> '#rrggbb'."""
    if c is None:
        return None
    c = tuple(c)
    if len(c) == 1:
        r = g = b = c[0]
    elif len(c) == 4:
        k = c[3]
        r, g, b = [(1 - v) * (1 - k) for v in c[:3]]
    else:
        r, g, b = c[:3]
    return "#%02x%02x%02x" % tuple(int(round(max(0.0, min(1.0, v)) * 255)) for v in (r, g, b))


def norm_hex(s):
    s = s.strip().lower()
    if not s.startswith("#"):
        s = "#" + s
    if len(s) == 4:
        s = "#" + "".join(ch * 2 for ch in s[1:])
    return s


def igual_color(h1, h2, tol=8):
    if h1 is None or h2 is None:
        return False
    a = [int(h1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(h2[i:i + 2], 16) for i in (1, 3, 5)]
    return max(abs(x - y) for x, y in zip(a, b)) <= tol


def nombre_color(h):
    r, g, b = [int(h[i:i + 2], 16) for i in (1, 3, 5)]
    if max(r, g, b) < 60:
        return "negro"
    if min(r, g, b) > 215:
        return "blanco"
    hh, ss, vv = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    if ss < 0.18:
        return "gris"
    d = hh * 360
    for lim, nom in ((15, "rojo"), (40, "naranja"), (70, "amarillo"), (165, "verde"),
                     (200, "cian"), (260, "azul"), (345, "magenta")):
        if d < lim:
            return nom
    return "rojo"


# --------------------------------------------------------------------------
# Modelo de datos
# --------------------------------------------------------------------------
class Entrada:
    """Resultado de leer un archivo: operaciones de dibujo en mm (Y hacia arriba)."""

    def __init__(self, tipo):
        self.tipo = tipo
        self.ops = []      # [(modo, geometria)]  modo: 'sumar' (cobre) | 'restar' (borra, color de fondo)
        self.borde = []    # geometrias que solo sirven para dimensionar la placa
        self.avisos = []
        self.resumen = []  # lineas para el informe
        self.agujeros = []         # [(x, y, diametro)] en mm: taladros que se dejan expuestos (Gerber+Excellon)
        self.ranuras = []          # [(x0, y0, x1, y1, ancho)] en mm
        self.espejo_sugerido = True


class Contexto:
    """Filtros por color y estadisticas (PDF y SVG)."""

    def __init__(self, a, fondo):
        self.fondo = fondo
        self.incl = {norm_hex(x) for x in (a.incluir or [])}
        self.excl = {norm_hex(x) for x in (a.excluir or [])}
        self.borde = {norm_hex(x) for x in (a.borde or [])}
        self.stats = defaultdict(lambda: {"rellenos": 0, "trazos": 0, "anchos": Counter()})
        self.n_fino = 0
        self.n_cont = 0
        self.n_total = 0

    def poner(self, ent, hexc, g):
        if g is None or g.is_empty:
            return
        if hexc in self.excl:
            return
        if hexc in self.borde:
            ent.borde.append(g)
            return
        modo = "restar" if igual_color(hexc, self.fondo) else "sumar"
        if modo == "sumar" and self.incl and hexc not in self.incl:
            return
        ent.ops.append((modo, g))


# --------------------------------------------------------------------------
# Geometria: trazos y rellenos -> poligonos
# --------------------------------------------------------------------------
def trazo_a_poligono(pts, cerrado, ancho, cap="round", join="round"):
    """Convierte una polilinea con grosor en un poligono (tapas y esquinas incluidas)."""
    pts = np.asarray(pts, float)
    if len(pts) > 1:
        keep = np.r_[True, np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-9]
        pts = pts[keep]
    hw = ancho / 2.0
    q = segs_cuarto(hw)
    if len(pts) == 1:
        if cap == "round":
            return Point(pts[0]).buffer(hw, quad_segs=q)
        if cap == "square":
            return box(pts[0][0] - hw, pts[0][1] - hw, pts[0][0] + hw, pts[0][1] + hw)
        return None
    if cerrado and len(pts) >= 4 and np.linalg.norm(pts[0] - pts[-1]) < 1e-9:
        pts = pts[:-1]
    g = LinearRing(pts) if (cerrado and len(pts) >= 3) else LineString(pts)
    return g.buffer(hw, quad_segs=q, cap_style=cap, join_style=join)


def relleno_a_geom(subs, even_odd=True):
    """Subcaminos cerrados -> geometria rellena segun la regla par-impar o la de bobinado."""
    polys = []
    for pts, _ in subs:
        pts = np.asarray(pts, float)
        if len(pts) < 3:
            continue
        ar = area_firmada(pts)
        if abs(ar) < 1e-10:
            continue
        polys.append((arreglar(Polygon(pts)), 1 if ar > 0 else -1, abs(ar)))
    if not polys:
        return None
    if even_odd:
        g = polys[0][0]
        for p, _, _ in polys[1:]:
            g = g.symmetric_difference(p)
        return g
    ref = max(polys, key=lambda t: t[2])[1]
    pos = unary_union([p for p, s, _ in polys if s == ref])
    neg = [p for p, s, _ in polys if s != ref]
    return pos if not neg else pos.symmetric_difference(unary_union(neg))


def componer(ops):
    """Aplica las operaciones en orden (como las pintaria un visor): sumar = pinta cobre,
    restar = pinta con el color de fondo (borra). Agrupa operaciones seguidas del mismo tipo."""
    acc = Polygon()
    lote, modo_lote = [], None

    def volcar():
        nonlocal acc, lote
        if lote:
            g = unary_union(lote)
            acc = acc.union(g) if modo_lote == "sumar" else acc.difference(g)
            lote = []

    for modo, g in ops:
        if g is None or g.is_empty:
            continue
        if modo != modo_lote:
            volcar()
            modo_lote = modo
        lote.append(g)
    volcar()
    return acc


def anadir_forma(ent, ctx, a, subs, relleno, even_odd, trazo, ancho_mm, cap, join):
    """Registra una forma (relleno y/o trazo) en la entrada. subs = [(puntos_mm, cerrado)]."""
    ctx.n_total += 1
    if relleno:
        ctx.stats[relleno]["rellenos"] += 1
        ctx.poner(ent, relleno, relleno_a_geom(subs, even_odd))
    if not trazo:
        return
    reg = ctx.stats[trazo]
    reg["trazos"] += 1
    reg["anchos"][round(ancho_mm, 2)] += 1
    contornos, lineas = [], []
    for pts, cerrado in subs:
        if cerrado and not relleno:
            ctx.n_cont += 1
            if a.rellenar_contornos:
                contornos.append((pts, True))
                continue
        lineas.append((pts, cerrado))
    if contornos:
        ctx.poner(ent, trazo, relleno_a_geom(contornos, True))
    if not lineas:
        return
    w = ancho_mm
    if w < 0.02:
        if a.ancho_pista > 0:
            w = a.ancho_pista
        else:
            ctx.n_fino += len(lineas)
            return
    polis = [trazo_a_poligono(p, c, w, cap, join) for p, c in lineas]
    ctx.poner(ent, trazo, unary_union([p for p in polis if p is not None]))


def informe_colores(ent, ctx):
    ent.resumen.append("Colores encontrados:")
    orden = sorted(ctx.stats.items(), key=lambda kv: -(kv[1]["rellenos"] + kv[1]["trazos"]))
    for h, r in orden:
        anchos = ", ".join("%.2f x%d" % (w, n) for w, n in r["anchos"].most_common(4))
        extra = ("  | anchos de trazo (mm): " + anchos) if anchos else ""
        fondo = "  <- fondo (lo blanco/fondo se usa como borrador)" if igual_color(h, ctx.fondo) else ""
        ent.resumen.append("   %s %-10s rellenos %4d | trazos %4d%s%s"
                           % (h, "(%s)" % nombre_color(h), r["rellenos"], r["trazos"], extra, fondo))
    if ctx.n_fino:
        ent.avisos.append("%d trazos de grosor 0 (hairline) se ignoraron. Si son pistas, usa --ancho-pista 0.3."
                          % ctx.n_fino)
    if ctx.n_cont and not a_rellenar(ctx) and ctx.n_cont >= 0.3 * max(ctx.n_total, 1):
        ent.avisos.append("%d contornos cerrados sin relleno. Si exportaste 'solo contornos' de las pistas, "
                          "usa --rellenar-contornos." % ctx.n_cont)


def a_rellenar(ctx):
    return getattr(ctx, "rellenar", False)


# --------------------------------------------------------------------------
# Lectura de PDF (PyMuPDF)
# --------------------------------------------------------------------------
def leer_pdf(ruta, a):
    fitz = importar("pymupdf", "fitz", paquete="pymupdf")
    try:
        doc = fitz.open(ruta)
    except Exception as e:
        raise ErrorUsuario("No pude abrir el PDF: %s" % e)
    ent = Entrada("PDF")
    if doc.page_count == 0:
        raise ErrorUsuario("El PDF no tiene paginas.")
    if doc.page_count > 1:
        ent.avisos.append("El PDF tiene %d paginas; solo se usa la primera." % doc.page_count)
    pag = doc[0]
    rm = pag.rotation_matrix if pag.rotation else None
    k = MM_PT * a.escala
    W, H = pag.rect.width, pag.rect.height
    dibujos = pag.get_drawings()
    if not dibujos:
        raise ErrorUsuario("El PDF no contiene dibujo vectorial (puede ser una imagen). "
                           "Exporta de nuevo desde ARES con 'Export Adobe PDF File'.")
    if pag.get_images():
        ent.avisos.append("El PDF contiene imagenes raster; no se pueden convertir (solo se lee el vector).")
    try:
        if pag.get_text("text").strip():
            ent.avisos.append("El PDF contiene texto real (fuentes); no se convierte. "
                              "El texto dibujado como lineas si se lee.")
    except Exception:
        pass

    def es_fondo(d):
        it = d.get("items") or []
        r = d["rect"]
        return (d.get("fill") is not None and len(it) <= 2 and it[0][0] in ("re", "qu")
                and r.width * r.height >= 0.9 * W * H)

    fondo = "#ffffff"
    for d in dibujos:
        if es_fondo(d):
            fondo = a_hex(d["fill"])
            break
    ctx = Contexto(a, fondo)
    ctx.rellenar = a.rellenar_contornos
    CAPS = {0: "flat", 1: "round", 2: "square"}
    JOINS = {0: "mitre", 1: "round", 2: "bevel"}

    def P(x, y):
        if rm is not None:
            x, y = rm.a * x + rm.c * y + rm.e, rm.b * x + rm.d * y + rm.f
        return np.array([x * k, -y * k])

    for d in dibujos:
        if es_fondo(d):
            continue
        tipo = d.get("type") or ""
        subs, cur = [], []

        def cerrar_cur():
            nonlocal cur
            if len(cur) >= 2:
                cerr = np.linalg.norm(cur[0] - cur[-1]) < 1e-4 or (bool(d.get("closePath")) and len(cur) >= 3)
                subs.append((np.array(cur), cerr))
            cur = []

        for it in d["items"]:
            op = it[0]
            if op == "l":
                p0, p1 = P(it[1].x, it[1].y), P(it[2].x, it[2].y)
                if cur and np.linalg.norm(cur[-1] - p0) > 1e-4:
                    cerrar_cur()
                if not cur:
                    cur = [p0]
                cur.append(p1)
            elif op == "c":
                p0, p1, p2, p3 = [P(q.x, q.y) for q in it[1:5]]
                if cur and np.linalg.norm(cur[-1] - p0) > 1e-4:
                    cerrar_cur()
                if not cur:
                    cur = [p0]
                n = n_curva(np.linalg.norm(p1 - p0) + np.linalg.norm(p2 - p1) + np.linalg.norm(p3 - p2))
                cur.extend(list(bezier(p0, p1, p2, p3, n)[1:]))
            elif op == "re":
                cerrar_cur()
                r = it[1]
                pts = [P(r.x0, r.y0), P(r.x1, r.y0), P(r.x1, r.y1), P(r.x0, r.y1)]
                subs.append((np.array(pts), True))
            elif op == "qu":
                cerrar_cur()
                q = it[1]
                pts = [P(q.ul.x, q.ul.y), P(q.ur.x, q.ur.y), P(q.lr.x, q.lr.y), P(q.ll.x, q.ll.y)]
                subs.append((np.array(pts), True))
        cerrar_cur()
        if not subs:
            continue
        relleno = a_hex(d["fill"]) if ("f" in tipo and d.get("fill") is not None
                                       and (d.get("fill_opacity") is None or d["fill_opacity"] > 0)) else None
        trazo = a_hex(d["color"]) if ("s" in tipo and d.get("color") is not None
                                      and (d.get("stroke_opacity") is None or d["stroke_opacity"] > 0)) else None
        if not relleno and not trazo:
            continue
        lc = d.get("lineCap")
        lc = lc[0] if isinstance(lc, (tuple, list)) and lc else (lc or 0)
        lj = int(d.get("lineJoin") or 0)
        ancho = (d.get("width") or 0.0) * k
        anadir_forma(ent, ctx, a, subs, relleno, not (d.get("even_odd") is False), trazo, ancho,
                     CAPS.get(int(lc), "round"), JOINS.get(lj, "round"))

    ent.resumen.append("Pagina: %.1f x %.1f mm | elementos de dibujo: %d | fondo: %s"
                       % (W * MM_PT * a.escala, H * MM_PT * a.escala, len(dibujos), fondo))
    informe_colores(ent, ctx)
    return ent


# --------------------------------------------------------------------------
# Lectura de SVG (svgelements)
# --------------------------------------------------------------------------
def leer_svg(ruta, a):
    sv = importar("svgelements", paquete="svgelements")
    ent = Entrada("SVG")
    try:
        svg = sv.SVG.parse(ruta, reify=True)
    except Exception as e:
        raise ErrorUsuario("No pude leer el SVG: %s" % e)
    formas = [e for e in svg.elements() if isinstance(e, sv.Shape)]
    if any(isinstance(e, sv.SVGText) for e in svg.elements()):
        ent.avisos.append("El SVG contiene texto real; no se convierte (pasalo a trayecto en Inkscape).")
    if not formas:
        raise ErrorUsuario("El SVG no contiene formas vectoriales.")
    k = MM_PX * a.escala

    def visible(e):
        v = e.values
        return (str(v.get("display", "")).strip() != "none"
                and str(v.get("visibility", "")).strip() not in ("hidden", "collapse"))

    def color(c):
        if c is None or c.value is None or getattr(c, "alpha", 255) == 0:
            return None
        return c.hexrgb.lower()

    formas = [e for e in formas if visible(e)]
    cajas = [e.bbox() for e in formas]
    cajas_ok = [b for b in cajas if b is not None]
    if not cajas_ok:
        raise ErrorUsuario("El SVG no contiene geometria visible.")
    try:
        W, H = float(svg.width), float(svg.height)
    except Exception:
        W = max(b[2] for b in cajas_ok) - min(b[0] for b in cajas_ok)
        H = max(b[3] for b in cajas_ok) - min(b[1] for b in cajas_ok)
    fondo, idx_fondo = "#ffffff", set()
    for i, (e, b) in enumerate(zip(formas, cajas)):
        if b is not None and color(e.fill) and (b[2] - b[0]) * (b[3] - b[1]) >= 0.9 * W * H \
                and isinstance(e, (sv.Rect, sv.Path, sv.Polygon)):
            if not idx_fondo:
                fondo = color(e.fill)
            idx_fondo.add(i)
            break
    ctx = Contexto(a, fondo)
    ctx.rellenar = a.rellenar_contornos

    def P(pt):
        return np.array([pt.x * k, -pt.y * k], float)

    for i, e in enumerate(formas):
        if i in idx_fondo:
            continue
        try:
            path = sv.Path(e)
        except Exception:
            continue
        subs, cur = [], []

        def cerrar_cur(flag):
            nonlocal cur
            if len(cur) >= 2:
                subs.append((np.array(cur), bool(flag) or np.linalg.norm(cur[0] - cur[-1]) < 1e-6))
            cur = []

        for seg in path:
            if isinstance(seg, sv.Move):
                cerrar_cur(False)
                cur = [P(seg.end)]
            elif isinstance(seg, sv.Close):
                if not cur:
                    cur = [P(seg.start)]
                cerrar_cur(True)
            elif isinstance(seg, sv.Line):
                if not cur:
                    cur = [P(seg.start)]
                cur.append(P(seg.end))
            else:
                if not cur:
                    cur = [P(seg.start)]
                try:
                    n = n_curva(float(seg.length()) * k)
                except Exception:
                    n = 24
                for t in np.linspace(0.0, 1.0, n + 1)[1:]:
                    cur.append(P(seg.point(t)))
        cerrar_cur(False)
        if not subs:
            continue
        v = e.values
        relleno, trazo = color(e.fill), color(e.stroke)
        ancho = float(getattr(e, "implicit_stroke_width", e.stroke_width) or 0.0) * k
        cap = {"round": "round", "square": "square"}.get(str(v.get("stroke-linecap", "butt")), "flat")
        join = {"round": "round", "bevel": "bevel"}.get(str(v.get("stroke-linejoin", "miter")), "mitre")
        eo = str(v.get("fill-rule", "nonzero")) == "evenodd"
        anadir_forma(ent, ctx, a, subs, relleno, eo, trazo if ancho >= 0 else None, ancho, cap, join)

    ent.resumen.append("Hoja: %.1f x %.1f mm | formas: %d | fondo: %s"
                       % (W * MM_PX * a.escala, H * MM_PX * a.escala, len(formas), fondo))
    informe_colores(ent, ctx)
    return ent


# --------------------------------------------------------------------------
# Lectura de DXF (ezdxf)
# --------------------------------------------------------------------------
RE_BORDE = re.compile(r"edge|outline|contorno|borde|board", re.I)
RE_NOCOBRE = re.compile(r"silk|legend|resist|mask|paste|drill|hole|slot|text|assembl|dimens|keepout|"
                        r"courtyard|mech|fab|serigr|taladro|agujero", re.I)
INSUNITS = {1: ("in", 25.4), 2: ("ft", 304.8), 4: ("mm", 1.0), 5: ("cm", 10.0), 6: ("m", 1000.0),
            9: ("thou", 0.0254)}
UNIDADES = {"mm": 1.0, "cm": 10.0, "in": 25.4, "thou": 0.0254}


def encadenar(segs, tol):
    """Une segmentos sueltos (LINE/ARC) en contornos. Al llegar a un nodo donde se tocan
    varias formas, prefiere el segmento que sigue en el orden del archivo.
    Devuelve (cerrados, abiertos) como listas de arrays (n,2)."""
    n = len(segs)
    usado = [False] * n
    celdas = defaultdict(list)

    def clave(p):
        return (int(math.floor(p[0] / tol)), int(math.floor(p[1] / tol)))

    def cerca(p, q):
        return abs(p[0] - q[0]) <= tol and abs(p[1] - q[1]) <= tol

    for i, s in enumerate(segs):
        celdas[clave(s[0])].append((i, 0))
        celdas[clave(s[-1])].append((i, 1))

    def buscar(p, ref):
        kx, ky = clave(p)
        mejor = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j, e in celdas.get((kx + dx, ky + dy), ()):
                    if usado[j]:
                        continue
                    q = segs[j][0] if e == 0 else segs[j][-1]
                    if cerca(p, q):
                        sc = abs(j - ref)
                        if mejor is None or sc < mejor[0]:
                            mejor = (sc, j, e)
        return mejor

    def cerrada(cad):
        return len(cad) >= 2 and cerca(cad[0][0], cad[-1][-1])

    cerrados, abiertos = [], []
    for i in range(n):
        if usado[i]:
            continue
        usado[i] = True
        cad, ref = [segs[i]], i
        while not cerrada(cad):
            b = buscar(cad[-1][-1], ref)
            if b is None:
                break
            _, j, e = b
            usado[j], ref = True, j
            cad.append(segs[j] if e == 0 else segs[j][::-1])
        if not cerrada(cad):
            ref = i
            while True:
                b = buscar(cad[0][0], ref)
                if b is None:
                    break
                _, j, e = b
                usado[j], ref = True, j
                cad.insert(0, segs[j][::-1] if e == 0 else segs[j])
                if cerrada(cad):
                    break
        pts = np.vstack([cad[0]] + [c[1:] for c in cad[1:]])
        (cerrados if cerrada(cad) else abiertos).append(pts)
    return cerrados, abiertos


def lazos_a_geom(lazos, respetar_huecos):
    """Contornos cerrados -> geometria. Por defecto cada contorno es una forma rellena (se unen).
    Con respetar_huecos, los contornos anidados alternan relleno/hueco (zonas de cobre con aislamiento)."""
    base = []
    for pts in lazos:
        if len(pts) >= 3 and abs(area_firmada(pts)) > 1e-10:
            base.append(arreglar(Polygon(pts)))
    if not base:
        return None
    if not respetar_huecos:
        return unary_union(base)
    arbol = shapely.STRtree(base)
    pares = arbol.query(np.array(base, dtype=object), predicate="within")
    prof = np.zeros(len(base), int)
    for i, j in zip(pares[0], pares[1]):
        if i != j:
            prof[i] += 1
    res = Polygon()
    for d in range(int(prof.max()) + 1):
        capa = [base[i] for i in range(len(base)) if prof[i] == d]
        if capa:
            u = unary_union(capa)
            res = res.union(u) if d % 2 == 0 else res.difference(u)
    return res


def leer_dxf(ruta, a):
    ezdxf = importar("ezdxf", paquete="ezdxf")
    from ezdxf import bbox as ezbbox
    from ezdxf import path as ezpath
    try:
        doc = ezdxf.readfile(ruta)
    except Exception:
        try:
            from ezdxf import recover
            doc, _ = recover.readfile(ruta)
        except Exception as e:
            raise ErrorUsuario("No pude leer el DXF: %s" % e)
    ent = Entrada("DXF")

    ents = []

    def agregar(e, nivel=0):
        if e.dxftype() == "INSERT" and nivel < 6:
            try:
                for v in e.virtual_entities():
                    agregar(v, nivel + 1)
            except Exception:
                pass
        else:
            ents.append(e)

    for e in doc.modelspace():
        agregar(e)
    if not ents:
        raise ErrorUsuario("El DXF esta vacio.")

    # ---- clasificar por capa ----
    capas = Counter(e.dxf.layer for e in ents if e.dxf.hasattr("layer"))
    incl = [s.lower() for s in (a.incluir or [])]
    excl = [s.lower() for s in (a.excluir or [])]
    borde_k = [s.lower() for s in (a.borde or [])]

    def destino(capa):
        c = capa.lower()
        if borde_k:
            if any(s in c for s in borde_k):
                return "borde"
        elif not incl and RE_BORDE.search(capa):
            return "borde"
        if any(s in c for s in excl):
            return None
        if incl:
            return "cobre" if any(s in c for s in incl) else None
        return None if RE_NOCOBRE.search(capa) else "cobre"

    usados = [(e, destino(e.dxf.layer)) for e in ents]
    usados = [(e, d) for e, d in usados if d]
    ent.resumen.append("Capas del DXF (entidades): " + ", ".join("%s=%d [%s]" % (c, n, destino(c) or "ignorada")
                                                                for c, n in capas.most_common()))
    if not [1 for _, d in usados if d == "cobre"]:
        raise ErrorUsuario("Ninguna capa se reconocio como cobre. Capas: %s. Indica la capa con --incluir."
                           % ", ".join(capas))

    try:
        ext = ezbbox.extents([e for e, _ in usados])
        diag = float((ext.extmax - ext.extmin).magnitude)
    except Exception:
        diag = 0.0
    if not diag > 0:
        raise ErrorUsuario("No pude calcular el tamano del dibujo DXF.")
    dist = diag * 2e-5         # tolerancia de aplanado (relativa: no depende de las unidades)
    tol_enc = diag * 1e-4      # tolerancia para unir extremos de segmentos

    # ---- extraer primitivas (en unidades del DXF) ----
    cerrados, sueltos, trazos, rellenos, bordes_pts = [], [], [], [], []
    ignorados = Counter()
    for e, dest in usados:
        t = e.dxftype()
        try:
            if t == "HATCH":
                lazos = []
                for p in ezpath.from_hatch(e):
                    pts = np.array([(v.x, v.y) for v in p.flattening(dist)])
                    if len(pts) >= 3:
                        lazos.append(pts)
                if dest == "borde":
                    bordes_pts.extend(lazos)
                elif lazos:
                    rellenos.append(lazos)
            elif t in ("SOLID", "TRACE"):
                pts = np.array([(v.x, v.y) for v in e.wcs_vertices()])
                (bordes_pts if dest == "borde" else rellenos).append(pts if dest == "borde" else [pts])
            elif t in ("LINE", "ARC", "CIRCLE", "ELLIPSE", "SPLINE", "LWPOLYLINE", "POLYLINE"):
                ancho = 0.0
                if t == "LWPOLYLINE":
                    ancho = float(e.dxf.const_width or 0.0)
                elif t == "POLYLINE":
                    ancho = max(float(e.dxf.default_start_width or 0.0), float(e.dxf.default_end_width or 0.0))
                for sp in ezpath.make_path(e).sub_paths():
                    pts = np.array([(v.x, v.y) for v in sp.flattening(dist)])
                    if len(pts) < 2:
                        continue
                    cerr = bool(sp.is_closed)
                    if dest == "borde":
                        bordes_pts.append(pts)
                    elif ancho > 0:
                        trazos.append((pts, cerr, ancho))
                    elif cerr and len(pts) >= 4:
                        cerrados.append(pts)
                    else:
                        sueltos.append(pts)
            else:
                ignorados[t] += 1
        except Exception:
            ignorados[t + " (error)"] += 1
    if ignorados:
        ent.avisos.append("Entidades ignoradas: " + ", ".join("%s=%d" % kv for kv in ignorados.items()))

    # ---- unir segmentos sueltos en contornos ----
    abiertos = []
    if sueltos:
        c2, abiertos = encadenar(sueltos, tol_enc)
        cerrados.extend(c2)

    # ---- unidades ----
    todos = cerrados + abiertos + [p for p, _, _ in trazos] + [p for r in rellenos for p in r] + bordes_pts
    xs = np.concatenate([p[:, 0] for p in todos])
    ys = np.concatenate([p[:, 1] for p in todos])
    w_raw, h_raw = float(xs.max() - xs.min()), float(ys.max() - ys.min())
    if a.unidades != "auto":
        f, motivo = UNIDADES[a.unidades], "indicadas por ti"
    else:
        ins = doc.header.get("$INSUNITS", 0)
        cand = ([INSUNITS[ins] + ("declaradas en el DXF",)] if ins in INSUNITS else [])
        cand += [(n, v, "deducidas por el tamano") for n, v in (("mm", 1.0), ("thou", 0.0254), ("in", 25.4), ("cm", 10.0))]
        f, motivo = 1.0, "no se pudo deducir; se asume mm"
        for nombre, v, why in cand:
            if 3 <= max(w_raw, h_raw) * v <= 600:
                f, motivo = v, "%s: %s" % (nombre, why)
                break
    k = f * a.escala
    ent.resumen.append("Unidades del DXF: %s | tamano total %.2f x %.2f mm (VERIFICA contra ARES; "
                       "si no coincide usa --unidades mm|in|thou|cm)" % (motivo, w_raw * k, h_raw * k))
    ent.resumen.append("Contornos cerrados: %d | trazos con grosor: %d | rellenos: %d | lineas sin cerrar: %d"
                       % (len(cerrados), len(trazos), len(rellenos), len(abiertos)))

    # ---- geometria ----
    g = lazos_a_geom([p * k for p in cerrados], a.respetar_huecos)
    if g is not None:
        ent.ops.append(("sumar", g))
        if not a.respetar_huecos and len(cerrados) > 1:
            try:
                arbol = shapely.STRtree([Polygon(p * k) for p in cerrados if len(p) >= 3])
                anidados = len(set(arbol.query([Polygon(p * k) for p in cerrados if len(p) >= 3],
                                               predicate="within")[0]))
                if anidados > len(cerrados) * 0.02 and anidados > 3:
                    ent.avisos.append("Hay contornos dentro de otros (taladros de pads o zonas de cobre). Se rellenan "
                                      "(lo normal para el sticker). Si tienes zona de cobre/plano de masa con "
                                      "aislamientos, usa --respetar-huecos.")
            except Exception:
                pass
    for pts, cerr, ancho in trazos:
        ent.ops.append(("sumar", trazo_a_poligono(pts * k, cerr, ancho * k, "flat", "mitre")))
    for lazos in rellenos:
        r = lazos_a_geom([p * k for p in lazos], True)
        if r is not None:
            ent.ops.append(("sumar", r))
    if abiertos:
        if a.ancho_pista > 0:
            ent.ops.append(("sumar", unary_union([trazo_a_poligono(p * k, False, a.ancho_pista, "round", "round")
                                                  for p in abiertos])))
            ent.avisos.append("%d lineas abiertas se engrosaron a %.2f mm (--ancho-pista)." % (len(abiertos), a.ancho_pista))
        else:
            ent.avisos.append("%d lineas abiertas (sin cerrar) se ignoraron. Si son pistas dibujadas como linea central, "
                              "usa --ancho-pista 0.3." % len(abiertos))
    for pts in bordes_pts:
        if len(pts) >= 2:
            ent.borde.append(LineString(pts * k))
    return ent


# --------------------------------------------------------------------------
# Lectura de Gerber RS-274X y Excellon (analizador propio: no necesita librerias extra)
# --------------------------------------------------------------------------
def _evaluar(expr, variables):
    """Evalua una expresion de macro de aperturas: numeros, $n, + - x / ( )."""
    expr = re.sub(r"\$(\d+)", lambda m: "(%r)" % float(variables.get(int(m.group(1)), 0.0)), expr.strip())
    expr = expr.replace("X", "*").replace("x", "*")
    if not expr:
        return 0.0
    if not re.fullmatch(r"[0-9eE+\-*/(). ]+", expr):
        raise ValueError("expresion no valida: %s" % expr)
    return float(eval(expr, {"__builtins__": {}}, {}))      # solo digitos y operadores (validado arriba)


def _macro_a_geom(cuerpo, params):
    """Construye la forma de una apertura-macro (en unidades del archivo, centrada en el origen)."""
    var = {i + 1: float(v) for i, v in enumerate(params)}
    acc, no_soportados = Polygon(), set()
    for sent in cuerpo:
        sent = sent.strip()
        if not sent or re.match(r"0(\s|$)", sent):
            continue
        if sent.startswith("$"):
            nom, expr = sent.split("=", 1)
            var[int(nom.strip()[1:])] = _evaluar(expr, var)
            continue
        v = [_evaluar(t, var) for t in sent.split(",")]
        cod = int(round(v[0]))
        rot = 0.0
        if cod == 1:                                     # circulo: expo, diam, cx, cy [, rot]
            g = Point(v[3], v[4]).buffer(v[2] / 2.0, quad_segs=16)
            rot = v[5] if len(v) > 5 else 0.0
        elif cod in (2, 20):                             # linea con grosor: expo, ancho, x1,y1,x2,y2, rot
            g = LineString([(v[3], v[4]), (v[5], v[6])]).buffer(v[2] / 2.0, cap_style="flat")
            rot = v[7] if len(v) > 7 else 0.0
        elif cod == 21:                                  # rectangulo centrado: expo, ancho, alto, cx, cy, rot
            g = box(v[4] - v[2] / 2, v[5] - v[3] / 2, v[4] + v[2] / 2, v[5] + v[3] / 2)
            rot = v[6] if len(v) > 6 else 0.0
        elif cod == 22:                                  # rectangulo por esquina: expo, ancho, alto, x, y, rot
            g = box(v[4], v[5], v[4] + v[2], v[5] + v[3])
            rot = v[6] if len(v) > 6 else 0.0
        elif cod == 4:                                   # contorno: expo, n, x0,y0 ... xn,yn, rot
            n = int(round(v[2]))
            g = Polygon([(v[3 + 2 * k], v[4 + 2 * k]) for k in range(n + 1)])
            rot = v[3 + 2 * (n + 1)] if len(v) > 3 + 2 * (n + 1) else 0.0
        elif cod == 5:                                   # poligono regular: expo, n, cx, cy, diam, rot
            n, cx, cy, d = int(round(v[2])), v[3], v[4], v[5]
            g = Polygon([(cx + d / 2 * math.cos(2 * math.pi * k / n), cy + d / 2 * math.sin(2 * math.pi * k / n))
                         for k in range(n)])
            rot = v[6] if len(v) > 6 else 0.0
        elif cod == 7:                                   # termico: cx, cy, d ext, d int, rendija, rot
            cx, cy, do, di, gap = v[1:6]
            anillo = Point(cx, cy).buffer(do / 2.0, quad_segs=16).difference(Point(cx, cy).buffer(di / 2.0, quad_segs=16))
            barras = unary_union([box(cx - do, cy - gap / 2, cx + do, cy + gap / 2),
                                  box(cx - gap / 2, cy - do, cx + gap / 2, cy + do)])
            g = anillo.difference(barras)
            rot = v[6] if len(v) > 6 else 0.0
            v = [7, 1.0]
        else:
            no_soportados.add(cod)
            continue
        if rot:
            g = affinity.rotate(g, rot, origin=(0, 0))
        acc = acc.union(g) if v[1] else acc.difference(g)
    return acc, no_soportados


class Apertura:
    def __init__(self, tipo, forma, diam=0.0):
        self.tipo, self.forma, self.diam = tipo, forma, diam      # forma en mm, centrada en el origen


class Gerber:
    """Interprete de Gerber RS-274X. Produce self.ops = [('sumar'|'restar', geometria_mm)] en el orden
    del archivo (polaridad oscura = pinta cobre, clara = borra), mas contadores y avisos."""

    def __init__(self, texto, nombre=""):
        self.nombre = nombre
        self.ops, self.avisos = [], []
        self.n_trazos = self.n_flashes = self.n_regiones = 0
        self.u = None                      # mm por unidad del archivo
        self.fs = {"zeros": "L", "xi": 2, "xd": 4, "yi": 2, "yd": 4}
        self.abs = True
        self.aperturas, self.macros = {}, {}
        self.ap = None
        self.x = self.y = 0.0
        self.modo = 1                      # 1 lineal | 2 horario | 3 antihorario
        self.cuadrante = None              # 74 un cuadrante | 75 multicuadrante | None = detectar
        self.oscuro = True
        self.region, self.contorno, self.contornos = False, None, []
        self.d_modal = 2
        self.sr = None
        self.fin = False
        self._avisados = set()
        self._ejecutar(texto)

    # ---- utilidades ----
    def _aviso(self, msg):
        if msg not in self._avisados:
            self._avisados.add(msg)
            self.avisos.append("%s: %s" % (self.nombre or "Gerber", msg))

    def _factor(self):
        if self.u is None:
            self._aviso("el archivo no declara unidades (%MOIN/%MOMM); se asumen mm")
            self.u = 1.0
        return self.u

    def _modo_op(self):
        return "sumar" if self.oscuro else "restar"

    def _numero(self, s, eje):
        """Texto de coordenada -> mm segun el formato %FS."""
        u = self._factor()
        if "." in s:
            return float(s) * u
        ni, nd = (self.fs["xi"], self.fs["xd"]) if eje == "x" else (self.fs["yi"], self.fs["yd"])
        neg = s.startswith("-")
        s = s.lstrip("+-")
        if self.fs["zeros"] == "T":
            s = s.ljust(ni + nd, "0")
        v = (int(s) if s else 0) / (10.0 ** nd)
        return (-v if neg else v) * u

    # ---- bucle principal ----
    def _ejecutar(self, texto):
        texto = texto.replace("\r", "").replace("\n", "")
        i, n = 0, len(texto)
        while i < n and not self.fin:
            c = texto[i]
            if c == "%":
                j = texto.find("%", i + 1)
                j = n if j < 0 else j
                self._parametro([w for w in texto[i + 1:j].split("*") if w.strip()])
                i = j + 1
            elif c in " \t":
                i += 1
            else:
                j = texto.find("*", i)
                j = n if j < 0 else j
                self._bloque(texto[i:j].strip())
                i = j + 1
        self._cerrar_sr()

    # ---- comandos extendidos %...% ----
    def _parametro(self, palabras):
        if not palabras:
            return
        if palabras[0].strip().startswith("AM"):
            self.macros[palabras[0].strip()[2:].strip()] = [p.strip() for p in palabras[1:]]
            return
        for w in palabras:
            w = w.strip()
            if w.startswith("FS"):
                m = re.match(r"FS([LTD])?([AI])?.*?X(\d)(\d)Y(\d)(\d)", w)
                if m:
                    self.fs = {"zeros": m.group(1) or "L", "xi": int(m.group(3)), "xd": int(m.group(4)),
                               "yi": int(m.group(5)), "yd": int(m.group(6))}
                    self.abs = (m.group(2) or "A") == "A"
            elif w.startswith("MO"):
                self.u = 25.4 if w[2:4].upper() == "IN" else 1.0
            elif w.startswith("AD"):
                m = re.match(r"ADD(\d+)(.*)$", w)
                if m:
                    self._definir_apertura(int(m.group(1)), m.group(2))
            elif w.startswith("LP"):
                self.oscuro = (w[2:3].upper() == "D")
            elif w.startswith("SR"):
                self._sr(w)
            elif w.startswith("IP") and "NEG" in w.upper():
                self._aviso("imagen en polaridad negativa (%IPNEG): no soportado")
            elif w[:2] in ("OF", "SF", "MI", "IR", "AS") and re.search(r"[1-9]", w[2:]):
                self._aviso("comando obsoleto %s ignorado" % w)

    def _definir_apertura(self, codigo, resto):
        u = self._factor()
        m = re.match(r"([^,]+)(?:,(.*))?$", resto.strip())
        if not m:
            return
        nombre, args = m.group(1).strip(), (m.group(2) or "")
        p = [float(t) for t in re.split(r"[Xx]", args) if t.strip()] if args else []
        tipo = nombre.upper() if nombre.upper() in ("C", "R", "O", "P") else "M"
        forma, diam = None, 0.0
        try:
            if tipo == "C":
                diam = p[0] * u
                forma = Point(0, 0).buffer(diam / 2.0, quad_segs=segs_cuarto(diam / 2.0))
                if len(p) > 1 and p[1] > 0:
                    forma = forma.difference(Point(0, 0).buffer(p[1] * u / 2.0, quad_segs=8))
            elif tipo == "R":
                w, h = p[0] * u, p[1] * u
                forma = box(-w / 2, -h / 2, w / 2, h / 2)
                if len(p) > 2 and p[2] > 0:
                    forma = forma.difference(Point(0, 0).buffer(p[2] * u / 2.0, quad_segs=8))
            elif tipo == "O":
                w, h = p[0] * u, p[1] * u
                if w >= h:
                    forma = LineString([(-(w - h) / 2, 0), ((w - h) / 2, 0)]).buffer(h / 2.0, quad_segs=segs_cuarto(h / 2.0))
                else:
                    forma = LineString([(0, -(h - w) / 2), (0, (h - w) / 2)]).buffer(w / 2.0, quad_segs=segs_cuarto(w / 2.0))
                if len(p) > 2 and p[2] > 0:
                    forma = forma.difference(Point(0, 0).buffer(p[2] * u / 2.0, quad_segs=8))
            elif tipo == "P":
                d, nv = p[0] * u, int(round(p[1]))
                rot = math.radians(p[2]) if len(p) > 2 else 0.0
                forma = Polygon([(d / 2 * math.cos(rot + 2 * math.pi * k / nv), d / 2 * math.sin(rot + 2 * math.pi * k / nv))
                                 for k in range(nv)])
                if len(p) > 3 and p[3] > 0:
                    forma = forma.difference(Point(0, 0).buffer(p[3] * u / 2.0, quad_segs=8))
            else:
                cuerpo = self.macros.get(nombre)
                if cuerpo is None:
                    self._aviso("macro de apertura '%s' no definida" % nombre)
                else:
                    g, nosop = _macro_a_geom(cuerpo, p)
                    if nosop:
                        self._aviso("macro '%s' usa primitivas no soportadas %s" % (nombre, sorted(nosop)))
                    forma = affinity.scale(g, u, u, origin=(0, 0))
        except Exception as e:
            self._aviso("no pude construir la apertura D%d (%s): %s" % (codigo, nombre, e))
        self.aperturas[codigo] = Apertura(tipo, forma, diam)

    # ---- repeticion en bloque (SR) ----
    def _sr(self, w):
        self._cerrar_sr()
        m = re.match(r"SR(?:X(\d+))?(?:Y(\d+))?(?:I([\d.+-]+))?(?:J([\d.+-]+))?", w)
        if not m:
            return
        nx, ny = int(m.group(1) or 1), int(m.group(2) or 1)
        if nx > 1 or ny > 1:
            u = self._factor()
            self.sr = (nx, ny, float(m.group(3) or 0) * u, float(m.group(4) or 0) * u, len(self.ops))

    def _cerrar_sr(self):
        if self.sr is None:
            return
        nx, ny, dx, dy, ini = self.sr
        bloque = self.ops[ini:]
        for i in range(nx):
            for j in range(ny):
                if i or j:
                    self.ops.extend((m, affinity.translate(g, i * dx, j * dy)) for m, g in bloque)
        self.sr = None

    # ---- bloques de datos ----
    def _bloque(self, w):
        if not w or re.match(r"G0?4(?!\d)", w):
            return
        xs = ys = is_ = js = None
        d, gs = None, []
        for letra, val in re.findall(r"([GDMXYIJ])\s*([+-]?[\d.]+)", w):
            if letra == "G":
                gs.append(int(float(val)))
            elif letra == "D":
                d = int(float(val))
            elif letra == "M":
                if int(float(val)) in (0, 2):
                    self.fin = True
            elif letra == "X":
                xs = val
            elif letra == "Y":
                ys = val
            elif letra == "I":
                is_ = val
            elif letra == "J":
                js = val
        for g in gs:
            if g in (1, 2, 3):
                self.modo = g
            elif g == 36:
                self._region_ini()
            elif g == 37:
                self._region_fin()
            elif g == 70:
                self.u = 25.4
            elif g == 71:
                self.u = 1.0
            elif g in (74, 75):
                self.cuadrante = g
            elif g == 90:
                self.abs = True
            elif g == 91:
                self.abs = False
        if d is not None and d >= 10:
            self.ap = self.aperturas.get(d)
            if self.ap is None:
                self._aviso("se usa la apertura D%d pero no esta definida" % d)
            d = None
        hay_coord = xs is not None or ys is not None
        op = d if d in (1, 2, 3) else (self.d_modal if hay_coord else None)
        if op is None:
            return
        self.d_modal = op
        nx = self.x if xs is None else (self._numero(xs, "x") if self.abs else self.x + self._numero(xs, "x"))
        ny = self.y if ys is None else (self._numero(ys, "y") if self.abs else self.y + self._numero(ys, "y"))
        if op == 2:
            if self.region:
                self._contorno_nuevo((nx, ny))
        elif op == 1:
            self._dibujar((self.x, self.y), (nx, ny),
                          0.0 if is_ is None else self._numero(is_, "x"),
                          0.0 if js is None else self._numero(js, "y"))
        elif op == 3 and not self.region:
            self._flash(nx, ny)
        self.x, self.y = nx, ny

    # ---- regiones ----
    def _region_ini(self):
        self.region = True
        self.contornos = []
        self.contorno = [(self.x, self.y)]

    def _contorno_nuevo(self, p):
        if self.contorno and len(self.contorno) >= 3:
            self.contornos.append(self.contorno)
        self.contorno = [p]

    def _region_fin(self):
        if not self.region:
            return
        if self.contorno and len(self.contorno) >= 3:
            self.contornos.append(self.contorno)
        polis = []
        for pts in self.contornos:
            p = arreglar(Polygon(pts))
            if not p.is_empty:
                polis.append(p)
        if polis:
            self.ops.append((self._modo_op(), unary_union(polis)))
            self.n_regiones += 1
        self.region, self.contorno, self.contornos = False, None, []

    # ---- trazos, arcos y flashes ----
    def _flash(self, x, y):
        if self.ap is None or self.ap.forma is None:
            return
        self.ops.append((self._modo_op(), affinity.translate(self.ap.forma, x, y)))
        self.n_flashes += 1

    def _dibujar(self, p0, p1, i, j):
        if self.modo == 1:
            pts = [p0, p1]
        else:
            pts = self._arco(p0, p1, i, j)
        if self.region:
            if self.contorno is not None:
                self.contorno.extend(pts[1:])
            return
        ap = self.ap
        if ap is None or ap.forma is None:
            return
        if ap.tipo == "C":
            r = ap.diam / 2.0
            if r <= 0:
                return
            q = segs_cuarto(r)
            if len(pts) == 2 and math.hypot(p1[0] - p0[0], p1[1] - p0[1]) < 1e-9:
                g = Point(p0).buffer(r, quad_segs=q)
            else:
                g = LineString(pts).buffer(r, quad_segs=q, cap_style="round", join_style="round")
        else:                                             # apertura no circular: barrido (casco convexo)
            partes = []
            for a, b in zip(pts[:-1], pts[1:]):
                fa, fb = affinity.translate(ap.forma, a[0], a[1]), affinity.translate(ap.forma, b[0], b[1])
                partes.append(unary_union([fa, fb]).convex_hull)
            g = unary_union(partes)
        self.ops.append((self._modo_op(), g))
        self.n_trazos += 1

    def _arco(self, p0, p1, i, j):
        horario = self.modo == 2
        plena = math.hypot(p1[0] - p0[0], p1[1] - p0[1]) < 1e-9

        def barrido(c, un_cuadrante):
            a0 = math.atan2(p0[1] - c[1], p0[0] - c[0])
            a1 = math.atan2(p1[1] - c[1], p1[0] - c[0])
            if plena:
                b = 0.0 if un_cuadrante else (-2 * math.pi if horario else 2 * math.pi)
            elif horario:
                b = -((a0 - a1) % (2 * math.pi))
            else:
                b = (a1 - a0) % (2 * math.pi)
            return a0, b

        cands = []
        if self.cuadrante in (None, 75):
            cands.append(((p0[0] + i, p0[1] + j), False))
        if self.cuadrante in (None, 74):
            for si in (1, -1):
                for sj in (1, -1):
                    cands.append(((p0[0] + si * abs(i), p0[1] + sj * abs(j)), True))
        mejor = None
        for c, un_cuadrante in cands:
            r0 = math.hypot(p0[0] - c[0], p0[1] - c[1])
            r1 = math.hypot(p1[0] - c[0], p1[1] - c[1])
            a0, b = barrido(c, un_cuadrante)
            if un_cuadrante and abs(b) > math.pi / 2 + 1e-6:
                continue
            err = abs(r0 - r1)
            if mejor is None or err < mejor[0] - 1e-9:
                mejor = (err, c, (r0 + r1) / 2.0, a0, b)
            if self.cuadrante is None and mejor[0] <= max(1e-4, 1e-3 * mejor[2]):
                break
        if mejor is None or (mejor[2] > 0 and mejor[0] > max(1e-3, 0.02 * mejor[2])):
            self._aviso("arco inconsistente; se dibuja como recta")
            return [p0, p1]
        _, c, r, a0, b = mejor
        if r <= 1e-9 or b == 0:
            return [p0, p1]
        paso = 2.0 * math.acos(max(0.0, min(1.0, 1.0 - TOL / r))) if r > TOL else math.pi / 4
        n = int(min(720, max(2, math.ceil(abs(b) / max(paso, 1e-3)))))
        ang = a0 + b * np.linspace(0.0, 1.0, n + 1)
        pts = [(c[0] + r * math.cos(t), c[1] + r * math.sin(t)) for t in ang]
        pts[0], pts[-1] = p0, p1
        return pts


class Excellon:
    """Archivo de taladros Excellon. Guarda los textos de coordenadas sin interpretar: el formato
    (unidades, ceros, digitos) a veces no viene declarado, asi que se decide despues comparando con el cobre."""

    def __init__(self, texto, nombre=""):
        self.nombre = nombre
        self.unidad = None          # 'in' | 'mm' declarada
        self.ceros = None           # 'LZ' | 'TZ' declarado
        self.digitos = None         # (enteros, decimales) declarado
        self.herr = {}              # numero de herramienta -> diametro (unidades del archivo)
        self.taladros = []          # (xs, ys, herr)
        self.ranuras = []           # (x0s, y0s, x1s, y1s, herr)
        self._leer(texto)

    def _leer(self, texto):
        cab, herr, bajo, ult, rut = True, None, False, [None, None], False
        for linea in texto.splitlines():
            l = linea.strip()
            if not l:
                continue
            if l.startswith(";"):
                m = re.search(r"FILE_FORMAT\s*=\s*(\d):(\d)", l)
                if m:
                    self.digitos = (int(m.group(1)), int(m.group(2)))
                continue
            u = l.upper()
            if u.startswith(("INCH", "METRIC")):
                self.unidad = "in" if u.startswith("INCH") else "mm"
                self.ceros = "LZ" if ",LZ" in u else ("TZ" if ",TZ" in u else self.ceros)
                m = re.search(r"(0+)\.(0+)", u)
                if m:
                    self.digitos = (len(m.group(1)), len(m.group(2)))
                continue
            if cab:
                if u in ("%", "M95"):
                    cab = False
                    continue
                if u.startswith("M71") or u.startswith("M72"):
                    self.unidad = "mm" if u.startswith("M71") else "in"
                    continue
                m = re.match(r"T(\d+).*?C([\d.]+)", u)
                if m:
                    self.herr[int(m.group(1))] = float(m.group(2))
                    continue
                if re.match(r"T\d+$", u) or re.match(r"[XY][+-]?[\d.]", u):
                    cab = False            # archivo sin '%' de cierre de cabecera
                else:
                    continue
            if u.startswith("M30") or u == "M00":
                break
            if u.startswith("M15"):
                bajo = True
                continue
            if u.startswith(("M16", "M17")):
                bajo = False
                continue
            m = re.match(r"T(\d+)(?:.*?C([\d.]+))?", u)
            if m and not re.match(r"[XY]", u):
                herr = int(m.group(1))
                if m.group(2):
                    self.herr[herr] = float(m.group(2))
                continue
            if u.startswith("G05"):
                rut = False
                continue
            if "G85" in u:                                  # ranura: X.Y.G85X.Y.
                a, b = u.split("G85", 1)
                ca, cb = dict(re.findall(r"([XY])([+-]?[\d.]+)", a)), dict(re.findall(r"([XY])([+-]?[\d.]+)", b))
                if "X" in ca and "Y" in ca and "X" in cb and "Y" in cb and herr is not None:
                    self.ranuras.append((ca["X"], ca["Y"], cb["X"], cb["Y"], herr))
                continue
            cs = dict(re.findall(r"([XY])([+-]?[\d.]+)", u))
            if not cs:
                if u.startswith("G00") or u.startswith("G01"):
                    rut = True
                continue
            nx, ny = cs.get("X", ult[0]), cs.get("Y", ult[1])
            if u.startswith("G00"):
                rut = True
            elif u.startswith("G01") and bajo and ult[0] is not None and herr is not None:
                self.ranuras.append((ult[0], ult[1], nx, ny, herr))
            elif not rut and herr is not None and nx is not None and ny is not None:
                self.taladros.append((nx, ny, herr))
            ult = [nx, ny]

    def candidatos(self):
        unis = [self.unidad] if self.unidad else ["mm", "in"]
        modos = ["omit_trailing", "omit_leading"] if self.ceros == "LZ" else ["omit_leading", "omit_trailing"]
        fmts = {"in": [(2, 4), (2, 3), (3, 3), (2, 5), (1, 4)], "mm": [(3, 3), (2, 4), (3, 2), (4, 2), (2, 3), (3, 4)]}
        out = []
        for u in unis:
            lista = list(fmts[u])
            if self.digitos:
                lista = [self.digitos] + [f for f in lista if f != self.digitos]
            for f in lista:
                for m in modos:
                    out.append((u, m, f))
        return out

    @staticmethod
    def _valor(s, modo, ni, nd):
        if "." in s:
            return float(s)
        neg = s.startswith("-")
        s = s.lstrip("+-")
        if modo == "omit_trailing":
            s = s.ljust(ni + nd, "0")
        v = (int(s) if s else 0) / (10.0 ** nd)
        return -v if neg else v

    def posiciones(self, unidad, modo, fmt):
        """(taladros, ranuras) en mm: [(x, y, diametro)], [(x0, y0, x1, y1, diametro)]."""
        f = 25.4 if unidad == "in" else 1.0
        ni, nd = fmt
        V = lambda s: self._valor(s, modo, ni, nd) * f
        hs = [(V(x), V(y), self.herr.get(t, 0.0) * f) for x, y, t in self.taladros]
        rs = [(V(a), V(b), V(c), V(d), self.herr.get(t, 0.0) * f) for a, b, c, d, t in self.ranuras]
        return hs, rs


def tipo_texto(t):
    """'gerber' | 'excellon' | None segun el contenido."""
    cab = t[:20000]
    if re.search(r"%FS[LTD]?[AI]?", cab) or re.search(r"%MO(IN|MM)", cab) or re.search(r"%ADD\d+", cab):
        return "gerber"
    if re.search(r"^\s*M48\s*$", cab, re.M) or (re.search(r"^\s*(INCH|METRIC)", cab, re.M | re.I)
                                                 and re.search(r"^\s*T\d+.*C[\d.]+", cab, re.M)):
        return "excellon"
    return None


def rol_capa(nombre, texto):
    """Deduce el papel de un Gerber por su nombre y sus comentarios G04."""
    comentarios = " ".join(re.findall(r"G0?4([^*]*)\*", texto[:3000]))
    s = (nombre + " " + comentarios).lower().replace("_", " ").replace("-", " ")
    if re.search(r"edge|outline|contorno|\bgko\b|\.gm1|mechanical|\bmech\b", s):
        return "borde"
    if re.search(r"silk|legend|resist|mask|paste|assembl|serigraf|\.gto|\.gbo|\.gts|\.gbs", s):
        return "otra"
    if re.search(r"bottom copper|bottom cu|\bb cu\b|\.gbl|back copper|cobre inferior|bot copper|copper bottom|bottom layer", s):
        return "cobre_inf"
    if re.search(r"top copper|top cu|\bf cu\b|\.gtl|front copper|cobre superior|copper top|top layer", s):
        return "cobre_sup"
    if re.search(r"inner|internal", s):
        return "interna"
    return None


EXT_GERBER = {".gbr", ".ger", ".gbl", ".gtl", ".gbo", ".gto", ".gbs", ".gts", ".gko", ".gm1", ".art",
              ".pho", ".grb", ".txt", ".gbx", ".drl", ".xln", ".drd", ".exc", ".ncd"}


def _decodificar(b):
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return b.decode("latin-1")


def expandir_zip_bytes(datos):
    """{nombre: texto} con los archivos de un ZIP (en memoria)."""
    import io
    import zipfile
    out = {}
    try:
        with zipfile.ZipFile(io.BytesIO(datos)) as z:
            for info in z.infolist():
                if not info.is_dir() and info.file_size < 60e6:
                    out[os.path.basename(info.filename)] = _decodificar(z.read(info))
    except zipfile.BadZipFile:
        raise ErrorUsuario("El ZIP esta danado o no es un ZIP valido.")
    return out


def reunir_archivos(ruta, a):
    """Devuelve {nombre: texto} desde un ZIP, una carpeta o un archivo suelto (mas sus taladros vecinos)."""
    out = {}
    if os.path.isdir(ruta):
        for nom in sorted(os.listdir(ruta)):
            p = os.path.join(ruta, nom)
            if os.path.isfile(p) and os.path.splitext(nom)[1].lower() in EXT_GERBER and os.path.getsize(p) < 60e6:
                with open(p, "rb") as f:
                    out[nom] = _decodificar(f.read())
    elif ruta.lower().endswith(".zip"):
        with open(ruta, "rb") as f:
            out.update(expandir_zip_bytes(f.read()))
    else:
        with open(ruta, "rb") as f:
            out[os.path.basename(ruta)] = _decodificar(f.read())
        carpeta = os.path.dirname(os.path.abspath(ruta))            # buscar taladros junto al archivo
        for nom in sorted(os.listdir(carpeta)):
            p = os.path.join(carpeta, nom)
            if nom not in out and os.path.isfile(p) and os.path.splitext(nom)[1].lower() in EXT_GERBER \
                    and os.path.getsize(p) < 60e6:
                with open(p, "rb") as f:
                    t = _decodificar(f.read())
                if tipo_texto(t) == "excellon":
                    out[nom] = t
    if a.taladros:
        if not os.path.isfile(a.taladros):
            raise ErrorUsuario("No encuentro el archivo de taladros: %s" % a.taladros)
        with open(a.taladros, "rb") as f:
            out[os.path.basename(a.taladros)] = _decodificar(f.read())
    return out


def leer_gerber(ruta, a):
    return leer_gerber_desde(reunir_archivos(ruta, a), a, ruta)


def leer_gerber_desde(archivos, a, origen="los archivos subidos"):
    """archivos: {nombre: texto}. Opciones opcionales en `a`: capa_cobre, archivo_borde, archivos_taladro."""
    ent = Entrada("Gerber")
    forzados = set(getattr(a, "archivos_taladro", None) or [])
    gerbers = {n: t for n, t in archivos.items() if n not in forzados and tipo_texto(t) == "gerber"}
    excellons = {n: t for n, t in archivos.items() if n in forzados or tipo_texto(t) == "excellon"}
    if not gerbers:
        raise ErrorUsuario("No encontre archivos Gerber (RS-274X) en: %s" % origen)
    roles = {n: rol_capa(n, t) for n, t in gerbers.items()}

    elegida, rol_el = None, None
    if a.capa_cobre:
        cand = [n for n in gerbers if n == a.capa_cobre] or [n for n in gerbers if a.capa_cobre.lower() in n.lower()]
        if not cand:
            raise ErrorUsuario("Ningun Gerber contiene '%s' en su nombre. Archivos: %s" % (a.capa_cobre, ", ".join(gerbers)))
        elegida, rol_el = cand[0], roles[cand[0]]
    else:
        for rol in ("cobre_inf", "cobre_sup"):
            cand = [n for n in gerbers if roles[n] == rol]
            if cand:
                elegida, rol_el = cand[0], rol
                if len(cand) > 1:
                    ent.avisos.append("Hay %d archivos de '%s'; se usa %s." % (len(cand), rol, elegida))
                break
        if elegida is None and len(gerbers) == 1:
            elegida = next(iter(gerbers))
    if elegida is None:
        raise ErrorUsuario("No pude saber cual Gerber es el cobre. Archivos: %s. Indica parte del nombre con "
                           "--capa-cobre (ej. --capa-cobre \"Bottom Copper\")." % "; ".join(gerbers))
    ent.espejo_sugerido = rol_el != "cobre_sup"

    ent.resumen.append("Archivos encontrados:")
    for n in gerbers:
        marca = "  <- COBRE USADO" if n == elegida else ""
        ent.resumen.append("   %-55s Gerber (%s)%s" % (n, roles[n] or "sin clasificar", marca))
    for n in excellons:
        ent.resumen.append("   %-55s Taladros (Excellon)" % n)

    g = Gerber(gerbers[elegida], elegida)
    ent.ops = g.ops
    ent.avisos.extend(g.avisos)
    if not g.ops:
        raise ErrorUsuario("El Gerber de cobre no contiene dibujo.")
    ent.resumen.append("Cobre (%s): %d trazos, %d pads/flashes, %d regiones | unidades: %s"
                       % (elegida, g.n_trazos, g.n_flashes, g.n_regiones, "pulgadas" if g.u == 25.4 else "mm"))

    for n, t in gerbers.items():                       # contorno de placa: solo para dimensionar
        if (roles[n] == "borde" or n == getattr(a, "archivo_borde", None)) and n != elegida:
            ge = Gerber(t, n)
            geom = componer(ge.ops)
            if not geom.is_empty:
                ent.borde.append(geom)
                ent.resumen.append("Contorno de placa: %s (se usa para dimensionar el rectangulo)" % n)

    if excellons:
        cobre_aprox = componer(ent.ops)
        shapely.prepare(cobre_aprox)
        for n, t in excellons.items():
            ex = Excellon(t, n)
            if not ex.taladros and not ex.ranuras:
                continue
            mejor, mejor_pt = None, -1
            for cand in ex.candidatos():
                hs, rs = ex.posiciones(*cand)
                pts = [(h[0], h[1]) for h in hs] + [((r[0] + r[2]) / 2, (r[1] + r[3]) / 2) for r in rs]
                if not pts:
                    continue
                dentro = int(shapely.contains(cobre_aprox, shapely.points(pts)).sum())
                if dentro > mejor_pt:
                    mejor, mejor_pt = cand, dentro
            hs, rs = ex.posiciones(*mejor)
            total = len(hs) + len(rs)
            ent.agujeros.extend(hs)
            ent.ranuras.extend(rs)
            ent.resumen.append("Taladros (%s): %d agujeros, %d ranuras | formato %s, %s, %d.%d | %d de %d caen sobre cobre"
                               % (n, len(hs), len(rs), "pulgadas" if mejor[0] == "in" else "mm",
                                  "ceros iniciales omitidos" if mejor[1] == "omit_leading" else "ceros finales omitidos",
                                  mejor[2][0], mejor[2][1], mejor_pt, total))
            if mejor_pt < 0.5 * total:
                ent.avisos.append("Pocos taladros coinciden con el cobre (%d de %d). Puede que el archivo de taladros "
                                  "tenga otro origen/formato; revisa la vista previa." % (mejor_pt, total))
    ent.agujeros = list({(round(x, 3), round(y, 3), round(d, 3)): (x, y, d) for x, y, d in ent.agujeros}.values())
    ent.ranuras = list({tuple(round(v, 3) for v in r): r for r in ent.ranuras}.values())
    if not excellons:
        ent.avisos.append("No se encontro archivo de taladros (Excellon): los pads saldran completos, sin el circulo "
                          "del agujero. Incluye el archivo Drill del ZIP de ARES.")
    return ent


# --------------------------------------------------------------------------
# Construccion del negativo
# --------------------------------------------------------------------------
def construir_negativo(ent, a, espejo, margen, log):
    cobre = componer(ent.ops)
    if cobre.is_empty:
        raise ErrorUsuario("No se encontro cobre. Revisa los colores/capas con --listar "
                           "(puedes filtrar con --incluir / --excluir).")

    # ---- aviso: contorno de placa (marco fino) incluido en el cobre ----
    cx0, cy0, cx1, cy1 = cobre.bounds
    for p in poligonos(cobre):
        b = p.bounds
        if (b[2] - b[0]) >= 0.95 * (cx1 - cx0) and (b[3] - b[1]) >= 0.95 * (cy1 - cy0) and p.interiors:
            hmax = max((Polygon(r).bounds for r in p.interiors), key=lambda t: (t[2] - t[0]) * (t[3] - t[1]))
            if (hmax[2] - hmax[0]) >= 0.85 * (cx1 - cx0) and (hmax[3] - hmax[1]) >= 0.85 * (cy1 - cy0):
                log("AVISO: parece que el CONTORNO DE LA PLACA (un marco que rodea todo) quedo dentro del cobre. "
                    "En ARES desmarca 'Edge (will appear on all layers)' al generar el Gerber, o saldra un marco "
                    "protegido alrededor de la placa.")
                break

    # ---- perforaciones: se restan del cobre para que queden EXPUESTAS (circulo dentro de cada pad) ----
    huecos_u, centros = None, None
    if (ent.agujeros or ent.ranuras) and not a.sin_agujeros:
        huecos, diams = [], Counter()
        for x, y, d in ent.agujeros:
            dd = a.diam_agujero if a.diam_agujero > 0 else d + (a.holgura_agujero or 0.0)
            if dd > 0:
                huecos.append(Point(x, y).buffer(dd / 2.0, quad_segs=segs_cuarto(dd / 2.0)))
                diams[round(dd, 2)] += 1
        for x0, y0, x1, y1, d in ent.ranuras:
            dd = a.diam_agujero if a.diam_agujero > 0 else d + (a.holgura_agujero or 0.0)
            if dd > 0:
                huecos.append(LineString([(x0, y0), (x1, y1)]).buffer(dd / 2.0, quad_segs=segs_cuarto(dd / 2.0)))
                diams[round(dd, 2)] += 1
        if huecos:
            arr = np.array(huecos, dtype=object)
            shapely.prepare(cobre)
            dentro = shapely.contains(cobre, arr)                    # perforaciones rodeadas por cobre (pads)
            ctrs = [h.representative_point() for h, ok in zip(huecos, dentro) if ok]
            centros = unary_union(ctrs) if ctrs else None
            log("Perforaciones: %d (diametros en mm: %s) -> se dejan expuestas como circulos dentro de cada pad"
                % (len(huecos), ", ".join("%.2f x%d" % kv for kv in sorted(diams.items()))))
            if int(dentro.sum()) < len(huecos):
                log("AVISO: %d de %d perforaciones no quedan rodeadas de cobre (agujero sin pad, NPTH, o el pad es "
                    "mas chico que el taladro)." % (len(huecos) - int(dentro.sum()), len(huecos)))
            huecos_u = unary_union(huecos)
            cobre = cobre.difference(huecos_u)

    cobre = cobre.buffer(CIERRE, quad_segs=4).buffer(-CIERRE, quad_segs=4)      # cierra rendijas numericas
    if a.compensacion:
        cobre = cobre.buffer(a.compensacion, quad_segs=8)
    bordes = list(ent.borde)
    if espejo:
        esp = lambda g: affinity.scale(g, xfact=-1, yfact=1, origin=(0, 0))
        cobre = esp(cobre)
        bordes = [esp(b) for b in bordes]
        centros = esp(centros) if centros is not None else None
    minx, miny, maxx, maxy = unary_union([cobre] + bordes).bounds
    cw, ch = maxx - minx, maxy - miny
    m = margen
    rect = box(minx - m, miny - m, maxx + m, maxy + m)
    dx, dy = -rect.bounds[0], -rect.bounds[1]
    cobre = affinity.translate(cobre, dx, dy)
    rect = affinity.translate(rect, dx, dy)
    centros = affinity.translate(centros, dx, dy) if centros is not None else None

    # ---- trabajo 2: circulos de corte de las perforaciones, en el MISMO sistema de coordenadas ----
    kerf = a.kerf or 0.0
    perfo = {"circulos": [], "ranuras": []}

    def T(x, y):
        return (-x if espejo else x) + dx, y + dy

    for x, y, d in ent.agujeros:
        if d - kerf > 0.05:
            perfo["circulos"].append(T(x, y) + (d - kerf,))
    for x0, y0, x1, y1, d in ent.ranuras:
        if d - kerf > 0.05:
            perfo["ranuras"].append(T(x0, y0) + T(x1, y1) + (d - kerf,))
    fuente = cobre if a.positivo else rect.difference(cobre)      # positivo: se graba el cobre; negativo: el fondo
    if SIMPLIF > 0:
        fuente = fuente.simplify(SIMPLIF, preserve_topology=True)
    polys = []
    for p in poligonos(fuente):
        huecos_p = [h for h in p.interiors if Polygon(h).area >= AREA_MIN]
        q = Polygon(p.exterior, huecos_p)
        if q.area >= AREA_MIN:
            polys.append(orient(q, 1.0))
    if not polys:
        raise ErrorUsuario("El resultado salio vacio (revisa los archivos subidos o --listar).")
    W, H = rect.bounds[2], rect.bounds[3]
    log("Cobre: %.2f x %.2f mm | area de cobre %.1f mm2" % (cw, ch, cobre.area))
    log("Rectangulo a grabar: %.2f x %.2f mm (margen %.1f mm) | se graba el %.0f%% del rectangulo"
        % (W, H, m, 100.0 * sum(p.area for p in polys) / rect.area))
    if centros is not None and not a.positivo:
        pts = shapely.get_parts(centros)
        negu = unary_union(polys)
        shapely.prepare(negu)
        ok = int(shapely.contains(negu, pts).sum())
        log("Verificacion: %d de %d circulos de perforacion aparecen en el negativo (zona que el laser graba)"
            % (ok, len(pts)))
        if ok < len(pts):
            log("AVISO: algunas perforaciones se perdieron (muy chicas para la simplificacion o el cierre de rendijas).")
    if cobre.area > 0.9 * rect.area:
        log("AVISO: el cobre ocupa casi todo el rectangulo. Quiza entro el fondo o la placa completa: "
            "mira los colores con --listar y excluye con --excluir.")
    return polys, W, H, perfo


# --------------------------------------------------------------------------
# Escritura de archivos
# --------------------------------------------------------------------------
def anillos(p):
    return [p.exterior] + list(p.interiors)


_PLANTILLA_DXF_R12 = (
    "eNqVV99z2ygQftdf4Yd7uk48Av2y+2QM2GYsSypgx8lLJtPmZjrXSTpt7ub6398KkISQ0yQvAn18uwvLsiyzWRwpTrWoq2g2"
    "w9GOE8YldJfRH4QSdjI/KCIUxfHS4ux6S2vGG7LlACQRqZS4QzjDdlhUak0UDKE4iudxNMOuTVzbkvhZH0RlOOjhA7Ys10v6"
    "XsckZ8O8GqhXA/fKI5fi0Kkdme7GnKIUd6N4WXTjtdS7+gALi2YFiFlQ8i2vehBZcCPKMsSYJNsec574pNvJBxoPQsoW94Vr"
    "NYjG2SJxE9aKkhLgFIjdLInWoelWmxK3lojnmUMlofxasLG44eqbVuusiNTz/eOX+x9f7BgtyY3Z7kU3V8pLfdO03Dxa/yrv"
    "fz386AZoXdbAzXGEs9y5QBwuTBhQom7HcwOMn2uDxfMcDygrhUGTeTGAsmKOGntEPsH4uTPtadTNhKcPU6jdj2CGlFcTTLuV"
    "jGTr0t9fQCDSAkSLXYjUIaI4miA4lCJsFHTicNuGOyCLHlmXe3NkPffXgZDahUhTKx1IXYBKHcwHEDuhxIc2nY+SZVLEcbEo"
    "0sE3G0InAaLrTRlMSJ8ubJw4hx6qzyyESOhYcAgaLwQQPEa6M4EjoeorL35oKUMLAPEpFLpGX1roljRh1NO6lkz5q4cjzgTx"
    "1QEi+SefsyMVnJVq62PlsRJa+QmoPDaSU4O4HVB7rulOVMHU2ozGIWO4bNE7nQwqe6RX6YwceHU0zpy7hF3yE7EXir9/zQu4"
    "3gm6r7hSYxjOEN1xuvdt0x05bLgkY6ZD12NU7Zu6vPGlNaOSE+3SZJqjJMnnyyJepOkyLjrOsWGvckQFV2CwCnZUUkMOkmN8"
    "QL39rbb2gsw8IoBMSH++DQuvjob1aX5wK0RBn+aH1TflRpLDSFoZqs3ngOY+qvhW+WmkDS/FOWidZdGq7a0c/Sg3mqzRSIXF"
    "8AQLTQF2DIGTDxypqsys4Rz2SC23v68lgHM2vkMuol8g3XSkThMKSc10As1bZtC8aQrNm+ZwVFyK0U1gEDxBkgmSTpAsRCQK"
    "4xMwfAFLLmDpBSwbY9e1LNlJ8Gs/4NUOikq25f5NYTGxMVjRnSJR8rC6gZqNUHMbtHGS9lE/TQ/N24vK5l1VZfO7srJ5ra5s"
    "U2h4kk9CSa6Ju7uRd5S3vBqtSfVVoCPGEW/PJjVdv3SHI1hyZWDTNeCpqaX2ZN0/nGqcGMKf5PPz138fOpvhChFy8exahN24"
    "a1Hi/l2LUmghJrBrUeb+XYtyx3dtkju9hcNdm7jW3Eax7bZTSNoYxFFmEWc0TW3b5dMM2bZA7ZIKbFRAJ2mdALVIe3IyM5Sb"
    "b2G+tuo17gX/BX4s+2SWmJGuLgY/poaw/rX+9vT5737vkshYzjNjN/ZOyUg4c8Kutn6fcG6E6dPj89fHf57++fkm+cvLs5X/"
    "sDz3EgAjhSHEnW4o+AvzHvDNjiUWRoI9/PX96evj8893SDIjuaKkIStPbGUeG6tLwhdX44q5Pui74g4sLC2hf/dYG517Uhfl"
    "XSDZAIJ4s2+AJHr+7xnaNHrZdp/7fuNtuAdeo5CmEczbEPffLoFYAiXMU+KNWwU7AnXemtD9VtbHirnn2iX+xvD5LTtvXpvV"
    "UCj3vu0hFIOu9UX3mni0Xssis4vm9TmUoOB34+EUd5UxHG37BoSzbd9zaeb2yGWOtMsQC9Mi91JDThdKscOT4CWCjMp2JHOM"
    "3P0XnfU+ccTdAQK5eJI1kMkaiwiZZSKQggzTphv4JgZJzTcz39x8C/O9kGteyOrrsqZ7m9VN1/gQLfpXOoYb8unLw7c79f3+"
    "84tpvC8wkoDevoEgOyK3/yZR2jKZ3a1v7jwUzdN5NluBSpxfofgKY42Tj2jxMUbzJV4kCf4Qxx9jm6+jaym05tW7dCxjjJwO"
    "5xD7lIX1Lvv1jtxAfTc0998ffrzdDSO6fQqObTLP5gubwysttIBLd9X2GFQ0ahXQeb2J/gfcXJQa")    # cabecera/tablas de un DXF R12 generado con ezdxf (zlib+base64)


def _dxf_r12(ruta, entidades, capa, color):
    """DXF R12 (ASCII) sin librerias. entidades: ('poli', [(x, y), ...]) cerrada | ('circulo', x, y, r)."""
    import base64
    import zlib
    plantilla = zlib.decompress(base64.b64decode(_PLANTILLA_DXF_R12)).decode("ascii")
    h = [0x100]

    def nuevo():
        h[0] += 1
        return "%X" % h[0]

    def f(v):
        return repr(round(float(v), 4) + 0.0)

    out = []
    for e in entidades:
        if e[0] == "poli":
            out += ["  0", "POLYLINE", "  5", nuevo(), "  8", capa, " 66", "     1",
                    " 10", "0.0", " 20", "0.0", " 30", "0.0", " 70", "     1"]
            for x, y in e[1]:
                out += ["  0", "VERTEX", "  5", nuevo(), "  8", capa, " 10", f(x), " 20", f(y), " 30", "0.0",
                        " 70", "     0"]
            out += ["  0", "SEQEND", "  5", nuevo(), "  8", capa]
        else:
            out += ["  0", "CIRCLE", "  5", nuevo(), "  8", capa, " 10", f(e[1]), " 20", f(e[2]), " 30", "0.0",
                    " 40", f(e[3])]
    texto = (plantilla.replace("@CAPA@", capa).replace("@COLOR@", str(color)).replace("@SEED@", "%X" % (h[0] + 1))
             .replace("@ENTIDADES@", "\n".join(out) + ("\n" if out else "")))
    with open(ruta, "w", newline="\n") as fh:
        fh.write(texto)


def _hay_ezdxf():
    import importlib.util
    return importlib.util.find_spec("ezdxf") is not None


def escribir_dxf(ruta, polys, version):
    if version == "R12" or not _hay_ezdxf():
        _dxf_r12(ruta, [("poli", [tuple(c) for c in r.coords]) for p in polys for r in anillos(p)], "NEGATIVO", 5)
        return
    import ezdxf
    import logging
    logging.getLogger("ezdxf").setLevel(logging.ERROR)
    doc = ezdxf.new(version)
    doc.units = 4      # milimetros
    capa = "NEGATIVO"
    doc.layers.add(capa, color=5)
    msp = doc.modelspace()
    for p in polys:
        for r in anillos(p):
            pts = [(round(x, 4), round(y, 4)) for x, y in r.coords]     # el ultimo punto repite al primero
            msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": capa})
    doc.saveas(ruta)


def escribir_plt(ruta, polys):
    u = 40.0   # HPGL: 40 unidades por mm (0.025 mm)
    out = ["IN;", "SP1;"]
    for p in polys:
        for r in anillos(p):
            pts = [(int(round(x * u)), int(round(y * u))) for x, y in r.coords]
            out.append("PU%d,%d;" % pts[0])
            for i in range(1, len(pts), 40):
                out.append("PD" + ",".join("%d,%d" % q for q in pts[i:i + 40]) + ";")
    out += ["PU;", "SP0;", ""]
    with open(ruta, "w", newline="\r\n") as f:
        f.write("\n".join(out))


def escribir_svg(ruta, polys, W, H, contorno=False):
    partes = []
    for p in polys:
        for r in anillos(p):
            c = list(r.coords)[:-1]
            partes.append("M" + " L".join("%.3f %.3f" % (x, H - y) for x, y in c) + " Z")
    estilo = ('fill="none" stroke="#000000" stroke-width="0.05"' if contorno
              else 'fill="#000000" fill-rule="evenodd" stroke="none"')
    svg = ('<?xml version="1.0" encoding="UTF-8"?>\n'
           '<svg xmlns="http://www.w3.org/2000/svg" width="%.3fmm" height="%.3fmm" viewBox="0 0 %.3f %.3f">\n'
           '<path %s d="%s"/>\n</svg>\n'
           % (W, H, W, H, estilo, " ".join(partes)))
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(svg)


def escribir_png(ruta, polys, W, H, dpi, suave=True):
    """PNG a escala real (dpi en los metadatos). Negro = lo que graba el laser, blanco = lo que se protege.
    Con suave=True los bordes se suavizan por supermuestreo (4x4): mismo tamano en pixeles y misma escala."""
    from PIL import Image, ImageDraw
    s = dpi / 25.4
    wpx, hpx = int(math.ceil(W * s)), int(math.ceil(H * s))
    if wpx * hpx > 600e6:
        raise ErrorUsuario("El PNG seria demasiado grande (%dx%d px). Baja --dpi." % (wpx, hpx))
    orden = sorted(polys, key=lambda q: -Polygon(q.exterior).area)
    if not suave:
        img = Image.new("1", (wpx, hpx), 1)           # 1 = blanco (protegido), 0 = negro (el laser graba)
        d = ImageDraw.Draw(img)
        for p in orden:
            d.polygon([(x * s, (H - y) * s) for x, y in p.exterior.coords], fill=0)
            for h in p.interiors:
                d.polygon([(x * s, (H - y) * s) for x, y in h.coords], fill=1)
        img.save(ruta, dpi=(dpi, dpi), optimize=True)
        return wpx, hpx

    k = 4
    def en_px(r):
        x, y = r.xy
        return np.column_stack((np.asarray(x) * s * k, (H - np.asarray(y)) * s * k))

    capas = []                                         # (contorno, [(hueco, ymin, ymax)], ymin, ymax) en px supermuestreados
    for p in orden:
        e = en_px(p.exterior)
        hs = []
        for h in p.interiors:
            a = en_px(h)
            hs.append((a, a[:, 1].min(), a[:, 1].max()))
        capas.append((e, hs, e[:, 1].min(), e[:, 1].max()))
    img = Image.new("L", (wpx, hpx), 255)
    banda = max(32, int(25e6 / max(wpx * k * k, 1)))   # filas de salida por banda: limita la memoria
    for y0 in range(0, hpx, banda):
        y1 = min(hpx, y0 + banda)
        off, lim = y0 * k, y1 * k
        trabajo = Image.new("L", (wpx * k, (y1 - y0) * k), 255)
        d = ImageDraw.Draw(trabajo)
        for e, hs, ya, yb in capas:
            if yb < off - 2 or ya > lim + 2:
                continue
            c = e.copy()
            c[:, 1] -= off
            d.polygon(c.ravel().tolist(), fill=0)
            for hcur, ha, hb in hs:
                if hb < off - 2 or ha > lim + 2:
                    continue
                c = hcur.copy()
                c[:, 1] -= off
                d.polygon(c.ravel().tolist(), fill=255)
        img.paste(trabajo.reduce(k), (0, y0))
    img.save(ruta, dpi=(dpi, dpi), optimize=True)
    return wpx, hpx


def poligonos_perforacion(perfo):
    """Circulos (y ranuras) de corte como poligonos, para PNG, PLT y SVG."""
    out = []
    for x, y, d in perfo["circulos"]:
        out.append(Point(x, y).buffer(d / 2.0, quad_segs=max(8, segs_cuarto(d / 2.0))))
    for x0, y0, x1, y1, d in perfo["ranuras"]:
        out.append(LineString([(x0, y0), (x1, y1)]).buffer(d / 2.0, quad_segs=max(8, segs_cuarto(d / 2.0))))
    return [orient(p, 1.0) for p in out]


def escribir_dxf_perfo(ruta, perfo, version):
    """DXF de perforaciones: circulos verdaderos (entidad CIRCLE) en capa roja, para modo Cut."""
    ents = [("circulo", x, y, d / 2.0) for x, y, d in perfo["circulos"]]
    ents += [("poli", [tuple(c) for c in p.exterior.coords])
             for p in poligonos_perforacion({"circulos": [], "ranuras": perfo["ranuras"]})]
    if version == "R12" or not _hay_ezdxf():
        _dxf_r12(ruta, ents, "PERFORACIONES", 1)
        return
    import ezdxf
    import logging
    logging.getLogger("ezdxf").setLevel(logging.ERROR)
    doc = ezdxf.new(version)
    doc.units = 4
    capa = "PERFORACIONES"
    doc.layers.add(capa, color=1)
    msp = doc.modelspace()
    for e in ents:
        if e[0] == "circulo":
            msp.add_circle((round(e[1], 4), round(e[2], 4)), round(e[3], 4), dxfattribs={"layer": capa})
        else:
            msp.add_lwpolyline([(round(x, 4), round(y, 4)) for x, y in e[1]], close=True, dxfattribs={"layer": capa})
    doc.saveas(ruta)


def es_interactivo():
    try:
        return bool(sys.stdin) and sys.stdin.isatty()
    except Exception:
        return False


def preguntar(texto, defecto=""):
    try:
        r = input(texto).strip().replace(",", ".")
    except EOFError:
        return defecto
    return r if r else defecto


# --------------------------------------------------------------------------
# Programa principal
# --------------------------------------------------------------------------
def crear_parser():
    p = argparse.ArgumentParser(
        prog="ares_a_laser",
        description="Convierte la exportacion de Proteus ARES (Gerber/ZIP, PDF, SVG o DXF) en el negativo del cobre "
                    "para grabar el sticker con la cortadora laser. Las perforaciones quedan como circulos expuestos.")
    p.add_argument("entrada", nargs="?",
                   help="ZIP o carpeta de Gerber de ARES, un Gerber suelto (.TXT/.gbl), o un .pdf/.svg/.dxf")
    p.add_argument("-o", "--salida", help="carpeta o prefijo de salida (por defecto, junto al archivo de entrada)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--espejo", dest="espejo", action="store_true",
                   help="forzar espejo horizontal (cobre hacia arriba en la maquina)")
    g.add_argument("--sin-espejo", dest="espejo", action="store_false",
                   help="no espejar (usalo si ya espejaste en ARES)")
    p.set_defaults(espejo=None)       # None = automatico: espejo para cobre inferior, sin espejo para cobre superior
    p.set_defaults(archivo_borde=None, archivos_taladro=None)
    p.add_argument("--margen", type=float, default=None,
                   help="mm de margen alrededor del diseno (def. 2; 0.5 si hay contorno de placa)")
    p.add_argument("--compensacion", type=float, default=0.0,
                   help="mm que se ensancha el cobre para compensar el grosor del haz (ej. 0.05)")
    p.add_argument("--positivo", action="store_true",
                   help="invertir: el laser graba el COBRE (pistas y pads) en vez del fondo")
    p.add_argument("--solo-perforaciones", dest="solo_perfo", action="store_true",
                   help="entregar solo los archivos de perforaciones (trabajo 2), sin el trabajo completo")
    p.add_argument("--sin-agujeros", action="store_true",
                   help="no dejar expuestas las perforaciones (los pads quedan completos)")
    p.add_argument("--diam-agujero", type=float, default=0.0, metavar="MM",
                   help="forzar el diametro (mm) del circulo expuesto en vez del diametro del taladro")
    p.add_argument("--capa-cobre", metavar="TEXTO",
                   help="Gerber: parte del nombre del archivo de cobre a usar (def. 'Bottom Copper')")
    p.add_argument("--taladros", metavar="ARCHIVO", help="Gerber: archivo Excellon de taladros, si no esta junto al cobre")
    p.add_argument("--dpi", type=int, default=1000, help="resolucion del PNG (def. 1000)")
    p.add_argument("--sin-suavizado", action="store_true",
                   help="PNG de 1 bit con bordes sin suavizar (por defecto los bordes se suavizan; misma escala)")
    gp = p.add_mutually_exclusive_group()
    gp.add_argument("--perforaciones", dest="perfo", action="store_true",
                    help="generar tambien los archivos SOLO de las perforaciones (sin preguntar)")
    gp.add_argument("--sin-perforaciones", dest="perfo", action="store_false",
                    help="no generar ni preguntar por los archivos de perforaciones")
    p.set_defaults(perfo=None)
    p.add_argument("--kerf", type=float, default=None, metavar="MM",
                   help="ancho del corte del laser (mm): los circulos de corte se achican esto para que el agujero "
                        "final salga del diametro exacto")
    p.add_argument("--holgura-agujero", type=float, default=None, metavar="MM",
                   help="mm extra del circulo expuesto en el negativo respecto al taladro (deja material limpio "
                        "alrededor del corte)")
    p.add_argument("--formatos", nargs="+", choices=["dxf", "plt", "svg", "png"], default=["png", "dxf", "plt", "svg"])
    p.add_argument("--dxf-version", choices=["R12", "R2000"], default="R12",
                   help="version del DXF de salida (def. R12, la mas compatible)")
    p.add_argument("--unidades", choices=["auto", "mm", "cm", "in", "thou"], default="auto",
                   help="unidades del DXF de entrada (def. auto)")
    p.add_argument("--escala", type=float, default=1.0, help="factor de escala extra (def. 1)")
    p.add_argument("--incluir", nargs="+", metavar="X",
                   help="PDF/SVG: colores #rrggbb a usar | DXF: nombres (o parte) de capa de cobre")
    p.add_argument("--excluir", nargs="+", metavar="X", help="colores #rrggbb o capas a ignorar")
    p.add_argument("--borde", nargs="+", metavar="X",
                   help="color/capa del borde de placa: no es cobre, solo agranda el rectangulo")
    p.add_argument("--ancho-pista", type=float, default=0.0,
                   help="mm de grosor para lineas sin grosor (si las pistas vienen como linea central)")
    p.add_argument("--rellenar-contornos", action="store_true",
                   help="PDF/SVG: tratar contornos cerrados sin relleno como formas rellenas")
    p.add_argument("--respetar-huecos", action="store_true",
                   help="DXF: contornos anidados alternan relleno/hueco (zonas de cobre con aislamiento)")
    p.add_argument("--listar", action="store_true", help="solo diagnostico: no escribe archivos")
    p.add_argument("--pausa", action="store_true", help="espera Enter al terminar")
    p.add_argument("--version", action="version", version="ares_a_laser " + VERSION)
    return p


def elegir_lector(ruta):
    if os.path.isdir(ruta):
        return leer_gerber
    ext = os.path.splitext(ruta)[1].lower()
    if ext == ".pdf":
        return leer_pdf
    if ext == ".svg":
        return leer_svg
    if ext == ".dxf":
        return leer_dxf
    if ext == ".zip" or ext in EXT_GERBER:
        return leer_gerber
    return None


def generar_salidas(ent, a, base1, base2, espejo, margen, quiere, log):
    """Construye el trabajo y escribe los archivos. Devuelve (rutas, W, H, polys, perfo)."""
    polys, W, H, perfo = construir_negativo(ent, a, espejo, margen, lambda s: log("  " + s))
    log("  Espejo horizontal: %s" % ("SI (cobre hacia arriba en la maquina)" if espejo else "NO"))
    rutas = []
    suave = not a.sin_suavizado
    nv = sum(len(r.coords) for p in polys for r in anillos(p))
    log("  %s: %d poligonos, %d vertices" % ("Cobre (positivo)" if a.positivo else "Negativo", len(polys), nv))

    if not a.solo_perfo:
        log("Archivos generados (TRABAJO 1: %s):" % ("cobre en positivo" if a.positivo else "negativo del cobre"))
        if "png" in a.formatos:
            wpx, hpx = escribir_png(base1 + ".png", polys, W, H, a.dpi, suave)
            rutas.append(base1 + ".png")
            log("  %s.png   (%dx%d px a %d dpi, bordes %s; NEGRO = lo que graba el laser, BLANCO = lo que no)"
                % (base1, wpx, hpx, a.dpi, "suavizados" if suave else "sin suavizar"))
        if "dxf" in a.formatos:
            escribir_dxf(base1 + ".dxf", polys, a.dxf_version)
            rutas.append(base1 + ".dxf")
            log("  %s.dxf   (DXF %s, mm, origen en 0,0)" % (base1, a.dxf_version))
        if "plt" in a.formatos:
            escribir_plt(base1 + ".plt", polys)
            rutas.append(base1 + ".plt")
            log("  %s.plt   (HPGL, respaldo para RDWorks)" % base1)
        if "svg" in a.formatos:
            escribir_svg(base1 + ".svg", polys, W, H)
            rutas.append(base1 + ".svg")
            log("  %s.svg" % base1)

    if quiere:
        if not perfo["circulos"] and not perfo["ranuras"]:
            log("AVISO: con ese kerf no queda ninguna perforacion que cortar.")
        else:
            polis = poligonos_perforacion(perfo)
            log("")
            log("Archivos generados (TRABAJO 2: perforaciones, MISMO origen, orientacion y tamano que el trabajo 1):")
            if "png" in a.formatos:
                wpx, hpx = escribir_png(base2 + ".png", polis, W, H, a.dpi, suave)
                rutas.append(base2 + ".png")
                log("  %s.png   (%d circulos negros, %dx%d px; para modo Scan)" % (base2, len(polis), wpx, hpx))
            if "dxf" in a.formatos:
                escribir_dxf_perfo(base2 + ".dxf", perfo, a.dxf_version)
                rutas.append(base2 + ".dxf")
                log("  %s.dxf   (circulos verdaderos, capa roja; para modo Cut)" % base2)
            if "plt" in a.formatos:
                escribir_plt(base2 + ".plt", polis)
                rutas.append(base2 + ".plt")
                log("  %s.plt" % base2)
            if "svg" in a.formatos:
                escribir_svg(base2 + ".svg", polis, W, H, contorno=True)
                rutas.append(base2 + ".svg")
                log("  %s.svg" % base2)
            log("  Diametros de corte (mm): %s%s" % (
                ", ".join("%.2f x%d" % kv for kv in sorted(Counter(round(d, 2) for _, _, d in perfo["circulos"]).items())),
                (" | kerf compensado: %.2f mm" % a.kerf) if a.kerf else ""))
    log("")
    log("TAMANO REAL del trabajo: %.2f x %.2f mm  <- comprueba que RDWorks muestre esta medida" % (W, H))
    if quiere and not a.solo_perfo:
        log("Orden: 1) graba el trabajo 1 (Scan)  2) acido  3) coloca la placa EXACTAMENTE en la misma posicion "
            "y corta las perforaciones (trabajo 2).")
    return rutas, W, H, polys, perfo


def procesar(a):
    log = print
    ruta = a.entrada
    if not ruta or not os.path.exists(ruta):
        raise ErrorUsuario("No encuentro el archivo o carpeta: %s" % ruta)
    if a.margen is not None and a.margen < 0:
        raise ErrorUsuario("--margen no puede ser negativo.")
    lector = elegir_lector(ruta)
    if lector is None:
        raise ErrorUsuario("Formato no soportado (%s). Usa un ZIP/carpeta de Gerber, o .pdf, .svg, .dxf."
                           % os.path.splitext(ruta)[1])
    log("Leyendo %s ..." % os.path.basename(ruta.rstrip("/\\")))
    ent = lector(ruta, a)
    for linea in ent.resumen:
        log("  " + linea)
    for av in ent.avisos:
        log("  AVISO: " + av)
    if a.listar:
        return 0
    espejo = a.espejo if a.espejo is not None else ent.espejo_sugerido
    margen = a.margen if a.margen is not None else (0.5 if ent.borde else 2.0)

    # ---- archivos SOLO de las perforaciones (para cortarlas despues con el laser): preguntar si no se indico ----
    hay_perfo = bool(ent.agujeros or ent.ranuras)
    quiere = a.perfo
    if a.solo_perfo:
        quiere = True
    if quiere and not hay_perfo:
        log("  AVISO: no hay archivo de taladros (Excellon); no se pueden generar los archivos de perforaciones.")
        if a.solo_perfo:
            raise ErrorUsuario("Con --solo-perforaciones se necesita el archivo de taladros (Excellon).")
    if hay_perfo and quiere is None and es_interactivo():
        print()
        r = preguntar("Quieres tambien los archivos SOLO de las perforaciones (circulos para cortar los agujeros "
                      "con el laser despues del acido)? [s/N]: ")
        quiere = r.lower().startswith(("s", "y"))
        if quiere:
            try:
                if a.kerf is None:
                    a.kerf = float(preguntar("  Ancho del corte del laser (kerf) en mm, para que el agujero salga "
                                             "del diametro exacto [0.1]: ", "0.1"))
                if a.holgura_agujero is None:
                    a.holgura_agujero = float(preguntar("  Holgura extra del circulo expuesto en el negativo (mm), "
                                                        "deja material limpio alrededor del corte [0.1]: ", "0.1"))
            except ValueError:
                raise ErrorUsuario("Numero no valido.")
        print()
    quiere = bool(quiere) and hay_perfo

    sufijo = "_positivo" if a.positivo else "_negativo"
    if a.salida and not os.path.isdir(a.salida):
        stem = os.path.splitext(a.salida)[0]            # -o PREFIJO: se usa tal cual
        sufijo = ""
    elif a.salida:
        stem = os.path.join(a.salida, os.path.splitext(os.path.basename(ruta.rstrip("/\\")))[0])
    elif os.path.isdir(ruta):
        stem = ruta.rstrip("/\\")
    else:
        stem = os.path.splitext(ruta)[0]
    generar_salidas(ent, a, stem + sufijo, stem + "_perforaciones", espejo, margen, quiere, log)
    return 0


def asistente():
    print("=== Proteus ARES -> laser: negativo del cobre ===")
    ruta = input("Arrastra aqui el ZIP de Gerber (o PDF/SVG/DXF) y pulsa Enter: ").strip().strip('"').strip("'")
    esp = input("Espejo horizontal? Enter = automatico (si para cobre inferior), n = no, s = si: ").strip().lower()
    mar = input("Margen alrededor del diseno en mm [automatico]: ").strip().replace(",", ".")
    argv = [ruta]
    if esp.startswith("n"):
        argv.append("--sin-espejo")
    elif esp.startswith("s"):
        argv.append("--espejo")
    if mar:
        argv += ["--margen", mar]
    return crear_parser().parse_args(argv)


# --------------------------------------------------------------------------
# Punto de entrada para la version web (Pyodide, dentro del navegador)
# --------------------------------------------------------------------------
def web_convertir(dir_in, dir_out, opciones_json):
    """Lee los archivos que indican las opciones (guardados en dir_in), escribe los resultados en dir_out y
    devuelve un JSON con el registro, la lista de archivos y una vista previa."""
    import json
    import traceback
    import zipfile
    lineas = []

    def log(s):
        lineas.append(s.replace(dir_out + os.sep, "").replace(dir_out + "/", ""))

    res = {"ok": False, "error": "", "log": "", "archivos": [], "vista": "", "ancho": 0, "alto": 0}
    try:
        o = json.loads(opciones_json)
        textos, forzados, cobre, borde = {}, [], None, None
        for it in o.get("archivos", []):
            with open(os.path.join(dir_in, it["id"]), "rb") as fh:
                datos = fh.read()
            if it["rol"] == "zip":
                textos.update(expandir_zip_bytes(datos))
                continue
            textos[it["nombre"]] = _decodificar(datos)
            if it["rol"] == "cobre":
                cobre = it["nombre"]
            elif it["rol"] == "borde":
                borde = it["nombre"]
            elif it["rol"] == "taladro":
                forzados.append(it["nombre"])
        if not textos:
            raise ErrorUsuario("Sube el ZIP de ARES o, como minimo, el Gerber de cobre.")

        a = crear_parser().parse_args(["web"])
        a.capa_cobre, a.archivo_borde, a.archivos_taladro = cobre, borde, forzados
        a.espejo = {"si": True, "no": False}.get(o.get("espejo"))
        a.positivo = o.get("que_graba") == "cobre"
        salida = o.get("salida", "completo")
        a.solo_perfo = salida == "perforaciones"
        quiere = salida in ("perforaciones", "ambos")
        a.margen = o.get("margen")
        a.compensacion = float(o.get("compensacion") or 0)
        a.dpi = int(o.get("dpi") or 1000)
        a.sin_suavizado = not o.get("suavizado", True)
        a.sin_agujeros = bool(o.get("sin_agujeros"))
        a.diam_agujero = float(o.get("diam_agujero") or 0)
        a.holgura_agujero = float(o.get("holgura") or 0)
        a.kerf = float(o.get("kerf") or 0)
        a.formatos = [f for f in o.get("formatos", ["png", "dxf"]) if f in ("png", "dxf", "plt", "svg")] or ["png"]
        a.dxf_version = "R12"
        if a.margen is not None and a.margen < 0:
            raise ErrorUsuario("El margen no puede ser negativo.")

        ent = leer_gerber_desde(textos, a)
        for linea in ent.resumen:
            log("  " + linea)
        for av in ent.avisos:
            log("  AVISO: " + av)
        hay_perfo = bool(ent.agujeros or ent.ranuras)
        if quiere and not hay_perfo:
            log("  AVISO: no hay archivo de taladros (Excellon); no se pueden generar los archivos de perforaciones.")
            if a.solo_perfo:
                raise ErrorUsuario("Para 'solo perforaciones' hace falta el archivo de taladros (Drill / Excellon).")
        quiere = quiere and hay_perfo
        espejo = a.espejo if a.espejo is not None else ent.espejo_sugerido
        margen = a.margen if a.margen is not None else (0.5 if ent.borde else 2.0)

        os.makedirs(dir_out, exist_ok=True)
        stem = re.sub(r"[^\w.-]+", "_", os.path.splitext(cobre or "placa")[0]).strip(" ._") or "placa"
        if cobre is None and ent.resumen:
            m = re.search(r"Cobre \((.+?)\):", " ".join(ent.resumen))
            if m:
                stem = re.sub(r"[^\w.-]+", "_", os.path.splitext(m.group(1))[0]).strip(" ._") or "placa"
        sufijo = "_positivo" if a.positivo else "_negativo"
        rutas, W, H, polys, perfo = generar_salidas(ent, a, os.path.join(dir_out, stem + sufijo),
                                                    os.path.join(dir_out, stem + "_perforaciones"),
                                                    espejo, margen, quiere, log)
        # vista previa (no se descarga): lo que graba el laser, negro sobre blanco
        prev = poligonos_perforacion(perfo) if a.solo_perfo else polys
        dpi_prev = max(40.0, min(200.0, 1400.0 / (W / 25.4)))
        escribir_png(os.path.join(dir_out, "_vista.png"), prev, W, H, dpi_prev, True)
        res["vista"] = "_vista.png"
        if len(rutas) > 1:
            zr = os.path.join(dir_out, stem + "_resultado.zip")
            with zipfile.ZipFile(zr, "w", zipfile.ZIP_DEFLATED) as z:
                for r in rutas:
                    z.write(r, os.path.basename(r))
            rutas.append(zr)
        mimes = {"png": "image/png", "dxf": "application/dxf", "plt": "application/vnd.hp-HPGL",
                 "svg": "image/svg+xml", "zip": "application/zip"}
        res["archivos"] = [{"nombre": os.path.basename(r), "mime": mimes.get(r.rsplit(".", 1)[-1], "application/octet-stream")}
                           for r in rutas]
        res["ancho"], res["alto"], res["ok"] = round(W, 2), round(H, 2), True
    except ErrorUsuario as e:
        res["error"] = str(e)
    except Exception as e:      # fallo inesperado: se muestra el detalle para poder diagnosticarlo
        res["error"] = "Error inesperado: %s\n%s" % (e, traceback.format_exc()[-1500:])
    res["log"] = "\n".join(lineas)
    return json.dumps(res)


def main(argv=None):
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    interactivo = not (argv if argv is not None else sys.argv[1:])
    codigo = 0
    pausa = False
    try:
        a = asistente() if interactivo else crear_parser().parse_args(argv)
        pausa = interactivo or a.pausa
        codigo = procesar(a)
    except ErrorUsuario as e:
        print("\nERROR: %s" % e)
        codigo = 1
    except KeyboardInterrupt:
        codigo = 130
    except Exception as e:      # fallo inesperado: mostrar el detalle para poder diagnosticarlo
        import traceback
        traceback.print_exc()
        print("\nERROR inesperado: %s\nSi no entiendes el mensaje, ejecuta con --listar y mandame la salida." % e)
        codigo = 2
    if pausa and sys.stdin and sys.stdin.isatty():
        input("\nPulsa Enter para salir...")
    return codigo


if __name__ == "__main__":
    sys.exit(main())
