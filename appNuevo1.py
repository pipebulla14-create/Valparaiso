import streamlit as st
import geopandas as gpd
import folium
from folium import plugins
from streamlit_folium import st_folium
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import io, base64
from PIL import Image

# ─────────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="GeoVisualizador Valparaíso", layout="wide")
st.title("🔥 GeoVisualizador de Severidad de Incendios - Valparaíso 2024")
st.write("Análisis territorial de daños utilizando imágenes satelitales y datos oficiales.")

DATA = Path("data")

# ─────────────────────────────────────────────────────────────
# PASO 1: Paletas cartográficas
# ─────────────────────────────────────────────────────────────

COLORES_USO_SUELO = {
    "Áreas Urbanas e Industriales": "#E60000",
    "Terrenos Agrícolas": "#FFAA00",
    "Praderas y Matorrales": "#E6E600",
    "Bosques": "#38A800",
    "Áreas Desprovistas de Vegetación": "#CCCCCC",
    "Cuerpos de Agua": "#004DA8"
}

# Especies de la capa Plantación (valores reales confirmados en especie_t)
COLORES_ESPECIES = {
    "Eucalyptus globulus": "#2E7D32",
    "Pinus radiata": "#F9A825",
    "Quillaja saponaria": "#6D4C41",
    "Especies nativas": "#1B5E20",
    "Otras especies": "#9E9E9E",
    "Acacia spp.": "#8D6E63",
}

# Severidad de incendio (dNBR clasificado): 1=Baja, 2=Moderada, 3=Alta
COLORES_SEVERIDAD = {
    1: (255, 255, 0),    # Baja - amarillo
    2: (255, 153, 0),    # Moderada - naranjo
    3: (255, 0, 0),      # Alta - rojo
}
ETIQUETAS_SEVERIDAD = {1: "Baja (0.10–0.27)", 2: "Moderada (0.27–0.44)", 3: "Alta (≥ 0.44)"}

# Encoding específico por shapefile (Red_Vial1 viene en cp1252, el resto en latin-1)
ENCODING_POR_ARCHIVO = {
    "if_magnitud_2023_2024": "latin-1",
    "Plantacion": "latin-1",
    "Red_Vial1": "cp1252",
}

ESTILOS_TIPO_TRANSPORTE = {
    0: {"color": "#CC0000", "weight": 4},
    1: {"color": "#CC0000", "weight": 4},
    2: {"color": "#E05000", "weight": 3},
    3: {"color": "#FF8800", "weight": 2.5},
    5: {"color": "#DAA520", "weight": 2},
    7: {"color": "#8B6914", "weight": 1.5},
    9: {"color": "#999999", "weight": 1},
}
ESTILO_VIAL_DEFAULT = {"color": "#888888", "weight": 1}

# ─────────────────────────────────────────────────────────────
# PASO 2: Funciones de color dinámico
# ─────────────────────────────────────────────────────────────

def crear_style_categorico(col, paleta, color_default="#555555"):
    """Estilo genérico: colorea por cualquier columna categórica y su paleta."""
    def style_fn(feature):
        val = str(feature["properties"].get(col, ""))
        color = paleta.get(val, color_default)
        return {"fillColor": color, "color": "#333333", "weight": 0.5, "fillOpacity": 0.65}
    return style_fn

def crear_style_uso_suelo(col):
    def style_fn(feature):
        val = str(feature["properties"].get(col, ""))
        color = COLORES_USO_SUELO.get(val, "#555555")
        return {"fillColor": color, "color": "#333333", "weight": 0.5, "fillOpacity": 0.65}
    return style_fn

def crear_style_transporte(col_tipo):
    def style_fn(feature):
        raw = feature["properties"].get(col_tipo)
        try:
            codigo = int(raw)
        except (TypeError, ValueError):
            codigo = None
        vals = ESTILOS_TIPO_TRANSPORTE.get(codigo, ESTILO_VIAL_DEFAULT)
        return {"color": vals["color"], "weight": vals["weight"], "opacity": 0.9}
    return style_fn

def crear_style_perimetro():
    return lambda x: {'color': '#FF0000', 'weight': 3, 'fillOpacity': 0.05, 'dashArray': '5, 5'}

# ─────────────────────────────────────────────────────────────
# PASO 3: Funciones de leyenda HTML
# ─────────────────────────────────────────────────────────────

def leyenda_categorica_html(titulo, color_map, icono="🔲", posicion_top="10px", posicion_right="10px"):
    items = ""
    for etiqueta, color in color_map.items():
        items += f"""
        <div style="display:flex;align-items:center;margin:3px 0;">
          <div style="background:{color};width:16px;height:16px;border:1px solid #555;margin-right:7px;border-radius:2px;flex-shrink:0;"></div>
          <span style="font-size:11px;color:#222;">{etiqueta}</span>
        </div>"""
    return f"""
    <div style="position:fixed;top:{posicion_top};right:{posicion_right};z-index:1000;background:rgba(255,255,255,0.93);padding:10px 14px;border-radius:8px;border:1px solid #bbb;box-shadow:2px 2px 6px rgba(0,0,0,0.25);font-family:Arial, sans-serif;">
      <b style="font-size:12px;">{icono} {titulo}</b><hr style="margin:5px 0;border-color:#ddd;">{items}
    </div>"""

# ─────────────────────────────────────────────────────────────
# PASO 4: Procesamiento de Imágenes Satelitales (RGB) y Severidad (1 banda)
# ─────────────────────────────────────────────────────────────

def _reproyectar_a_4326(src, n_bandas):
    """Reproyecta n_bandas del raster a EPSG:4326. Devuelve (data, bounds_wgs84)."""
    if src.crs and src.crs.to_epsg() != 4326:
        transform, width, height = calculate_default_transform(
            src.crs, "EPSG:4326", src.width, src.height, *src.bounds
        )
        data = np.zeros((n_bandas, height, width), dtype=np.float32)
        for i in range(1, n_bandas + 1):
            reproject(
                source=rasterio.band(src, i), destination=data[i - 1],
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=transform, dst_crs="EPSG:4326", resampling=Resampling.nearest
            )
        bounds_wgs84 = rasterio.transform.array_bounds(height, width, transform)
    else:
        data = src.read(list(range(1, n_bandas + 1))).astype(np.float32)
        bounds_wgs84 = src.bounds
    return data, bounds_wgs84


def procesar_raster_rgb(raster_path):
    """Convierte un raster multiespectral (>=3 bandas) en imagen RGB renderizable para Folium."""
    with rasterio.open(raster_path) as src:
        n_bandas = min(3, src.count)
        data, bounds_wgs84 = _reproyectar_a_4326(src, n_bandas)

        rgb = np.zeros((data.shape[1], data.shape[2], 4), dtype=np.uint8)
        for i in range(3):
            banda = data[i] if i < data.shape[0] else data[-1]
            min_val, max_val = np.percentile(banda[banda > 0], (2, 98)) if np.any(banda > 0) else (0, 1)
            rgb[..., i] = np.clip(255 * (banda - min_val) / (max_val - min_val + 1e-5), 0, 255)

        rgb[..., 3] = np.where(np.max(data, axis=0) > 0, 255, 0)

        img_pil = Image.fromarray(rgb)
        buf = io.BytesIO()
        img_pil.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        bounds = [[bounds_wgs84[1], bounds_wgs84[0]], [bounds_wgs84[3], bounds_wgs84[2]]]

        return img_b64, bounds


def procesar_raster_severidad(raster_path):
    """Convierte un raster de 1 banda clasificado (1=Baja,2=Moderada,3=Alta) en imagen
    renderizable para Folium, usando la paleta de severidad (no estiramiento RGB)."""
    with rasterio.open(raster_path) as src:
        data, bounds_wgs84 = _reproyectar_a_4326(src, 1)
        clase = data[0]

        rgba = np.zeros((clase.shape[0], clase.shape[1], 4), dtype=np.uint8)
        for valor, color in COLORES_SEVERIDAD.items():
            mask = np.isclose(clase, valor)
            rgba[mask, 0] = color[0]
            rgba[mask, 1] = color[1]
            rgba[mask, 2] = color[2]
            rgba[mask, 3] = 200
        # fuera de rango / nodata queda transparente (alpha=0 por defecto)

        img_pil = Image.fromarray(rgba)
        buf = io.BytesIO()
        img_pil.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        bounds = [[bounds_wgs84[1], bounds_wgs84[0]], [bounds_wgs84[3], bounds_wgs84[2]]]

        return img_b64, bounds


def procesar_raster_continuo(raster_path, cmap_name="terrain", nodata_override=None):
    """Convierte un raster de 1 banda CONTINUA (ej. DEM de elevación) en imagen
    renderizable para Folium, usando una rampa de color (colormap de matplotlib)."""
    import matplotlib
    with rasterio.open(raster_path) as src:
        nodata = nodata_override if nodata_override is not None else src.nodata
        data, bounds_wgs84 = _reproyectar_a_4326(src, 1)
        banda = data[0]

        valido = banda != nodata if nodata is not None else np.ones_like(banda, dtype=bool)
        valido &= ~np.isnan(banda)

        if not np.any(valido):
            raise ValueError("El raster no tiene píxeles válidos (todo nodata).")

        vmin, vmax = np.percentile(banda[valido], (2, 98))
        norm = np.clip((banda - vmin) / (vmax - vmin + 1e-9), 0, 1)

        colormap = matplotlib.colormaps[cmap_name]
        rgba_float = colormap(norm)  # (H, W, 4) en [0,1]
        rgba = (rgba_float * 255).astype(np.uint8)
        rgba[~valido, 3] = 0  # transparente donde no hay dato

        img_pil = Image.fromarray(rgba)
        buf = io.BytesIO()
        img_pil.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        bounds = [[bounds_wgs84[1], bounds_wgs84[0]], [bounds_wgs84[3], bounds_wgs84[2]]]

        return img_b64, bounds, float(vmin), float(vmax)

# ─────────────────────────────────────────────────────────────
# Cargar archivos (Lectura dinámica)
# ─────────────────────────────────────────────────────────────

archivos_vec = list(DATA.glob("*.gpkg")) + list(DATA.glob("*.shp")) + list(DATA.glob("*.geojson"))
capas = {}
for archivo in archivos_vec:
    nombre = archivo.stem.replace("_", " ")
    enc = ENCODING_POR_ARCHIVO.get(archivo.stem, "utf-8")
    try:
        gdf = gpd.read_file(archivo, encoding=enc)
        if gdf.crs is not None:
            gdf = gdf.to_crs(4326)
        capas[nombre] = gdf
    except Exception as e:
        st.warning(f"No fue posible cargar {archivo.name}: {e}")

archivos_raster = list(DATA.glob("*.tif")) + list(DATA.glob("*.tiff"))
rasters = {archivo.stem.replace("_", " "): archivo for archivo in archivos_raster}

# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────

st.sidebar.title("Control de Capas")
st.sidebar.subheader("🗺️ Capas Vectoriales")
capas_activas = [n for n in capas if st.sidebar.checkbox(n, value=True, key=f"vec_{n}")]

st.sidebar.subheader("🛰️ Imágenes Satelitales")
rasters_activos = [n for n in rasters if st.sidebar.checkbox(n, value=False, key=f"rst_{n}")]

# ─────────────────────────────────────────────────────────────
# Mapa base
# ─────────────────────────────────────────────────────────────

centro = [-33.08, -71.48]
m = folium.Map(location=centro, zoom_start=11, tiles="OpenStreetMap")
folium.TileLayer("CartoDB positron", name="Mapa claro").add_to(m)
folium.TileLayer("CartoDB dark_matter", name="Mapa oscuro").add_to(m)

# ─────────────────────────────────────────────────────────────
# PASO 5: Agregar Rasters y Comparador (SideBySide)
# ─────────────────────────────────────────────────────────────
capas_raster_folium = {}
leyendas_html = []
offset_top = 10

for nombre in rasters_activos:
    nombre_low = nombre.lower()
    try:
        with st.spinner(f"Procesando {nombre}..."):
            # Bifurcación clave: DEM (continuo) vs. severidad (1 banda clasificada) vs. satelital (RGB)
            if "dem" in nombre_low:
                img_b64, bounds, vmin, vmax = procesar_raster_continuo(rasters[nombre], cmap_name="terrain")
                icono = "⛰️"
                leyendas_html.append(
                    leyenda_categorica_html(
                        f"Elevación (DEM, {vmin:.0f}–{vmax:.0f} m)",
                        {"Bajo": "#2c7a3f", "Medio": "#c9b458", "Alto": "#8a5a3a", "Muy alto": "#ffffff"},
                        "⛰️", f"{offset_top}px", "10px"
                    )
                )
                offset_top += 160
            elif "severidad" in nombre_low:
                img_b64, bounds = procesar_raster_severidad(rasters[nombre])
                icono = "🔥"
                leyendas_html.append(
                    leyenda_categorica_html(
                        "Severidad del incendio",
                        {ETIQUETAS_SEVERIDAD[k]: f"rgb{v}" for k, v in COLORES_SEVERIDAD.items()},
                        "🔥", f"{offset_top}px", "10px"
                    )
                )
                offset_top += 140
            else:
                img_b64, bounds = procesar_raster_rgb(rasters[nombre])
                icono = "🛰"

            img_layer = folium.raster_layers.ImageOverlay(
                image=f"data:image/png;base64,{img_b64}",
                bounds=bounds,
                opacity=1.0,
                name=f"{icono} {nombre}",
            )
            img_layer.add_to(m)
            capas_raster_folium[nombre_low] = img_layer
    except Exception as e:
        st.sidebar.error(f"Error procesando {nombre}: {e}")

# Funcionalidad Avanzada: Comparador Automático si "pre" y "post" están activos
clave_pre = next((k for k in capas_raster_folium if "pre" in k), None)
clave_post = next((k for k in capas_raster_folium if "post" in k), None)

if clave_pre and clave_post:
    plugins.SideBySideLayers(
        layer_left=capas_raster_folium[clave_pre],
        layer_right=capas_raster_folium[clave_post]
    ).add_to(m)
    st.info("💡 **Modo Comparador activado:** Desliza la barra central en el mapa para comparar el Antes y el Después.")

# ─────────────────────────────────────────────────────────────
# PASO 6: Agregar Vectores con Estilos Específicos
# ─────────────────────────────────────────────────────────────

for nombre in capas_activas:
    gdf = capas[nombre]
    nombre_low = nombre.lower()

    # ── Plantación (uso de suelo forestal) ────────────────────
    if "plantacion" in nombre_low:
        col = "especie_t" if "especie_t" in gdf.columns else None
        if col:
            folium.GeoJson(
                gdf, name=f"🌲 {nombre}",
                style_function=crear_style_categorico(col, COLORES_ESPECIES),
                tooltip=folium.GeoJsonTooltip(
                    fields=["especie_t", "sup_ha"],
                    aliases=["Especie:", "Superficie (ha):"]
                )
            ).add_to(m)
            especies = sorted(gdf["especie_t"].dropna().unique())
            paleta_especies = {esp: COLORES_ESPECIES.get(esp, "#9E9E9E") for esp in especies}
            leyendas_html.append(leyenda_categorica_html("Plantación forestal", paleta_especies, "🌲", f"{offset_top}px", "10px"))
            offset_top += 40 + 20 * len(especies)
        else:
            folium.GeoJson(gdf, name=nombre).add_to(m)

    # ── Red Vial ─────────────────────────────────────────────
    elif "vial" in nombre_low or "red" in nombre_low:
        folium.GeoJson(
            gdf, name=f"🛣️ {nombre}",
            style_function=crear_style_transporte("Clase_Ruta"),
            tooltip=folium.GeoJsonTooltip(
                fields=["Nom_Ruta", "Clase_Ruta"],
                aliases=["Nombre ruta:", "Clase:"]
            )
        ).add_to(m)
        # Leyenda dinámica: solo los códigos de vía que realmente existen en tus datos
        codigos_presentes = sorted(gdf["Clase_Ruta"].dropna().astype(int).unique())
        paleta_vial = {}
        for codigo in codigos_presentes:
            estilo = ESTILOS_TIPO_TRANSPORTE.get(codigo, ESTILO_VIAL_DEFAULT)
            paleta_vial[f"Clase {codigo}"] = estilo["color"]
        leyendas_html.append(leyenda_categorica_html("Red vial (por clase)", paleta_vial, "🛣️", f"{offset_top}px", "10px"))
        offset_top += 40 + 20 * len(paleta_vial)

    # ── Magnitud / perímetro de incendios ─────────────────────
    elif "magnitud" in nombre_low or "incen" in nombre_low or "perimetro" in nombre_low:
        folium.GeoJson(
            gdf, name=f"🔥 {nombre}",
            style_function=crear_style_perimetro(),
            tooltip=folium.GeoJsonTooltip(
                fields=["NOM_INCEN", "COMUNA", "CAUSA", "SUPERFICIE"],
                aliases=["Incendio:", "Comuna:", "Causa:", "Superficie (ha):"]
            )
        ).add_to(m)
        leyendas_html.append(
            leyenda_categorica_html("Incendios (perímetro)", {"Área quemada": "#FF0000"}, "🔥", f"{offset_top}px", "10px")
        )
        offset_top += 60

    # ── Resto ────────────────────────────────────────────────
    else:
        folium.GeoJson(
            gdf, name=nombre,
            tooltip=folium.GeoJsonTooltip(fields=list(gdf.columns[:2]))
        ).add_to(m)

# ─────────────────────────────────────────────────────────────
# PASO 7: Renderizado Final
# ─────────────────────────────────────────────────────────────

for html in leyendas_html:
    m.get_root().html.add_child(folium.Element(html))

folium.LayerControl(collapsed=False).add_to(m)

st_folium(m, width=1200, height=650)

# ─────────────────────────────────────────────────────────────
# Métricas Sidebar (Funcionalidad Avanzada)
# ─────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Estadísticas de Sesión")
st.sidebar.write(f"Vectores activos: **{len(capas_activas)}**")
st.sidebar.write(f"Rasters activos: **{len(rasters_activos)}**")
