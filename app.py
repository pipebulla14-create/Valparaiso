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

# Uso de Suelo: colores basados en las categorías oficiales de MINAGRI/CONAF
COLORES_USO_SUELO = {
    "Áreas Urbanas e Industriales": "#E60000",
    "Terrenos Agrícolas": "#FFAA00",
    "Praderas y Matorrales": "#E6E600",
    "Bosques": "#38A800",
    "Áreas Desprovistas de Vegetación": "#CCCCCC",
    "Cuerpos de Agua": "#004DA8"
}

# Transporte: jerarquía vial estándar
ESTILOS_TIPO_TRANSPORTE = {
    "autopista":        {"color": "#CC0000", "weight": 4},
    "ruta":             {"color": "#E05000", "weight": 3},
    "camino":           {"color": "#FF8800", "weight": 2},
    "pavimentado":      {"color": "#E05000", "weight": 3},
    "ripio":            {"color": "#DAA520", "weight": 2},
    "tierra":           {"color": "#8B6914", "weight": 2},
    "default":          {"color": "#888888", "weight": 2},
}

# ─────────────────────────────────────────────────────────────
# PASO 2: Funciones de color dinámico
# ─────────────────────────────────────────────────────────────

def crear_style_uso_suelo(col):
    """Estilo para la capa de Uso de Suelo usando la paleta oficial."""
    def style_fn(feature):
        val = str(feature["properties"].get(col, ""))
        color = COLORES_USO_SUELO.get(val, "#555555")
        return {
            "fillColor": color,
            "color": "#333333",
            "weight": 0.5,
            "fillOpacity": 0.65,
        }
    return style_fn

def crear_style_transporte(col_tipo):
    """Estilo para red vial buscando coincidencias en la jerarquía."""
    def style_fn(feature):
        tipo = str(feature["properties"].get(col_tipo, "")).lower()
        for key, vals in ESTILOS_TIPO_TRANSPORTE.items():
            if key in tipo:
                return {"color": vals["color"], "weight": vals["weight"], "opacity": 0.9}
        return {"color": ESTILOS_TIPO_TRANSPORTE["default"]["color"], "weight": 2, "opacity": 0.9}
    return style_fn

def crear_style_perimetro():
    """Estilo destacado para el polígono del incendio."""
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
# PASO 4: Procesamiento de Imágenes Satelitales (RGB)
# ─────────────────────────────────────────────────────────────

def procesar_raster_rgb(raster_path):
    """Convierte un raster multiespectral en una imagen renderizable para Folium."""
    with rasterio.open(raster_path) as src:
        if src.crs and src.crs.to_epsg() != 4326:
            transform, width, height = calculate_default_transform(src.crs, "EPSG:4326", src.width, src.height, *src.bounds)
            data = np.zeros((3, height, width), dtype=np.float32)
            for i in range(1, 4):
                reproject(
                    source=rasterio.band(src, i), destination=data[i - 1],
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=transform, dst_crs="EPSG:4326", resampling=Resampling.nearest
                )
            bounds_wgs84 = rasterio.transform.array_bounds(height, width, transform)
        else:
            data = src.read()[:3].astype(np.float32)
            bounds_wgs84 = src.bounds

        rgb = np.zeros((data.shape[1], data.shape[2], 4), dtype=np.uint8)
        for i in range(3):
            banda = data[i]
            min_val, max_val = np.percentile(banda[banda > 0], (2, 98)) if np.any(banda > 0) else (0, 1)
            rgb[..., i] = np.clip(255 * (banda - min_val) / (max_val - min_val + 1e-5), 0, 255)
        
        rgb[..., 3] = np.where(np.max(data, axis=0) > 0, 255, 0)
        
        img_pil = Image.fromarray(rgb)
        buf = io.BytesIO()
        img_pil.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        bounds = [[bounds_wgs84[1], bounds_wgs84[0]], [bounds_wgs84[3], bounds_wgs84[2]]]
        
        return img_b64, bounds

# ─────────────────────────────────────────────────────────────
# Cargar archivos (Lectura dinámica)
# ─────────────────────────────────────────────────────────────

archivos_vec = list(DATA.glob("*.gpkg")) + list(DATA.glob("*.shp")) + list(DATA.glob("*.geojson"))
capas = {}
for archivo in archivos_vec:
    nombre = archivo.stem.replace("_", " ")
    try:
        gdf = gpd.read_file(archivo)
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

# Centrar en Valparaíso por defecto
centro = [-33.08, -71.48]
m = folium.Map(location=centro, zoom_start=11, tiles="OpenStreetMap")
folium.TileLayer("CartoDB positron", name="Mapa claro").add_to(m)
folium.TileLayer("CartoDB dark_matter", name="Mapa oscuro").add_to(m)

# ─────────────────────────────────────────────────────────────
# PASO 5: Agregar Rasters y Comparador (SideBySide)
# ─────────────────────────────────────────────────────────────
capas_raster_folium = {}

for nombre in rasters_activos:
    try:
        with st.spinner(f"Procesando {nombre}..."):
            img_b64, bounds = procesar_raster_rgb(rasters[nombre])
            
            img_layer = folium.raster_layers.ImageOverlay(
                image=f"data:image/png;base64,{img_b64}",
                bounds=bounds,
                opacity=1.0,
                name=f"🛰 {nombre}",
            )
            img_layer.add_to(m)
            capas_raster_folium[nombre.lower()] = img_layer
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
leyendas_html = []
offset_top = 10

for nombre in capas_activas:
    gdf = capas[nombre]
    nombre_low = nombre.lower()

    # ── Uso de Suelo ─────────────────────────────────────────
    if "uso" in nombre_low or "suelo" in nombre_low:
        col = "USO" if "USO" in gdf.columns else "uso" if "uso" in gdf.columns else None
        
        if col:
            folium.GeoJson(
                gdf, name=f"🌳 {nombre}",
                style_function=crear_style_uso_suelo(col),
                tooltip=folium.GeoJsonTooltip(fields=[col])
            ).add_to(m)
            
            leyendas_html.append(leyenda_categorica_html("Uso de Suelo", COLORES_USO_SUELO, "🌳", f"{offset_top}px", "10px"))
            offset_top += 180
        else:
            folium.GeoJson(gdf, name=nombre).add_to(m)

    # ── Red Vial ─────────────────────────────────────────────
    elif "vial" in nombre_low or "red" in nombre_low:
        folium.GeoJson(
            gdf, name=f"🛣️ {nombre}",
            style_function=crear_style_transporte("TIPO"),
            tooltip=folium.GeoJsonTooltip(fields=list(gdf.columns[:2]))
        ).add_to(m)

    # ── Perímetro ────────────────────────────────────────────
    elif "perimetro" in nombre_low:
        folium.GeoJson(
            gdf, name=f"🔥 {nombre}",
            style_function=crear_style_perimetro(),
            tooltip=folium.GeoJsonTooltip(fields=list(gdf.columns[:1]))
        ).add_to(m)

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
