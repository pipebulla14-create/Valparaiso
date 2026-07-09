import streamlit as st
import geopandas as gpd
import folium
from folium import plugins
from streamlit_folium import st_folium
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
from shapely.geometry import box
import io, base64
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# 1. CONFIGURACIÓN GENERAL
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="GeoVisualizador Valparaíso", layout="wide")
st.title("🔥 GeoVisualizador de Severidad de Incendios - Valparaíso 2024")
st.write("Análisis territorial de daños utilizando imágenes satelitales y datos oficiales.")

DATA = Path("data")

# ─────────────────────────────────────────────────────────────
# 2. FUNCIONES DE APOYO
# ─────────────────────────────────────────────────────────────
def generar_leyenda_severidad_html():
    return """
    <div style="position: fixed; bottom: 50px; left: 20px; width: 250px; z-index:9999; background-color: rgba(255, 255, 255, 0.85); backdrop-filter: blur(6px); padding: 15px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.4);">
        <b style="font-size: 15px;">🔥 Nivel de Daño (dNBR)</b><hr style="margin: 5px 0;">
        <div style="display:flex; align-items:center;"><div style="background-color: rgb(255,255,0); width: 20px; height: 20px; margin-right: 10px;"></div> Baja (0.10 - 0.27)</div>
        <div style="display:flex; align-items:center;"><div style="background-color: rgb(255,153,0); width: 20px; height: 20px; margin-right: 10px;"></div> Moderada (0.27 - 0.44)</div>
        <div style="display:flex; align-items:center;"><div style="background-color: rgb(255,0,0); width: 20px; height: 20px; margin-right: 10px;"></div> Alta (≥ 0.44)</div>
    </div>"""

def reproyectar_raster(src):
    if src.crs and src.crs.to_epsg() != 4326:
        transform, width, height = calculate_default_transform(src.crs, "EPSG:4326", src.width, src.height, *src.bounds)
        data = np.zeros((1, height, width), dtype=np.float32)
        reproject(source=rasterio.band(src, 1), destination=data[0], src_transform=src.transform, src_crs=src.crs, dst_transform=transform, dst_crs="EPSG:4326", resampling=Resampling.nearest)
        return data[0], rasterio.transform.array_bounds(height, width, transform)
    return src.read(1).astype(np.float32), src.bounds

def procesar_raster_color(ruta, tipo):
    with rasterio.open(ruta) as src:
        banda, bounds = reproyectar_raster(src)
        rgba = np.zeros((banda.shape[0], banda.shape[1], 4), dtype=np.uint8)
        valido = (banda != (src.nodata or 0)) & (~np.isnan(banda)) & (banda > -999)
        if tipo == "severidad":
            rgba[(banda >= 0.10) & (banda < 0.27) & valido] = [255, 255, 0, 200]
            rgba[(banda >= 0.27) & (banda < 0.44) & valido] = [255, 153, 0, 200]
            rgba[(banda >= 0.44) & valido] = [255, 0, 0, 200]
        img_pil = Image.fromarray(rgba)
        buf = io.BytesIO()
        img_pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8"), [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]

def calcular_area_quemada(ruta):
    with rasterio.open(ruta) as src:
        banda = src.read(1)
        valido = (banda != (src.nodata or 0)) & (~np.isnan(banda))
        area_pixel_ha = abs(src.res[0] * src.res[1]) / 10000
        max_val = np.nanmax(np.where(valido, banda, np.nan))
        quemado = (banda >= 0.10) & valido if max_val < 5 else (banda >= 100) & valido
        return np.sum(quemado) * area_pixel_ha

# ─────────────────────────────────────────────────────────────
# 3. INTERFAZ Y MAPA
# ─────────────────────────────────────────────────────────────
st.sidebar.title("Control de Capas")
show_pobladas = st.sidebar.checkbox("Áreas Pobladas", True)
show_severidad = st.sidebar.checkbox("Severidad dNBR", True)

m = folium.Map(location=[-33.08, -71.48], zoom_start=11)
folium.TileLayer("CartoDB positron").add_to(m)

if show_severidad and (DATA / "VALPO_severidad_dNBR_solo_quemado.tif").exists():
    img, bnds = procesar_raster_color(DATA / "VALPO_severidad_dNBR_solo_quemado.tif", "severidad")
    folium.raster_layers.ImageOverlay(f"data:image/png;base64,{img}", bounds=bnds).add_to(m)
    m.get_root().html.add_child(folium.Element(generar_leyenda_severidad_html()))

if show_pobladas and (DATA / "AreasPobladas.shp").exists():
    gdf_pob = gpd.read_file(DATA / "AreasPobladas.shp").to_crs(4326)
    folium.GeoJson(gdf_pob, name="Pobladas", style_function=lambda x: {"fillColor": "#2980B9", "color": "#1C5980"}).add_to(m)

st_folium(m, width=1200, height=500)

# ─────────────────────────────────────────────────────────────
# 4. DASHBOARD
# ─────────────────────────────────────────────────────────────
st.header("📊 Análisis Territorial Detallado")
try:
    c1, c2, c3 = st.columns(3)
    
    if show_severidad and (DATA / "VALPO_severidad_dNBR_solo_quemado.tif").exists():
        ha_quemadas = calcular_area_quemada(DATA / "VALPO_severidad_dNBR_solo_quemado.tif")
        c2.metric("🔥 Superficie Total Quemada", f"{ha_quemadas:,.1f} ha")
    
    if show_pobladas and show_severidad and 'gdf_pob' in locals():
        with rasterio.open(DATA / "VALPO_severidad_dNBR_solo_quemado.tif") as src:
            gdf_pob_proj = gdf_pob.to_crs(src.crs)
            if gdf_pob_proj.intersects(box(*src.bounds)).any():
                out_image, _ = mask(src, gdf_pob_proj.geometry.values, crop=True, nodata=0)
                ha_urbana_quemada = np.sum(out_image[0] > 0.1) * (abs(src.res[0]*src.res[1])/10000)
                c3.metric("🏘️ Área Poblada Quemada", f"{ha_urbana_quemada:,.1f} ha")

    if show_pobladas and 'gdf_pob' in locals():
        st.subheader("🏙️ Registro de Áreas Pobladas")
        st.dataframe(gdf_pob.drop(columns='geometry'), use_container_width=True)

except Exception as e:
    st.error(f"Error en dashboard: {e}")
