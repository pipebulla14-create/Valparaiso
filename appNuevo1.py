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
import matplotlib
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# 1. CONFIGURACIÓN GENERAL DE LA APLICACIÓN
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="GeoVisualizador Valparaíso", layout="wide")
st.title("🔥 GeoVisualizador de Severidad de Incendios - Valparaíso 2024")
st.write("Análisis territorial de daños utilizando imágenes satelitales y datos oficiales.")

DATA = Path("data")

# ─────────────────────────────────────────────────────────────
# 2. LEYENDAS INTERNAS DEL MAPA (HTML)
# ─────────────────────────────────────────────────────────────
def generar_leyenda_severidad_html():
    return """
    <div style="
        position: fixed; 
        bottom: 50px; 
        left: 20px; 
        width: 250px;
        height: auto; 
        z-index:9999; 
        background-color: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(6px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        padding: 15px; 
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        font-size: 13px; 
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.4);">
        
        <div style="text-align: center; margin-bottom: 10px;">
            <b style="font-size: 15px; color: #222;">🔥 Nivel de Daño (dNBR)</b>
        </div>
        <hr style="margin: 0 0 12px 0; border: 0; border-top: 1px solid #ddd;">
        
        <div style="display:flex; align-items:center; margin-bottom: 10px;">
            <div style="background-color: rgba(255,255,0,0.9); width: 22px; height: 22px; border-radius: 5px; margin-right: 12px; border: 1px solid #999; box-shadow: 1px 1px 3px rgba(0,0,0,0.2);"></div>
            <span style="color: #333; font-weight: 500;">Baja Severidad <br><small style="color: #666;">(0.10 - 0.27)</small></span>
        </div>
        <div style="display:flex; align-items:center; margin-bottom: 10px;">
            <div style="background-color: rgba(255,153,0,0.9); width: 22px; height: 22px; border-radius: 5px; margin-right: 12px; border: 1px solid #999; box-shadow: 1px 1px 3px rgba(0,0,0,0.2);"></div>
            <span style="color: #333; font-weight: 500;">Severidad Moderada <br><small style="color: #666;">(0.27 - 0.44)</small></span>
        </div>
        <div style="display:flex; align-items:center;">
            <div style="background-color: rgba(255,0,0,0.9); width: 22px; height: 22px; border-radius: 5px; margin-right: 12px; border: 1px solid #999; box-shadow: 1px 1px 3px rgba(0,0,0,0.2);"></div>
            <span style="color: #333; font-weight: 500;">Alta Severidad <br><small style="color: #666;">(≥ 0.44)</small></span>
        </div>
    </div>
    """

# ─────────────────────────────────────────────────────────────
# 3. FUNCIONES DE PROCESAMIENTO RASTER Y CÁLCULO
# ─────────────────────────────────────────────────────────────
def reproyectar_raster(src):
    if src.crs and src.crs.to_epsg() != 4326:
        transform, width, height = calculate_default_transform(
            src.crs, "EPSG:4326", src.width, src.height, *src.bounds
        )
        data = np.zeros((1, height, width), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1), destination=data[0],
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=transform, dst_crs="EPSG:4326", resampling=Resampling.nearest
        )
        bounds_wgs84 = rasterio.transform.array_bounds(height, width, transform)
    else:
        data = src.read(1).astype(np.float32)
        data = data[np.newaxis, ...]
        bounds_wgs84 = src.bounds
    return data[0], bounds_wgs84

def procesar_raster_color(ruta, tipo):
    with rasterio.open(ruta) as src:
        banda, bounds_wgs84 = reproyectar_raster(src)
        rgba = np.zeros((banda.shape[0], banda.shape[1], 4), dtype=np.uint8)
        
        nodata = src.nodata if src.nodata is not None else 0
        valido = (banda != nodata) & (~np.isnan(banda)) & (banda > -999)

        if tipo == "severidad":
            if np.any((banda == 1) | (banda == 2) | (banda == 3)):
                rgba[(banda == 1) & valido] = [255, 255, 0, 200]
                rgba[(banda == 2) & valido] = [255, 153, 0, 200]
                rgba[(banda == 3) & valido] = [255, 0, 0, 200]
            else:
                rgba[(banda >= 0.10) & (banda < 0.27) & valido] = [255, 255, 0, 200]
                rgba[(banda >= 0.27) & (banda < 0.44) & valido] = [255, 153, 0, 200]
                rgba[(banda >= 0.44) & valido] = [255, 0, 0, 200]
            
        elif tipo == "ndvi":
            vmin, vmax = np.percentile(banda[valido], (2, 98)) if np.any(valido) else (0, 100)
            norm = np.clip((banda - vmin) / (vmax - vmin + 1e-9), 0, 1)
            colormap = matplotlib.colormaps["YlOrRd"]
            rgba = (colormap(norm) * 255).astype(np.uint8)
            rgba[~valido, 3] = 0
            
        elif tipo == "dem":
            vmin, vmax = np.percentile(banda[valido], (2, 98)) if np.any(valido) else (0, 1000)
            norm = np.clip((banda - vmin) / (vmax - vmin + 1e-9), 0, 1)
            colormap = matplotlib.colormaps["terrain"]
            rgba = (colormap(norm) * 255).astype(np.uint8)
            rgba[~valido, 3] = 0
            
        elif tipo == "uso":
            unicos = np.unique(banda[valido])
            colores = [(0,100,0), (217,95,2), (189,189,189), (254,224,139), (128,177,211)]
            for i, val in enumerate(unicos[:5]):
                rgba[(banda == val) & valido] = list(colores[i]) + [200]

        img_pil = Image.fromarray(rgba)
        buf = io.BytesIO()
        img_pil.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        bounds = [[bounds_wgs84[1], bounds_wgs84[0]], [bounds_wgs84[3], bounds_wgs84[2]]]
        return img_b64, bounds

def calcular_area_quemada(ruta):
    with rasterio.open(ruta) as src:
        banda = src.read(1)
        nodata = src.nodata if src.nodata is not None else 0
        valido = (banda != nodata) & (~np.isnan(banda)) & (banda > -999)
        
        if np.any((banda == 1) | (banda == 2) | (banda == 3)):
            quemado = ((banda == 1) | (banda == 2) | (banda == 3)) & valido
        else:
            quemado = (banda >= 0.10) & valido
            
        if src.crs and not src.crs.is_projected:
            area_pixel_m2 = 30 * 30
        else:
            area_pixel_m2 = abs(src.res[0] * src.res[1])
            
        hectareas_totales = (np.sum(quemado) * area_pixel_m2) / 10000
        return hectareas_totales

# ─────────────────────────────────────────────────────────────
# 4. SIDEBAR: PANEL DE CONTROL DE CAPAS
# ─────────────────────────────────────────────────────────────
st.sidebar.title("Control de Capas")

st.sidebar.subheader("🗺️ Capas Vectoriales")
show_pobladas = st.sidebar.checkbox("Áreas Pobladas", value=True)
show_snaspe = st.sidebar.checkbox("Áreas SNASPE (Protegidas)", value=True)
show_red_vial = st.sidebar.checkbox("Red Vial (Caminos)", value=True)

st.sidebar.subheader("🛰️ Imágenes / Rasters")
show_dem = st.sidebar.checkbox("Modelo de Elevación (DEM 30m)", value=False)
show_uso = st.sidebar.checkbox("Uso de Vegetación", value=False)
show_ndvi = st.sidebar.checkbox("Pérdida de NDVI (%)", value=False)
show_severidad = st.sidebar.checkbox("Severidad dNBR (Solo Quemado)", value=True)

# ─────────────────────────────────────────────────────────────
# 5. INICIALIZACIÓN DEL MAPA FOLIUM
# ─────────────────────────────────────────────────────────────
m = folium.Map(location=[-33.08, -71.48], zoom_start=11, tiles="OpenStreetMap")
folium.TileLayer("CartoDB positron", name="Mapa claro").add_to(m)

# Nuevo Mapa Base: Imagen Satelital Esri
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri",
    name="🛰️ Satélite (Esri)"
).add_to(m)

capas_raster_activas = []

# ─────────────────────────────────────────────────────────────
# 6. CARGA Y RENDERIZADO DE RASTERS
# ─────────────────────────────────────────────────────────────
if show_severidad and (DATA / "VALPO_severidad_dNBR_solo_quemado.tif").exists():
    with st.spinner("Cargando Severidad..."):
        img, bnds = procesar_raster_color(DATA / "VALPO_severidad_dNBR_solo_quemado.tif", "severidad")
        capa = folium.raster_layers.ImageOverlay(image=f"data:image/png;base64,{img}", bounds=bnds, name="🔥 Severidad dNBR")
        capa.add_to(m)
        capas_raster_activas.append(capa)
        m.get_root().html.add_child(folium.Element(generar_leyenda_severidad_html()))

if show_ndvi and (DATA / "VALPO_perdida_NDVI_pct.tif").exists():
    with st.spinner("Cargando Pérdida NDVI..."):
        img, bnds = procesar_raster_color(DATA / "VALPO_perdida_NDVI_pct.tif", "ndvi")
        capa = folium.raster_layers.ImageOverlay(image=f"data:image/png;base64,{img}", bounds=bnds, name="📉 Pérdida NDVI")
        capa.add_to(m)
        capas_raster_activas.append(capa)

if show_uso and (DATA / "VALPO_UsoVegetacion.tif").exists():
    with st.spinner("Cargando Uso de Vegetación..."):
        img, bnds = procesar_raster_color(DATA / "VALPO_UsoVegetacion.tif", "uso")
        capa = folium.raster_layers.ImageOverlay(image=f"data:image/png;base64,{img}", bounds=bnds, name="🌿 Uso Vegetación")
        capa.add_to(m)
        capas_raster_activas.append(capa)

if show_dem and (DATA / "dem30.tif").exists():
    with st.spinner("Cargando DEM..."):
        img, bnds = procesar_raster_color(DATA / "dem30.tif", "dem")
        capa = folium.raster_layers.ImageOverlay(image=f"data:image/png;base64,{img}", bounds=bnds, name="⛰️ DEM 30m")
        capa.add_to(m)

if len(capas_raster_activas) == 2:
    plugins.SideBySideLayers(layer_left=capas_raster_activas[0], layer_right=capas_raster_activas[1]).add_to(m)
    st.info("💡 **Modo Comparador activado:** Desliza la barra central para comparar las capas raster.")

# ─────────────────────────────────────────────────────────────
# 7. CARGA Y RENDERIZADO DE VECTORES
# ─────────────────────────────────────────────────────────────
if show_pobladas and (DATA / "AreasPobladas.shp").exists():
    gdf_pob = gpd.read_file(DATA / "AreasPobladas.shp").to_crs(4326)
    folium.GeoJson(
        gdf_pob, name="🏙️ Áreas Pobladas",
        style_function=lambda x: {"fillColor": "#2980B9", "color": "#1C5980", "weight": 1.5, "fillOpacity": 0.5},
        tooltip=folium.GeoJsonTooltip(fields=list(gdf_pob.columns[:2])) if len(gdf_pob.columns) > 1 else "Área Poblada"
    ).add_to(m)

if show_snaspe and (DATA / "AreasSnaspe.shp").exists():
    gdf_snaspe = gpd.read_file(DATA / "AreasSnaspe.shp").to_crs(4326)
    folium.GeoJson(
        gdf_snaspe, name="🌲 Áreas SNASPE",
        style_function=lambda x: {"fillColor": "#2E7D32", "color": "#1B5E20", "weight": 2, "fillOpacity": 0.3, "dashArray": "5, 5"},
        tooltip=folium.GeoJsonTooltip(fields=list(gdf_snaspe.columns[:2])) if len(gdf_snaspe.columns) > 1 else "SNASPE"
    ).add_to(m)

if show_red_vial and (DATA / "redVial.shp").exists():
    gdf_vial = gpd.read_file(DATA / "redVial.shp").to_crs(4326)
    folium.GeoJson(
        gdf_vial, name="🛣️ Red Vial",
        style_function=lambda x: {"color": "#333333", "weight": 2, "opacity": 0.8},
        tooltip=folium.GeoJsonTooltip(fields=list(gdf_vial.columns[:2])) if len(gdf_vial.columns) > 1 else "Vía"
    ).add_to(m)

# ─────────────────────────────────────────────────────────────
# 8. VISUALIZACIÓN DEL MAPA EN LA APP
# ─────────────────────────────────────────────────────────────
folium.LayerControl(collapsed=False).add_to(m)
st_folium(m, width=1200, height=650)

# ─────────────────────────────────────────────────────────────
# 9. DASHBOARD: ESTADÍSTICAS Y GRÁFICOS INTERACTIVOS
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.header("📊 Análisis Territorial Detallado")

try:
    # 9.1 MÉTRICAS GLOBALES
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        if show_red_vial and 'gdf_vial' in locals():
            if gdf_vial.crs is None:
                gdf_vial = gdf_vial.set_crs(4326)
            gdf_vial_utm = gdf_vial.to_crs(32719)
            longitud_km = gdf_vial_utm.geometry.length.sum() / 1000
            st.metric("🛣️ Red Vial Afectada / Analizada", f"{longitud_km:,.1f} km")
            
    with col_m2:
        if show_severidad and (DATA / "VALPO_severidad_dNBR_solo_quemado.tif").exists():
            ha_quemadas = calcular_area_quemada(DATA / "VALPO_severidad_dNBR_solo_quemado.tif")
            st.metric("🔥 Superficie Total Quemada", f"{ha_quemadas:,.1f} hectáreas")
            
    st.markdown("---")

    col_tabla, col_grafico = st.columns(2)

    # 9.2 TABLA DE ATRIBUTOS (CON NOMBRES LIMPIOS Y WARNING CORREGIDO)
    if show_pobladas and 'gdf_pob' in locals():
        with col_tabla:
            st.subheader("🏙️ Registro de Áreas Pobladas")
            cols_mostrar = [c for c in gdf_pob.columns if c != "geometry"]
            df_limpio = gdf_pob[cols_mostrar].copy()
            
            # Renombramos las columnas crudas del Shapefile por nombres limpios para la presentación
            df_limpio = df_limpio.rename(columns={
                "objectid": "ID",
                "st_area_sh": "Área (m²)",
                "st_length_": "Perímetro (m)",
                "comuna": "Comuna"
            })
            # width="stretch" reemplaza al antiguo use_container_width=True para eliminar la advertencia
            st.dataframe(df_limpio, height=400, width="stretch")

    if show_snaspe and 'gdf_snaspe' in locals():
        with col_grafico:
            st.subheader("🌲 Superficie de Áreas Protegidas")
            if gdf_snaspe.crs is None:
                gdf_snaspe = gdf_snaspe.set_crs(4326)
                
            gdf_snaspe_utm = gdf_snaspe.to_crs(32719)
            gdf_snaspe['Area_ha'] = gdf_snaspe_utm.geometry.area / 10000
            
            cols_texto = gdf_snaspe.select_dtypes(include=['object']).columns
            eje_x = gdf_snaspe[cols_texto[0]].astype(str) if len(cols_texto) > 0 else gdf_snaspe.index.astype(str)
            
            fig, ax = plt.subplots(figsize=(10, 7))
            plt.rcParams.update({'font.size': 24})
            
            ax.bar(eje_x, gdf_snaspe['Area_ha'], color="#2E7D32", edgecolor="black")
            ax.set_ylabel("Hectáreas (ha)", fontsize=24)
            plt.xticks(rotation=45, ha='right', fontsize=24)
            
            if not gdf_snaspe.empty:
                max_idx = gdf_snaspe['Area_ha'].idxmax()
                max_val = gdf_snaspe['Area_ha'].max()
                ax.annotate(
                    'Mayor extensión', 
                    xy=(max_idx, max_val), 
                    xytext=(max_idx, max_val * 1.15),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=5, headwidth=15),
                    fontsize=24, 
                    ha='center'
                )
            
            plt.tight_layout()
            st.pyplot(fig)

except Exception as e:
    st.error(f"Ocurrió un error al cargar el dashboard de estadísticas: {e}")
