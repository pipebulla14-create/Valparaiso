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
# CONFIGURACIÓN GENERAL
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="GeoVisualizador Valparaíso", layout="wide")
st.title("🔥 GeoVisualizador de Severidad de Incendios - Valparaíso 2024")
st.write("Análisis territorial de daños utilizando imágenes satelitales y datos oficiales.")

DATA = Path("data")

# ─────────────────────────────────────────────────────────────
# FUNCIONES DE PROCESAMIENTO RASTER
# ─────────────────────────────────────────────────────────────
def reproyectar_raster(src):
    """Reproyecta un raster a WGS84 (EPSG:4326) para que Folium lo pueda leer."""
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
    """Convierte el raster a una imagen PNG en base64 con colores según su temática."""
    with rasterio.open(ruta) as src:
        banda, bounds_wgs84 = reproyectar_raster(src)
        rgba = np.zeros((banda.shape[0], banda.shape[1], 4), dtype=np.uint8)
        
        # Máscara para ignorar valores nulos o ceros absolutos fuera del área
        nodata = src.nodata if src.nodata is not None else 0
        valido = (banda != nodata) & (~np.isnan(banda))

        if tipo == "severidad":
            # 1: Baja (Amarillo), 2: Moderada (Naranjo), 3: Alta (Rojo)
            rgba[(banda == 1) & valido] = [255, 255, 0, 200]
            rgba[(banda == 2) & valido] = [255, 153, 0, 200]
            rgba[(banda == 3) & valido] = [255, 0, 0, 200]
            
        elif tipo == "ndvi":
            # Gradiente de pérdida (Amarillo a Rojo oscuro)
            vmin, vmax = np.percentile(banda[valido], (2, 98)) if np.any(valido) else (0, 100)
            norm = np.clip((banda - vmin) / (vmax - vmin + 1e-9), 0, 1)
            colormap = matplotlib.colormaps["YlOrRd"]
            rgba = (colormap(norm) * 255).astype(np.uint8)
            rgba[~valido, 3] = 0
            
        elif tipo == "dem":
            # Gradiente de terreno
            vmin, vmax = np.percentile(banda[valido], (2, 98)) if np.any(valido) else (0, 1000)
            norm = np.clip((banda - vmin) / (vmax - vmin + 1e-9), 0, 1)
            colormap = matplotlib.colormaps["terrain"]
            rgba = (colormap(norm) * 255).astype(np.uint8)
            rgba[~valido, 3] = 0
            
        elif tipo == "uso":
            # Clasificación de uso de vegetación (Colores aleatorios base, puedes ajustar)
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

# ─────────────────────────────────────────────────────────────
# SIDEBAR: CONTROL DE CAPAS
# ─────────────────────────────────────────────────────────────
st.sidebar.title("Control de Capas")

st.sidebar.subheader("🗺️ Capas Vectoriales")
show_pobladas = st.sidebar.checkbox("Áreas Pobladas", value=True)
show_snaspe = st.sidebar.checkbox("Áreas SNASPE (Protegidas)", value=True)

st.sidebar.subheader("🛰️ Imágenes / Rasters")
show_dem = st.sidebar.checkbox("Modelo de Elevación (DEM 30m)", value=False)
show_uso = st.sidebar.checkbox("Uso de Vegetación", value=False)
show_ndvi = st.sidebar.checkbox("Pérdida de NDVI (%)", value=False)
show_severidad = st.sidebar.checkbox("Severidad dNBR (Solo Quemado)", value=True)

# ─────────────────────────────────────────────────────────────
# INICIALIZACIÓN DEL MAPA
# ─────────────────────────────────────────────────────────────
m = folium.Map(location=[-33.08, -71.48], zoom_start=11, tiles="OpenStreetMap")
folium.TileLayer("CartoDB positron", name="Mapa claro").add_to(m)

# Diccionario para almacenar las capas raster activas y armar el comparador
capas_raster_activas = []

# ─────────────────────────────────────────────────────────────
# CARGA Y RENDERIZADO DE RASTERS
# ─────────────────────────────────────────────────────────────
if show_severidad and (DATA / "VALPO_severidad_dNBR_solo_quemado.tif").exists():
    with st.spinner("Cargando Severidad..."):
        img, bnds = procesar_raster_color(DATA / "VALPO_severidad_dNBR_solo_quemado.tif", "severidad")
        capa = folium.raster_layers.ImageOverlay(image=f"data:image/png;base64,{img}", bounds=bnds, name="🔥 Severidad dNBR")
        capa.add_to(m)
        capas_raster_activas.append(capa)

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

# Comparador automático si hay exactamente 2 rasters analíticos activos
if len(capas_raster_activas) == 2:
    plugins.SideBySideLayers(layer_left=capas_raster_activas[0], layer_right=capas_raster_activas[1]).add_to(m)
    st.info("💡 **Modo Comparador activado:** Desliza la barra central para comparar las capas raster.")

# ─────────────────────────────────────────────────────────────
# CARGA Y RENDERIZADO DE VECTORES
# ─────────────────────────────────────────────────────────────
if show_pobladas and (DATA / "AreasPobladas.shp").exists():
    gdf_pob = gpd.read_file(DATA / "AreasPobladas.shp").to_crs(4326)
    folium.GeoJson(
        gdf_pob, name="🏙️ Áreas Pobladas",
        style_function=lambda x: {"fillColor": "#B22222", "color": "#7A1515", "weight": 1.5, "fillOpacity": 0.45},
        tooltip=folium.GeoJsonTooltip(fields=list(gdf_pob.columns[:2])) if len(gdf_pob.columns) > 1 else "Área Poblada"
    ).add_to(m)

if show_snaspe and (DATA / "AreasSnaspe.shp").exists():
    gdf_snaspe = gpd.read_file(DATA / "AreasSnaspe.shp").to_crs(4326)
    folium.GeoJson(
        gdf_snaspe, name="🌲 Áreas SNASPE",
        style_function=lambda x: {"fillColor": "#2E7D32", "color": "#1B5E20", "weight": 2, "fillOpacity": 0.3, "dashArray": "5, 5"},
        tooltip=folium.GeoJsonTooltip(fields=list(gdf_snaspe.columns[:2])) if len(gdf_snaspe.columns) > 1 else "SNASPE"
    ).add_to(m)

# ─────────────────────────────────────────────────────────────
# RENDERIZADO FINAL
# ─────────────────────────────────────────────────────────────
folium.LayerControl(collapsed=False).add_to(m)
st_folium(m, width=1200, height=650)
# ─────────────────────────────────────────────────────────────
# DASHBOARD: ESTADÍSTICAS Y TABLAS (Funcionalidades Avanzadas)
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.header("📊 Análisis Territorial Detallado")

# Dividir la pantalla en dos columnas para mejor estética
col_tabla, col_grafico = st.columns(2)

# 1. TABLA INTERACTIVA: Áreas Pobladas
if show_pobladas and 'gdf_pob' in locals():
    with col_tabla:
        st.subheader("🏙️ Registro de Áreas Pobladas")
        st.write("Explora los atributos de los polígonos urbanos:")
        
        # Filtramos la columna 'geometry' porque no se lee bien en tablas
        cols_mostrar = [c for c in gdf_pob.columns if c != "geometry"]
        
        # st.dataframe crea una tabla interactiva (ordenable y con scroll)
        st.dataframe(gdf_pob[cols_mostrar], height=400, use_container_width=True)

# 2. GRÁFICO ESTADÍSTICO: Áreas Protegidas (SNASPE)
if show_snaspe and 'gdf_snaspe' in locals():
    with col_grafico:
        st.subheader("🌲 Superficie de Áreas Protegidas (ha)")
        
        # Calcular el área en hectáreas (reproyectando a UTM 19S para medir en metros)
        gdf_snaspe_utm = gdf_snaspe.to_crs(32719)
        gdf_snaspe['Area_ha'] = gdf_snaspe_utm.geometry.area / 10000
        
        # Buscamos la columna que tenga los nombres (o usamos el índice por defecto)
        cols_texto = gdf_snaspe.select_dtypes(include=['object']).columns
        eje_x = gdf_snaspe[cols_texto[0]].astype(str) if len(cols_texto) > 0 else gdf_snaspe.index.astype(str)
        
        # Crear gráfico con Matplotlib
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Configuramos tamaños de fuente grandes para legibilidad óptica
        plt.rcParams.update({'font.size': 24})
        
        barras = ax.bar(eje_x, gdf_snaspe['Area_ha'], color="#2E7D32", edgecolor="black")
        
        ax.set_ylabel("Hectáreas (ha)", fontsize=24)
        ax.set_xlabel("Unidad SNASPE", fontsize=24)
        plt.xticks(rotation=45, ha='right', fontsize=24)
        plt.yticks(fontsize=24)
        
        # Identificar el polígono más grande para destacarlo con una flecha
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
        
        # Ajustar los márgenes para que no se corten los textos
        plt.tight_layout()
        
        # Renderizar en la app
        st.pyplot(fig)
