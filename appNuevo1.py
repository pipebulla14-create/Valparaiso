import streamlit as st
import os
os.environ["SHAPE_RESTORE_SHX"] = "YES"  # por si algún shapefile llega sin .shx
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
# Configuración general
# ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="GeoVisualizador Valparaíso", layout="wide")
st.title("🔥 GeoVisualizador de Severidad de Incendios - Valparaíso 2024")
st.write("Análisis territorial de daños utilizando imágenes satelitales y datos oficiales.")

DATA = Path("data")

# ─────────────────────────────────────────────────────────────
# Paletas y etiquetas cartográficas
# ─────────────────────────────────────────────────────────────

COLORES_ESPECIES = {
    "Eucalyptus globulus": "#2E7D32",
    "Pinus radiata": "#F9A825",
    "Quillaja saponaria": "#6D4C41",
    "Especies nativas": "#1B5E20",
    "Otras especies": "#9E9E9E",
    "Acacia spp.": "#8D6E63",
}

COLORES_SEVERIDAD = {
    1: (255, 255, 0),
    2: (255, 153, 0),
    3: (255, 0, 0),
}
ETIQUETAS_SEVERIDAD = {1: "Baja (0.10–0.27)", 2: "Moderada (0.27–0.44)", 3: "Alta (≥ 0.44)"}

LEYENDA_USO_VEGETACION = {
    "Bosque / vegetación densa": "#006400",
    "Matorral / vegetación arbustiva": "#d95f02",
    "Suelo desnudo / sin vegetación": "#bdbdbd",
    "Pradera / herbáceo": "#fee08b",
    "Área urbana / cuerpos de agua": "#80b1d3",
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

ENCODING_POR_ARCHIVO = {
    "if_magnitud_2023_2024": "latin-1",
    "Plantacion": "latin-1",
    "Red_Vial1": "cp1252",
}

# ─────────────────────────────────────────────────────────────
# Funciones de estilo para capas vectoriales
# ─────────────────────────────────────────────────────────────

def crear_style_categorico(col, paleta, color_default="#555555"):
    def style_fn(feature):
        val = str(feature["properties"].get(col, ""))
        color = paleta.get(val, color_default)
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

def crear_style_perimetro(color="#FF0000"):
    return lambda x: {"color": color, "weight": 3, "fillOpacity": 0.05, "dashArray": "5, 5"}

def crear_style_solido(fill_color, line_color=None, fill_opacity=0.4, dash=None):
    line_color = line_color or fill_color
    def style_fn(feature):
        s = {"fillColor": fill_color, "color": line_color, "weight": 1.5, "fillOpacity": fill_opacity}
        if dash:
            s["dashArray"] = dash
        return s
    return style_fn

# ─────────────────────────────────────────────────────────────
# Funciones de leyenda HTML
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

def leyenda_gradiente_html(titulo, colores_css, etiqueta_min, etiqueta_max, icono="🎨",
                            posicion_top="10px", posicion_right="10px"):
    stops = ", ".join(colores_css)
    barra = f"background: linear-gradient(to right, {stops}); height:14px; border-radius:3px; border:1px solid #999;"
    return f"""
    <div style="position:fixed;top:{posicion_top};right:{posicion_right};z-index:1000;background:rgba(255,255,255,0.93);padding:10px 14px;border-radius:8px;border:1px solid #bbb;box-shadow:2px 2px 6px rgba(0,0,0,0.25);font-family:Arial, sans-serif;width:220px;">
      <b style="font-size:12px;">{icono} {titulo}</b><hr style="margin:5px 0;border-color:#ddd;">
      <div style="{barra}"></div>
      <div style="display:flex;justify-content:space-between;font-size:10px;margin-top:3px;">
        <span>{etiqueta_min}</span><span>{etiqueta_max}</span>
      </div>
    </div>"""

# ─────────────────────────────────────────────────────────────
# Procesamiento de rasters (detección automática por N° de bandas y valores)
# ─────────────────────────────────────────────────────────────

def _reproyectar_a_4326(src, n_bandas):
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


def procesar_raster_severidad(raster_path):
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

        img_pil = Image.fromarray(rgba)
        buf = io.BytesIO()
        img_pil.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        bounds = [[bounds_wgs84[1], bounds_wgs84[0]], [bounds_wgs84[3], bounds_wgs84[2]]]
        return img_b64, bounds


def procesar_raster_continuo(raster_path, cmap_name="terrain"):
    with rasterio.open(raster_path) as src:
        nodata = src.nodata
        data, bounds_wgs84 = _reproyectar_a_4326(src, 1)
        banda = data[0]

        valido = banda != nodata if nodata is not None else np.ones_like(banda, dtype=bool)
        valido &= ~np.isnan(banda)
        if not np.any(valido):
            raise ValueError("El raster no tiene píxeles válidos (todo nodata).")

        vmin, vmax = np.percentile(banda[valido], (2, 98))
        norm = np.clip((banda - vmin) / (vmax - vmin + 1e-9), 0, 1)

        colormap = matplotlib.colormaps[cmap_name]
        rgba = (colormap(norm) * 255).astype(np.uint8)
        rgba[~valido, 3] = 0

        img_pil = Image.fromarray(rgba)
        buf = io.BytesIO()
        img_pil.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        bounds = [[bounds_wgs84[1], bounds_wgs84[0]], [bounds_wgs84[3], bounds_wgs84[2]]]
        return img_b64, bounds, float(vmin), float(vmax)


def procesar_raster_prerenderizado(raster_path, umbral_transparencia=10):
    with rasterio.open(raster_path) as src:
        n_bandas = min(3, src.count)
        data, bounds_wgs84 = _reproyectar_a_4326(src, n_bandas)

        rgb = np.clip(data, 0, 255).astype(np.uint8)
        rgb = np.transpose(rgb, (1, 2, 0))

        rgba = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
        rgba[..., :3] = rgb
        opaco = np.max(rgb, axis=-1) > umbral_transparencia
        rgba[..., 3] = np.where(opaco, 255, 0)

        img_pil = Image.fromarray(rgba)
        buf = io.BytesIO()
        img_pil.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        bounds = [[bounds_wgs84[1], bounds_wgs84[0]], [bounds_wgs84[3], bounds_wgs84[2]]]
        return img_b64, bounds


def inspeccionar_raster(raster_path):
    with rasterio.open(raster_path) as src:
        n_bandas = src.count
        if n_bandas >= 3:
            return "prerenderizado"
        banda = src.read(1)
        nodata = src.nodata
        valido = banda != nodata if nodata is not None else np.ones_like(banda, dtype=bool)
        valores = banda[valido]
        if valores.size == 0:
            return "continuo"
        unicos = np.unique(valores)
        es_entero = np.allclose(unicos, np.round(unicos))
        if es_entero and len(unicos) <= 8 and unicos.max() <= 10:
            return "clasificado"
        return "continuo"

# ─────────────────────────────────────────────────────────────
# Cargar archivos (lectura dinámica desde data/)
# ─────────────────────────────────────────────────────────────

archivos_vec = list(DATA.glob("*.gpkg")) + list(DATA.glob("*.shp")) + list(DATA.glob("*.geojson"))
capas = {}
for archivo in archivos_vec:
    nombre = archivo.stem.replace("_", " ")
    enc = ENCODING_POR_ARCHIVO.get(archivo.stem, "utf-8")
    try:
        gdf = gpd.read_file(archivo, encoding=enc)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:3857", allow_override=True)
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

st.sidebar.subheader("🛰️ Imágenes / Rasters")
rasters_activos = [n for n in rasters if st.sidebar.checkbox(n, value=False, key=f"rst_{n}")]

# ─────────────────────────────────────────────────────────────
# Mapa base
# ─────────────────────────────────────────────────────────────

centro = [-33.08, -71.48]
m = folium.Map(location=centro, zoom_start=11, tiles="OpenStreetMap")
folium.TileLayer("CartoDB positron", name="Mapa claro").add_to(m)
folium.TileLayer("CartoDB dark_matter", name="Mapa oscuro").add_to(m)

# ─────────────────────────────────────────────────────────────
# Rasters: detección automática de tipo + leyenda según nombre de archivo
# ─────────────────────────────────────────────────────────────

capas_raster_folium = {}
leyendas_html = []
offset_top = 10

for nombre in rasters_activos:
    nombre_low = nombre.lower().replace(" ", "")
    try:
        with st.spinner(f"Procesando {nombre}..."):
            tipo = inspeccionar_raster(rasters[nombre])

            if tipo == "clasificado":
                img_b64, bounds = procesar_raster_severidad(rasters[nombre])
                icono = "🔥"
                leyendas_html.append(leyenda_categorica_html(
                    "Severidad del incendio (dNBR)",
                    {ETIQUETAS_SEVERIDAD[k]: f"rgb{v}" for k, v in COLORES_SEVERIDAD.items()},
                    "🔥", f"{offset_top}px", "10px"
                ))
                offset_top += 140

            elif tipo == "continuo":
                if "dem" in nombre_low:
                    img_b64, bounds, vmin, vmax = procesar_raster_continuo(rasters[nombre], cmap_name="terrain")
                    icono = "⛰️"
                    leyendas_html.append(leyenda_gradiente_html(
                        "Elevación (DEM)", ["#2c7a3f", "#c9b458", "#8a5a3a", "#ffffff"],
                        f"{vmin:.0f} m", f"{vmax:.0f} m", "⛰️", f"{offset_top}px", "10px"
                    ))
                    offset_top += 100
                elif "perdida" in nombre_low or "ndvi" in nombre_low:
                    img_b64, bounds, vmin, vmax = procesar_raster_continuo(rasters[nombre], cmap_name="YlOrRd")
                    icono = "📉"
                    leyendas_html.append(leyenda_gradiente_html(
                        "Pérdida de NDVI (%)", ["#ffffcc", "#fd8d3c", "#800026"],
                        f"{vmin:.0f}%", f"{vmax:.0f}%", "📉", f"{offset_top}px", "10px"
                    ))
                    offset_top += 100
                else:
                    img_b64, bounds, vmin, vmax = procesar_raster_continuo(rasters[nombre], cmap_name="viridis")
                    icono = "📊"
                    leyendas_html.append(leyenda_gradiente_html(
                        nombre, ["#440154", "#21908C", "#FDE725"],
                        f"{vmin:.1f}", f"{vmax:.1f}", "📊", f"{offset_top}px", "10px"
                    ))
                    offset_top += 100

            else:
                img_b64, bounds = procesar_raster_prerenderizado(rasters[nombre])
                if "uso" in nombre_low or "vegetacion" in nombre_low:
                    icono = "🌿"
                    leyendas_html.append(leyenda_categorica_html(
                        "Uso / cobertura de vegetación", LEYENDA_USO_VEGETACION, "🌿", f"{offset_top}px", "10px"
                    ))
                    offset_top += 40 + 20 * len(LEYENDA_USO_VEGETACION)
                elif "perdida" in nombre_low or "ndvi" in nombre_low:
                    icono = "📉"
                    leyendas_html.append(leyenda_gradiente_html(
                        "Pérdida de NDVI (%)", ["#ffffcc", "#fd8d3c", "#800026"],
                        "Baja", "Alta", "📉", f"{offset_top}px", "10px"
                    ))
                    offset_top += 100
                elif "severidad" in nombre_low or "dnbr" in nombre_low:
                    icono = "🔥"
                    leyendas_html.append(leyenda_gradiente_html(
                        "Severidad / dNBR", ["#006400", "#ffffbf", "#d7191c"],
                        "Sin daño", "Daño alto", "🔥", f"{offset_top}px", "10px"
                    ))
                    offset_top += 100
                else:
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

candidatos_pre = [k for k in capas_raster_folium if "pre" in k and "dnbr" not in k and "severidad" not in k]
candidatos_post = [k for k in capas_raster_folium if "post" in k and "dnbr" not in k and "severidad" not in k]
if candidatos_pre and candidatos_post and candidatos_pre[0] != candidatos_post[0]:
    plugins.SideBySideLayers(
        layer_left=capas_raster_folium[candidatos_pre[0]],
        layer_right=capas_raster_folium[candidatos_post[0]]
    ).add_to(m)
    st.info("💡 **Modo Comparador activado:** Desliza la barra central en el mapa para comparar el Antes y el Después.")

# ─────────────────────────────────────────────────────────────
# Vectores: estilos y leyendas específicas por capa
# ─────────────────────────────────────────────────────────────

for nombre in capas_activas:
    gdf = capas[nombre]
    nombre_low = nombre.lower()
    cols_no_geom = [c for c in gdf.columns if c != "geometry"]

    if "plantacion" in nombre_low and "especie_t" in gdf.columns:
        folium.GeoJson(
            gdf, name=f"🌲 {nombre}",
            style_function=crear_style_categorico("especie_t", COLORES_ESPECIES),
            tooltip=folium.GeoJsonTooltip(
                fields=["especie_t"] + (["sup_ha"] if "sup_ha" in gdf.columns else []),
                aliases=["Especie:"] + (["Superficie (ha):"] if "sup_ha" in gdf.columns else [])
            )
        ).add_to(m)
        especies = sorted(gdf["especie_t"].dropna().unique())
        paleta_especies = {esp: COLORES_ESPECIES.get(esp, "#9E9E9E") for esp in especies}
        leyendas_html.append(leyenda_categorica_html("Plantación forestal", paleta_especies, "🌲", f"{offset_top}px", "10px"))
        offset_top += 40 + 20 * len(especies)

    elif ("vial" in nombre_low or "red" in nombre_low) and "Clase_Ruta" in gdf.columns:
        folium.GeoJson(
            gdf, name=f"🛣️ {nombre}",
            style_function=crear_style_transporte("Clase_Ruta"),
            tooltip=folium.GeoJsonTooltip(
                fields=["Nom_Ruta", "Clase_Ruta"] if "Nom_Ruta" in gdf.columns else ["Clase_Ruta"],
                aliases=["Nombre ruta:", "Clase:"] if "Nom_Ruta" in gdf.columns else ["Clase:"]
            )
        ).add_to(m)
        codigos_presentes = sorted(gdf["Clase_Ruta"].dropna().astype(int).unique())
        paleta_vial = {f"Clase {c}": ESTILOS_TIPO_TRANSPORTE.get(c, ESTILO_VIAL_DEFAULT)["color"] for c in codigos_presentes}
        leyendas_html.append(leyenda_categorica_html("Red vial (por clase)", paleta_vial, "🛣️", f"{offset_top}px", "10px"))
        offset_top += 40 + 20 * len(paleta_vial)

    elif ("magnitud" in nombre_low or "incen" in nombre_low) and "NOM_INCEN" in gdf.columns:
        folium.GeoJson(
            gdf, name=f"🔥 {nombre}",
            style_function=crear_style_perimetro(),
            tooltip=folium.GeoJsonTooltip(
                fields=[c for c in ["NOM_INCEN", "COMUNA", "CAUSA", "SUPERFICIE"] if c in gdf.columns],
                aliases=[a for c, a in zip(["NOM_INCEN", "COMUNA", "CAUSA", "SUPERFICIE"],
                                            ["Incendio:", "Comuna:", "Causa:", "Superficie (ha):"]) if c in gdf.columns]
            )
        ).add_to(m)
        leyendas_html.append(leyenda_categorica_html("Incendios (perímetro)", {"Área quemada": "#FF0000"}, "🔥", f"{offset_top}px", "10px"))
        offset_top += 60

    elif "poblada" in nombre_low or "urbano" in nombre_low:
        tooltip = folium.GeoJsonTooltip(fields=cols_no_geom[:2]) if cols_no_geom else "Área poblada"
        folium.GeoJson(
            gdf, name=f"🏙️ {nombre}",
            style_function=crear_style_solido("#B22222", "#7A1515", 0.45),
            tooltip=tooltip
        ).add_to(m)
        leyendas_html.append(leyenda_categorica_html("Área urbana", {"Zona poblada": "#B22222"}, "🏙️", f"{offset_top}px", "10px"))
        offset_top += 60

    elif "snaspe" in nombre_low:
        tooltip = folium.GeoJsonTooltip(fields=cols_no_geom[:2]) if cols_no_geom else "Área silvestre protegida"
        folium.GeoJson(
            gdf, name=f"🌲🛡️ {nombre}",
            style_function=crear_style_solido("#2E7D32", "#1B5E20", 0.25, dash="4, 4"),
            tooltip=tooltip
        ).add_to(m)
        leyendas_html.append(leyenda_categorica_html("Áreas protegidas (SNASPE)", {"Área SNASPE": "#2E7D32"}, "🛡️", f"{offset_top}px", "10px"))
        offset_top += 60

    else:
        tooltip = folium.GeoJsonTooltip(fields=cols_no_geom[:2]) if cols_no_geom else None
        folium.GeoJson(gdf, name=nombre, tooltip=tooltip).add_to(m)

# ─────────────────────────────────────────────────────────────
# Renderizado del mapa
# ─────────────────────────────────────────────────────────────

for html in leyendas_html:
    m.get_root().html.add_child(folium.Element(html))

m.add_child(plugins.MiniMap(toggle_display=True))
m.add_child(plugins.Fullscreen())
m.add_child(plugins.MeasureControl())
folium.LayerControl(collapsed=False).add_to(m)

st_folium(m, width=1200, height=650)

# ─────────────────────────────────────────────────────────────
# Panel de análisis interactivo por capa
# ─────────────────────────────────────────────────────────────

st.markdown("---")
st.header("📊 Panel de análisis por capa")

COLUMNA_CATEGORICA_POR_CAPA = {
    "plantacion": "especie_t",
    "vial": "Clase_Ruta",
    "red": "Clase_Ruta",
    "magnitud": "COMUNA",
    "incen": "COMUNA",
}

if capas_activas:
    capa_analisis = st.selectbox("Selecciona una capa para analizar en detalle:", capas_activas, key="capa_analisis")
    gdf_sel = capas[capa_analisis]
    gdf_metric = gdf_sel.to_crs(32719)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("N° de elementos", f"{len(gdf_sel):,}")

    geom_type = gdf_sel.geom_type.iloc[0] if len(gdf_sel) > 0 else ""
    if "Polygon" in geom_type:
        area_total_ha = gdf_metric.geometry.area.sum() / 10_000
        with col_b:
            st.metric("Superficie total", f"{area_total_ha:,.1f} ha")
    elif "Line" in geom_type:
        long_total_km = gdf_metric.geometry.length.sum() / 1_000
        with col_b:
            st.metric("Longitud total", f"{long_total_km:,.1f} km")

    nombre_low = capa_analisis.lower()
    col_categoria = next((v for k, v in COLUMNA_CATEGORICA_POR_CAPA.items() if k in nombre_low), None)

    if col_categoria and col_categoria in gdf_sel.columns:
        conteo = gdf_sel[col_categoria].value_counts().sort_values(ascending=False)
        with col_c:
            st.metric("N° de categorías", len(conteo))

        fig, ax = plt.subplots(figsize=(7, 3))
        conteo.plot(kind="bar", ax=ax, color="#E05000")
        ax.set_ylabel("N° de elementos")
        ax.set_xlabel(col_categoria)
        ax.set_title(f"Distribución por {col_categoria} — {capa_analisis}")
        plt.xticks(rotation=40, ha="right")
        plt.tight_layout()
        st.pyplot(fig)

    st.subheader("Tabla de atributos")
    cols_mostrar = [c for c in gdf_sel.columns if c != "geometry"]
    if cols_mostrar:
        st.dataframe(gdf_sel[cols_mostrar], use_container_width=True, height=300)
    else:
        st.info("Esta capa no tiene atributos (solo geometría) — probablemente falta el archivo .dbf original.")

    if cols_mostrar:
        csv_bytes = gdf_sel[cols_mostrar].to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"⬇️ Descargar {capa_analisis} como CSV",
            data=csv_bytes,
            file_name=f"{capa_analisis.replace(' ', '_')}.csv",
            mime="text/csv",
        )
else:
    st.info("Activa al menos una capa vectorial en el sidebar para ver su análisis.")

# ─────────────────────────────────────────────────────────────
# Resumen en sidebar
# ─────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Estadísticas de Sesión")
st.sidebar.write(f"Vectores activos: **{len(capas_activas)}**")
st.sidebar.write(f"Rasters activos: **{len(rasters_activos)}**")
