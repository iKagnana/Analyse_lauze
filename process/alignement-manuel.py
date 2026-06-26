import os
import tempfile
import numpy as np
import streamlit as st
import pyvista as pv
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

pv.OFF_SCREEN = True

# =======================================================================================
# CODE COULEUR (COINS DE LA BOUNDING BOX)
# =======================================================================================
INFOS_POINTS = {
    "Coin Haut Gauche": {"color": "red", "emoji": "🔴"},
    "Coin Haut Droit": {"color": "orange", "emoji": "🟠"},
    "Coin Bas Gauche": {"color": "blue", "emoji": "🔵"},
    "Coin Bas Droit": {"color": "green", "emoji": "🟢"},
    "Centre Bounding Box": {"color": "purple", "emoji": "🟣"}
}

# =======================================================================================
# FONCTIONS OBB ET GÉOMÉTRIE
# =======================================================================================
@st.cache_data(show_spinner="Lecture du maillage 3D...", max_entries=4)
def load_mesh_from_bytes(file_bytes: bytes, file_suffix: str):
    """Charge un fichier 3D depuis ses bytes bruts. cache_data = sérialisable + libérable."""
    import pyvista as pv
    pv.OFF_SCREEN = True
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    mesh = pv.read(tmp_path)
    os.unlink(tmp_path)  # Nettoyage immédiat du fichier temporaire
    return mesh

def get_bounding_box_anchors(mesh, orientation_mode):
    """Calcule la Bounding Box orientée (OBB) et renvoie les 4 coins + le centre."""
    pts = mesh.points
    centre = pts.mean(axis=0)
    
    if orientation_mode == "Forcer Horizontale (Axe X principal)":
        axe_x, axe_y = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    elif orientation_mode == "Forcer Verticale (Axe Y principal)":
        axe_x, axe_y = np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])
    else:
        cov = np.cov(pts - centre, rowvar=False)
        _, vecteurs_propres = np.linalg.eigh(cov)
        axe_x, axe_y = vecteurs_propres[:, -1], vecteurs_propres[:, -2]

    proj_x = np.dot(pts - centre, axe_x)
    proj_y = np.dot(pts - centre, axe_y)
    
    min_x, max_x = proj_x.min(), proj_x.max()
    min_y, max_y = proj_y.min(), proj_y.max()
    
    coin_hg = centre + (max_x * axe_x) + (max_y * axe_y)
    coin_hd = centre + (max_x * axe_x) + (min_y * axe_y)
    coin_bg = centre + (min_x * axe_x) + (max_y * axe_y)
    coin_bd = centre + (min_x * axe_x) + (min_y * axe_y)
    
    return {
        "Coin Haut Gauche": coin_hg,
        "Coin Haut Droit": coin_hd,
        "Coin Bas Gauche": coin_bg,
        "Coin Bas Droit": coin_bd,
        "Centre Bounding Box": centre
    }

def plot_mesh_with_obb(mesh, anchors, color_mesh="gray"):
    """Génère le rendu 3D avec les points ET le rectangle de la Bounding Box."""
    pl = pv.Plotter(window_size=[600, 400])
    pl.set_background("white")
    pl.add_mesh(mesh, color=color_mesh, opacity=0.7)
    
    # 1. Dessiner les points colorés
    for name, coord in anchors.items():
        info = INFOS_POINTS[name]
        pl.add_mesh(pv.Sphere(radius=10.0, center=coord), color=info["color"])
        pl.add_point_labels([coord], [f" {info['emoji']} {name}"], point_size=15, text_color=info["color"], shape=None, font_size=9)
    
    # 2. Dessiner les lignes de la Bounding Box (Le Rectangle)
    corners = np.array([
        anchors["Coin Haut Gauche"], anchors["Coin Haut Droit"], 
        anchors["Coin Bas Droit"], anchors["Coin Bas Gauche"]
    ])
    lines = np.array([2, 0, 1, 2, 1, 2, 2, 2, 3, 2, 3, 0])
    box_wireframe = pv.PolyData(corners)
    box_wireframe.lines = lines
    pl.add_mesh(box_wireframe, color="black", line_width=2)
    
    pl.view_isometric()
    return pl.screenshot(return_img=True)

# Fonctions d'alignement (Conservées du code précédent)
def align_meshes_multi_points(mesh_bot, mesh_top, pts_bot_array, pts_top_array):
    centroid_bot = np.mean(pts_bot_array, axis=0)
    centroid_top = np.mean(pts_top_array, axis=0)
    rot, _ = R.align_vectors(pts_bot_array - centroid_bot, pts_top_array - centroid_top)
    aligned_top = mesh_top.copy()
    aligned_top.points = rot.apply(aligned_top.points - centroid_top) + centroid_bot
    return aligned_top

def resolve_mesh_collisions(mesh_bot, mesh_top):
    tree = cKDTree(mesh_bot.points[:, :2])
    _, indices = tree.query(mesh_top.points[:, :2], k=1)
    min_gap = np.min(mesh_top.points[:, 2] - mesh_bot.points[indices, 2])
    corrected_top = mesh_top.copy()
    corrected_top.points[:, 2] -= min_gap
    return corrected_top, -min_gap

def apply_manual_fine_tuning(mesh, tx, ty, tz, rx, ry, rz):
    adjusted = mesh.copy()
    center = adjusted.points.mean(axis=0)
    rotation_matrix = R.from_euler('xyz', [rx, ry, rz], degrees=True)
    adjusted.points = rotation_matrix.apply(adjusted.points - center) + center
    adjusted.points += np.array([tx, ty, tz])
    return adjusted

# =======================================================================================
# INTERFACE STREAMLIT
# =======================================================================================
st.sidebar.title("📍 Imports")

st.sidebar.divider()
uploaded_files = st.sidebar.file_uploader(
    "Charger maillages (.obj, .ply, .stl, .vtp)",
    type=["obj", "ply", "stl", "vtp", "vtk"],
    accept_multiple_files=True
)
if uploaded_files:
    names = [f.name for f in uploaded_files]
    c1, c2 = st.sidebar.columns(2)
    name_bottom = c1.selectbox("BAS", names, index=0)
    name_top    = c2.selectbox("HAUT", names, index=min(1, len(names) - 1))
    if st.sidebar.button("📥 Charger la paire brute", use_container_width=True):
        f_bot = next(f for f in uploaded_files if f.name == name_bottom)
        f_top = next(f for f in uploaded_files if f.name == name_top)
        S["mesh_bottom_raw"] = load_mesh_from_bytes(f_bot.getvalue(), os.path.splitext(f_bot.name)[1])
        S["mesh_top_raw"]    = load_mesh_from_bytes(f_top.getvalue(), os.path.splitext(f_top.name)[1])

st.header("🎯 Calage par Boîte Englobante (OBB)")

if "mesh_bottom_raw" in st.session_state.S and "mesh_top_raw" in st.session_state.S:
    
    # Préparation des maillages (Copies pour ne pas altérer les originaux bruts)
    m_bot_preview = st.session_state.S["mesh_bottom_raw"].copy()
    m_top_preview = st.session_state.S["mesh_top_raw"].copy()
    
    # --- 1. OUTILS DE FLIP DIRECT (PREVIEW) ---
    st.subheader("1. Outils de Retournement Rapide (Flip)")
    cf1, cf2 = st.columns(2)
    with cf1:
        st.markdown("**Lauze du Bas (Référence)**")
        flip_x_bot = st.checkbox("🔃 Inverser Recto-Verso (Axe X)", key="fb_x")
        flip_z_bot = st.checkbox("🔄 Pivoter Tête-Bêche (Axe Z)", key="fb_z")
    with cf2:
        st.markdown("**Lauze du Haut (Mobile)**")
        flip_x_top = st.checkbox("🔃 Inverser Recto-Verso (Axe X)", key="ft_x")
        flip_z_top = st.checkbox("🔄 Pivoter Tête-Bêche (Axe Z)", key="ft_z")
        
    # Application des Flips AVANT le calcul de l'OBB
    if flip_x_bot: m_bot_preview.rotate_x(180, inplace=True)
    if flip_z_bot: m_bot_preview.rotate_z(180, inplace=True)
    if flip_x_top: m_top_preview.rotate_x(180, inplace=True)
    if flip_z_top: m_top_preview.rotate_z(180, inplace=True)

    # --- 2. CONFIGURATION ORIENTATION ---
    st.subheader("2. Orientation de la Bounding Box")
    c_orient_1, c_orient_2 = st.columns(2)
    orient_mode_bot = c_orient_1.selectbox("Orientation Lauze BAS :", ["Automatique (PCA)", "Forcer Horizontale (Axe X principal)", "Forcer Verticale (Axe Y principal)"])
    orient_mode_top = c_orient_2.selectbox("Orientation Lauze HAUT :", ["Automatique (PCA)", "Forcer Horizontale (Axe X principal)", "Forcer Verticale (Axe Y principal)"])

    # Calcul de la Bounding Box sur les maillages (potentiellement flippés)
    anchors_bot = get_bounding_box_anchors(m_bot_preview, orient_mode_bot)
    anchors_top = get_bounding_box_anchors(m_top_preview, orient_mode_top)
    
    # --- 3. RENDU DES REPÈRES ---
    colA, colB = st.columns(2)
    with colA:
        st.image(plot_mesh_with_obb(m_bot_preview, anchors_bot, "lightgray"), caption="Lauze Référence (Bas)", use_container_width=True)
    with colB:
        st.image(plot_mesh_with_obb(m_top_preview, anchors_top, "sienna"), caption="Lauze Mobile (Haut)", use_container_width=True)

    # --- 4. TABLE D'ASSOCIATION UNIFIÉE (CODE COULEUR) ---
    st.subheader("🔗 3. Association des Couleurs")
    st.info("Associez visuellement les points de la boîte englobante. Par défaut, les couleurs identiques sont associées.")
    
    mapping = {}
    
    # On crée une liste propre avec l'émoji, la couleur et le nom pour le menu déroulant
    options_top = list(anchors_top.keys())
    options_top_affichees = [f"{INFOS_POINTS[k]['emoji']} {k}" for k in options_top]
    
    for i, pt_bot in enumerate(anchors_bot.keys()):
        info_color = INFOS_POINTS[pt_bot]
        c_badge, c_arrow, c_select = st.columns([2, 1, 3])
        
        c_badge.markdown(f"{info_color['emoji']} **{pt_bot}**")
        c_arrow.markdown("s'aligne avec 👉")
        
        # Le menu déroulant affiche l'émoji cible, rendant l'association évidente
        choix_index = c_select.selectbox(f"Cible {pt_bot}", range(len(options_top_affichees)), format_func=lambda x: options_top_affichees[x], index=i, label_visibility="collapsed")
        mapping[pt_bot] = options_top[choix_index]

    # --- 5. OPTIONS DE PRÉCISION ET CALCUL ---
    st.subheader("⚙️ 4. Paramètres de Précision")
    auto_collision = st.checkbox("🛡️ Activer la résolution automatique des collisions (Drop Test anti-pénétration)", value=True)
    
    with st.expander("🛠️ Ajustements Manuels Fins (Translation & Rotation)"):
        cm1, cm2, cm3 = st.columns(3)
        t_x = cm1.slider("Translation X (mm)", -50.0, 50.0, 0.0, 0.1)
        t_y = cm2.slider("Translation Y (mm)", -50.0, 50.0, 0.0, 0.1)
        t_z = cm3.slider("Translation Z (mm)", -50.0, 50.0, 0.0, 0.1)
        cm4, cm5, cm6 = st.columns(3)
        r_x = cm4.slider("Rotation X / Roulis (°)", -30.0, 30.0, 0.0, 0.5)
        r_y = cm5.slider("Rotation Y / Tangage (°)", -30.0, 30.0, 0.0, 0.5)
        r_z = cm6.slider("Rotation Z / Lacet (°)", -180.0, 180.0, 0.0, 1.0)

    if st.button("🔄 Lancer l'alignement mathématique", type="primary", use_container_width=True):
        with st.spinner("Exécution du pipeline géométrique (Procruste + KDTree)..."):
            
            matrix_bot = np.array([anchors_bot[k] for k in anchors_bot.keys()])
            matrix_top = np.array([anchors_top[mapping[k]] for k in anchors_bot.keys()])
            
            # On aligne le maillage "pré-flippé"
            caled_top = align_meshes_multi_points(m_bot_preview, m_top_preview, matrix_bot, matrix_top)
            
            if auto_collision:
                caled_top, z_offset_applied = resolve_mesh_collisions(m_bot_preview, caled_top)
                st.info(f"🛡️ Physique : La lauze supérieure a été déplacée verticalement de **{z_offset_applied:.2f} mm** pour corriger l'interpénétration.")
                
            final_top = apply_manual_fine_tuning(caled_top, t_x, t_y, t_z, r_x, r_y, r_z)
            
            # Enregistrement dans la session globale
            st.session_state.S["bot_final"] = m_bot_preview
            st.session_state.S["top_final"] = final_top
            st.session_state.S["calage_reussi"] = True

    # --- 6. EXPORT ET RÉSULTATS ---
    if st.session_state.S.get("calage_reussi", False):
        st.subheader("5. Rendu de Contrôle et Téléchargement")
        
        pl_res = pv.Plotter(window_size=[800, 450])
        pl_res.set_background("white")
        pl_res.add_mesh(st.session_state.S["bot_final"], color="lightgray", label="Lauze Bas")
        pl_res.add_mesh(st.session_state.S["top_final"], color="sienna", opacity=0.6, label="Lauze Haut")
        pl_res.add_legend()
        pl_res.view_isometric()
        st.image(pl_res.screenshot(return_img=True), use_container_width=True)
        
        format_export = st.radio("Format de fichier souhaité :", [".vtp (Format ParaView)", ".obj (Format universel)"], horizontal=True)
        suffix = ".vtp" if "vtp" in format_export else ".obj"
        
        cx1, cx2 = st.columns(2)
        with cx1:
            tmp_b = tempfile.mktemp(suffix=suffix)
            st.session_state.S["bot_final"].save(tmp_b)
            with open(tmp_b, "rb") as f:
                st.download_button(f"📥 Télécharger Lauze Bas ({suffix.upper()})", f, f"lauze_bas_alignee{suffix}", use_container_width=True)
        with cx2:
            tmp_t = tempfile.mktemp(suffix=suffix)
            st.session_state.S["top_final"].save(tmp_t)
            with open(tmp_t, "rb") as f:
                st.download_button(f"📥 Télécharger Lauze Haut ({suffix.upper()})", f, f"lauze_haut_alignee{suffix}", use_container_width=True)
else:
    st.warning("⚠️ Veuillez d'abord injecter les maillages originaux depuis la barre latérale.")