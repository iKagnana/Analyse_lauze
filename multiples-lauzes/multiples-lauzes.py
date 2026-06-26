"""
=======================================================================================
 SIMULATION DE 2 LAUZES SUPERPOSÉES — VERSION OPTIMISÉE
=======================================================================================
"""

import os
import io
import tempfile
import heapq

import numpy as np
import streamlit as st

# =======================================================================================
# CONFIG GENERALE ET CACHE
# =======================================================================================
st.set_page_config(page_title="Simulation de lauzes superposées", layout="wide", page_icon="🪨")

if "S" not in st.session_state:
    st.session_state.S = {}
S = st.session_state.S

if "galerie_multiples" not in st.session_state:
    st.session_state["galerie_multiples"] = []

# =======================================================================================
# DICTIONNAIRE DES SCÉNARIOS PRÉDÉFINIS
# =======================================================================================
SCENARIOS_RECHERCHE = {
    "Scénario 1": {"vent": 5,  "pluie": 0.0001, "desc": "Vent léger (18 km/h) avec pluie faible (0.1 mm/h)"},
    "Scénario 2": {"vent": 15, "pluie": 0.0001, "desc": "Vent soutenu (54 km/h) avec pluie faible (0.1 mm/h)"},
    "Scénario 3": {"vent": 5,  "pluie": 0.0005, "desc": "Vent léger (18 km/h) avec pluie modérée (0.5 mm/h)"},
    "Scénario 4": {"vent": 15, "pluie": 0.0005, "desc": "Tempête (90 km/h) avec pluie modérée (0.5 mm/h)"},
}

# =======================================================================================
# HELPERS 3D
# =======================================================================================

# FIX CRITIQUE : cache_data (sérialisable) au lieu de cache_resource (fuite mémoire prod)
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


def pv_screenshot(plotter):
    plotter.set_background("white")
    img = plotter.screenshot(return_img=True)
    plotter.close()
    return img


def show_mesh(plotter, key, use_3d_widget=False, disable_visu=False):
    if disable_visu:
        st.info("ℹ️ Visualisation 3D masquée (Mode Performance Activé).")
        return
    if use_3d_widget:
        try:
            from stpyvista import stpyvista
            stpyvista(plotter, key=key)
        except ImportError:
            st.image(pv_screenshot(plotter), use_container_width=True)
    else:
        st.image(pv_screenshot(plotter), use_container_width=True)


def _fig_to_png_bytes(fig) -> bytes:
    """Convertit une figure matplotlib en PNG bytes pour stockage léger en session_state."""
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()


# =======================================================================================
# FONCTIONS - ALIGNEMENT (PCA & CONTACT)
# =======================================================================================
def orient_mesh_by_extremities(mesh):
    aligned_mesh = mesh.copy()
    pts = aligned_mesh.points
    pts_centered = pts - pts.mean(axis=0)
    cov = np.cov(pts_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    sort_idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sort_idx]
    aligned_mesh.points = np.dot(pts_centered, eigenvectors)
    if aligned_mesh.points[:, 2].mean() < 0:
        aligned_mesh.points[:, 2] *= -1
    return aligned_mesh


def drop_to_realistic_contact(mesh_bottom, mesh_top, overlap_x=0.0,
                               decimate_threshold=50000, decimate_factor=0.9):
    from scipy.spatial import cKDTree
    m_bot = mesh_bottom.copy()
    m_top = mesh_top.copy()
    m_top.points[:, 0] += overlap_x
    z_offset_initial = m_bot.bounds[5] - m_top.bounds[4] + 100.0
    m_top.points[:, 2] += z_offset_initial

    tree = cKDTree(m_bot.points[:, :2])
    top_calc = m_top.decimate(decimate_factor) if m_top.n_points > decimate_threshold else m_top
    distances, indices = tree.query(top_calc.points[:, :2], k=1)

    gaps = top_calc.points[:, 2] - m_bot.points[indices, 2]
    min_gap = np.min(gaps)
    m_top.points[:, 2] -= min_gap
    return m_bot, m_top, -min_gap


def align_pipeline(mesh_bottom_raw, mesh_top_raw, pureau_mm, rotate_z_deg=90.0):
    mesh1_aligned = orient_mesh_by_extremities(mesh_bottom_raw)
    mesh2_aligned = orient_mesh_by_extremities(mesh_top_raw)
    bot_final, top_final, chute = drop_to_realistic_contact(mesh1_aligned, mesh2_aligned, overlap_x=pureau_mm)
    bot_final.rotate_z(rotate_z_deg, inplace=True)
    top_final.rotate_z(rotate_z_deg, inplace=True)
    return bot_final, top_final, chute


def count_contact_points(mesh_bottom, mesh_top, tolerance=0.5):
    from scipy.spatial import cKDTree
    tree = cKDTree(mesh_bottom.points)
    distances, _ = tree.query(mesh_top.points)
    return mesh_top.points[distances <= tolerance]


# =======================================================================================
# FONCTIONS - PROFILS / SNIP
# =======================================================================================
@st.cache_data(show_spinner="Création de la grille commune...", max_entries=4)
def build_common_grid(points_bottom: np.ndarray, points_top: np.ndarray, pas: float = 0.1):
    """
    OPTIMISÉ : un seul meshgrid, deux appels griddata vectorisés.
    Lazy import de griddata.
    """
    from scipy.interpolate import griddata

    all_xy = np.vstack((points_bottom[:, :2], points_top[:, :2]))
    x_min, x_max = all_xy[:, 0].min(), all_xy[:, 0].max()
    y_min, y_max = all_xy[:, 1].min(), all_xy[:, 1].max()

    gx = np.arange(x_min, x_max, pas)
    gy = np.arange(y_min, y_max, pas)
    grid_x, grid_y = np.meshgrid(gx, gy)

    orig_b = griddata(points_bottom[:, :2], points_bottom[:, 2], (grid_x, grid_y), method="linear")
    orig_t = griddata(points_top[:, :2],    points_top[:, 2],    (grid_x, grid_y), method="linear")
    return grid_x, grid_y, orig_b, orig_t


@st.cache_data(show_spinner="Filtrage SNIP...", max_entries=8)
def get_profiles_2D(original_profile: np.ndarray, m: int = 45):
    """
    OPTIMISÉ :
    - Suppression du tableau W intermédiaire (inutile, on écrit directement dans Z_ondulation)
    - Early-stop si la surface converge avant m itérations
    """
    original_profile_local = np.nan_to_num(original_profile, nan=np.nanmean(original_profile))
    Z_ondulation = original_profile_local.copy()
    Ny, Nx = Z_ondulation.shape

    for p in range(1, m + 1):
        P1 = Z_ondulation[2*p:Ny,    2*p:Nx]
        P2 = Z_ondulation[0:Ny-2*p,  2*p:Nx]
        P3 = Z_ondulation[2*p:Ny,    0:Nx-2*p]
        P4 = Z_ondulation[0:Ny-2*p,  0:Nx-2*p]
        S1 = np.maximum(Z_ondulation[2*p:Ny,   p:Nx-p],  (P1+P3)/2) - (P1+P3)/2
        S2 = np.maximum(Z_ondulation[p:Ny-p,   2*p:Nx],  (P1+P2)/2) - (P1+P2)/2
        S3 = np.maximum(Z_ondulation[p:Ny-p,   0:Nx-2*p],(P3+P4)/2) - (P3+P4)/2
        S4 = np.maximum(Z_ondulation[0:Ny-2*p, p:Nx-p],  (P2+P4)/2) - (P2+P4)/2

        a2 = (S1+S4)/2 + (S2+S3)/2 + (P1+P2+P3+P4)/4
        center = Z_ondulation[p:Ny-p, p:Nx-p]
        new_vals = np.minimum(center, a2)

        # Early-stop si convergence
        if np.allclose(center, new_vals, atol=1e-8):
            break
        Z_ondulation[p:Ny-p, p:Nx-p] = new_vals

    return Z_ondulation, original_profile_local - Z_ondulation


@st.cache_data(show_spinner="Calcul des masques...", max_entries=4)
def compute_masks(original_profile_bottom: np.ndarray, original_profile_top: np.ndarray):
    from scipy.ndimage import binary_dilation
    mask_bottom = ~np.isnan(original_profile_bottom)
    mask_top    = ~np.isnan(original_profile_top)
    mask_pureau  = mask_bottom & ~mask_top
    mask_overlap = mask_bottom & mask_top
    ligne_infiltration = binary_dilation(mask_pureau) & mask_overlap

    Ny = original_profile_bottom.shape[0]
    ligne_infiltration_1d = np.argmax(ligne_infiltration, axis=0)
    ligne_infiltration_1d[~np.any(ligne_infiltration, axis=0)] = Ny
    return mask_bottom, mask_top, mask_pureau, mask_overlap, ligne_infiltration_1d


# =======================================================================================
# FONCTIONS - MATPLOTLIB
# =======================================================================================
def visualize_control_masks(mask_bottom, mask_top, mask_overlap, mask_pureau):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.5))
    for ax, msk, title, cmap in zip(
        axes,
        [mask_bottom, mask_top, mask_overlap, mask_pureau],
        ["Bas", "Haut", "Recouvrement", "Pureau"],
        ["Blues", "Reds", "Purples", "Greens"]
    ):
        ax.imshow(msk, cmap=cmap, origin="lower")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    return fig


def plot_snip_profile_cuts(profil_brut, profil_ond, profil_rug, cut_y_axis, lauze_label):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(14, 10))
    ax[0].plot(profil_brut, label='Profil original (brut)', color='blue', alpha=0.5)
    ax[0].set_title(f"Profil de la {lauze_label} (Coupe Y={cut_y_axis})", fontsize=14, fontweight='bold')
    ax[0].set_xlabel("Position X (pixels)"); ax[0].set_ylabel("Hauteur Z (mm)")
    ax[1].plot(profil_ond, label='Ondulation (Ligne de base SNIP)', color='orange', linewidth=2)
    ax[1].set_title("Surface d'ondulation extraite (Macro-géométrie)", fontsize=12)
    ax[1].set_xlabel("Position X (pixels)"); ax[1].set_ylabel("Écart (mm)")
    ax[2].plot(profil_rug, label='Rugosité seule', color='red')
    ax[2].set_title("Surface de rugosité extraite (Micro-rugosité)", fontsize=12)
    ax[2].set_xlabel("Position X (pixels)"); ax[2].set_ylabel("Écart (mm)")
    for a in ax:
        a.legend(); a.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    return fig


def visualize_infiltration_map(z_bottom, mask_pureau, mask_overlap, water_matrix, pas_mm=0.02):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    Ny, Nx = z_bottom.shape
    extent = [0, Nx * pas_mm, 0, Ny * pas_mm]
    img_rgb = np.ones((Ny, Nx, 3))
    img_rgb[mask_overlap]                     = [0.85, 0.85, 0.85]
    img_rgb[mask_pureau]                      = [1.0,  0.4,  0.4]
    img_rgb[(water_matrix > 0) & mask_overlap] = [0.0,  0.5,  1.0]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(img_rgb, origin="lower", extent=extent)
    ax.set_xlabel("Largeur X (mm)"); ax.set_ylabel("Longueur toit Y (mm)")
    ax.legend(handles=[
        mpatches.Patch(color=[0.85,0.85,0.85], label='Overlap (Gris)'),
        mpatches.Patch(color=[1.0, 0.4, 0.4],  label='Pureau (Rouge)'),
        mpatches.Patch(color=[0.0, 0.5, 1.0],  label='Eau (Bleu)'),
    ])
    return fig


def visualize_multiple_infiltration_maps(z_bottom, mask_pureau, mask_overlap, scenarios_results, pas_mm=0.02):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    num_maps = len(scenarios_results)
    fig, axes = plt.subplots(1, num_maps, figsize=(7 * num_maps, 6))
    if num_maps == 1:
        axes = [axes]
    for ax, (label, water_matrix) in zip(axes, scenarios_results):
        Ny, Nx = z_bottom.shape
        extent = [0, Nx * pas_mm, 0, Ny * pas_mm]
        img_rgb = np.ones((Ny, Nx, 3))
        img_rgb[mask_overlap]                      = [0.85, 0.85, 0.85]
        img_rgb[mask_pureau]                       = [1.0,  0.4,  0.4]
        img_rgb[(water_matrix > 0) & mask_overlap] = [0.0,  0.5,  1.0]
        ax.imshow(img_rgb, origin="lower", extent=extent)
        ax.set_xlabel("Largeur X (mm)"); ax.set_ylabel("Longueur toit Y (mm)")
        ax.set_title(label)
    plt.tight_layout()
    return fig


# =======================================================================================
# FONCTIONS - SIMULATION CAPILLAIRE
# =======================================================================================
@st.cache_data(max_entries=8)
def create_W_from_superposition(z_bottom, z_top, min_spacing=0.1, max_spacing=5.0, W_target=2.0):
    W = np.abs(z_top - z_bottom)
    mean_W = np.nanmean(W)
    W = W * (W_target / mean_W) if mean_W > 0 else np.full_like(W, W_target)
    return np.clip(np.nan_to_num(W, nan=max_spacing), min_spacing, max_spacing)


@st.cache_data(show_spinner="Propagation de l'eau (Modèle de Wenzel)...", max_entries=16)
def simulate_capillary_infiltration(wind_speed, rain_intensity, Ny, Nx, W_matrix,
                                    profil_bot, profil_top,
                                    initial_water_matrix, overlap_start,
                                    gamma=72.8e-3, rho=1000, g=9.81, beta_deg=20,
                                    theta_deg=75, L=1.0, Cp=0.8, pas_mm=0.02):
    beta, theta = np.radians(beta_deg), np.radians(theta_deg)

    dy_bot, dx_bot = np.gradient(profil_bot, pas_mm)
    dy_top, dx_top = np.gradient(profil_top, pas_mm)
    r_bot = np.sqrt(1 + dx_bot**2 + dy_bot**2)
    r_top = np.sqrt(1 + dx_top**2 + dy_top**2)

    W_safe = np.clip(W_matrix, 0.01, 100.0)
    Fc = 0.6 * (L * gamma * (r_bot + r_top) * np.cos(theta) / (W_safe * 1e-3))
    Fv = 0.5 * 1.225 * Cp * (wind_speed ** 2) * L
    F_pluie = rho * g * (rain_intensity * 15) * L

    dist_rel = np.maximum(0, np.arange(Ny)[:, None] - overlap_start) * pas_mm * 1e-3
    Weight_y = rho * g * np.sin(beta) * dist_rel * L

    R = Weight_y - Fc - Fv - F_pluie

    water = initial_water_matrix.copy()
    invaded = np.zeros((Ny, Nx), dtype=bool)
    pq = []
    for x in range(Nx):
        y_start = overlap_start[x]
        if y_start < Ny:
            heapq.heappush(pq, (R[y_start, x], y_start, x))
            invaded[y_start, x] = True

    while pq:
        res, y, x = heapq.heappop(pq)
        if res > 0:
            continue
        water[y, x] = 1
        for dy, dx in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            ny_n, nx_n = y + dy, x + dx
            if (0 <= ny_n < Ny and 0 <= nx_n < Nx
                    and ny_n >= overlap_start[nx_n]
                    and not invaded[ny_n, nx_n]):
                invaded[ny_n, nx_n] = True
                heapq.heappush(pq, (R[ny_n, nx_n], ny_n, nx_n))
    return water


# =======================================================================================
# UI - SIDEBAR
# =======================================================================================
st.sidebar.title("📍 Imports & Mode")
import_type = st.sidebar.radio(
    "Que souhaites-tu charger ?",
    ["1. Maillages Bruts", "2. Maillages Alignés (.vtp)", "3. Grilles 2D (.npz)"]
)
st.sidebar.divider()

if import_type == "1. Maillages Bruts":
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
            st.sidebar.success("Maillages bruts prêts.")

elif import_type == "2. Maillages Alignés (.vtp)":
    uploaded_files = st.sidebar.file_uploader(
        "Charger maillages DÉJÀ ALIGNÉS (.vtp)",
        type=["vtp", "vtk"],
        accept_multiple_files=True
    )
    if uploaded_files:
        names = [f.name for f in uploaded_files]
        c1, c2 = st.sidebar.columns(2)
        name_bottom = c1.selectbox("BAS (Alignée)", names, index=0)
        name_top    = c2.selectbox("HAUT (Alignée)", names, index=min(1, len(names) - 1))
        if st.sidebar.button("📥 Injecter directement", use_container_width=True):
            f_bot = next(f for f in uploaded_files if f.name == name_bottom)
            f_top = next(f for f in uploaded_files if f.name == name_top)
            S["bot_final"] = load_mesh_from_bytes(f_bot.getvalue(), os.path.splitext(f_bot.name)[1])
            S["top_final"] = load_mesh_from_bytes(f_top.getvalue(), os.path.splitext(f_top.name)[1])
            st.sidebar.success("Lauzes alignées injectées !")

elif import_type == "3. Grilles 2D (.npz)":
    f = st.sidebar.file_uploader("Fichier global (.npz)", type=["npz"])
    if f and st.sidebar.button("📥 Restaurer analyse 2D", use_container_width=True):
        d = np.load(f, allow_pickle=True)
        S.update({k: d[k] for k in ["orig_b", "orig_t", "ond_b", "ond_t", "rug_b", "rug_t", "pas_mm"]})
        S["original_profile_bottom"],   S["original_profile_top"]   = S.pop("orig_b"), S.pop("orig_t")
        S["ondulation_profile_bottom"], S["ondulation_profile_top"] = S.pop("ond_b"),  S.pop("ond_t")
        S["rugosity_profile_bottom"],   S["rugosity_profile_top"]   = S.pop("rug_b"),  S.pop("rug_t")
        (S["mask_bottom"], S["mask_top"], S["mask_pureau"],
         S["mask_overlap"], S["ligne_infiltration_1d"]) = compute_masks(
            S["original_profile_bottom"], S["original_profile_top"]
        )
        st.sidebar.success("Analyse 2D restaurée depuis .npz !")

st.sidebar.divider()
disable_visu  = st.sidebar.checkbox("🚀 Mode Performance Éclair (Désactiver 3D)", value=False)
use_3d_widget = st.sidebar.checkbox("Visu 3D active (stpyvista)", value=False) if not disable_visu else False
export_dir    = st.sidebar.text_input("Chemin dossier local d'export", value=os.getcwd())

# =======================================================================================
# TABS
# =======================================================================================
st.title("🪨 Pipeline Lauzes — Simulation Infiltration d'eau - Modulaire")
tab_align, tab_profiles, tab_sim, tab_cross, tab_gallery, tab_guide = st.tabs([
    "1️⃣ Alignement 3D", "2️⃣ Profils SNIP", "3️⃣ Simulation infiltration eau",
    "📊 Coupe Transversale", "🖼️ Galerie des Résultats", "💡 Guide & Subtilités"
])

# ---------------------------------------------------------------------------------------
# TAB 1 : ALIGNEMENT
# ---------------------------------------------------------------------------------------
with tab_align:
    st.subheader("Calage Géométrique & Pureau")
    
    if "mesh_bottom_raw" not in S and "bot_final" not in S:
        st.warning("Veuillez charger des maillages dans la barre latérale.")
    else:
        # --- CHOIX DU MODE D'ALIGNEMENT ---
        mode_options = ["1. Automatique (Depuis les fichiers bruts)"]
        if S.get("calage_reussi", False):
            mode_options.append("2. Conserver l'alignement manuel OBB 🌟")
            
        mode_calage = st.radio(
            "Base de travail :", 
            mode_options, 
            index=len(mode_options)-1, # Sélectionne l'OBB par défaut s'il existe
            horizontal=True
        )
        
        c1, c2, c3 = st.columns(3)
        pureau_mm = c1.slider("Pureau (Glissement X, mm)", -50.0, 150.0, 30.0, 1.0)
        
        # La rotation Z n'est utile que pour le mode automatique
        rotate_z = c2.slider("Rotation Z globale (°)", -180.0, 180.0, 90.0, 5.0) if "1." in mode_calage else 0
        tolerance = c3.slider("Seuil contact (mm)", 0.05, 5.0, 0.5, 0.05)

        if st.button("🔄 Appliquer le Pureau et la Gravité", type="primary"):
            with st.spinner("Calcul spatial en cours..."):
                
                if "2." in mode_calage:
                    # --- MODE CONSERVATION OBB ---
                    # On récupère les lauzes parfaitement alignées manuellement
                    m_bot = S["bot_final"].copy()
                    m_top_coulisse = S["top_final"].copy()
                    
                    # 1. On la fait coulisser sur l'axe X (Pureau)
                    m_top_coulisse.points[:, 0] += pureau_mm
                    
                    # 2. On la soulève virtuellement de 10 cm pour être sûr qu'elle soit au-dessus
                    m_top_coulisse.points[:, 2] += 100.0
                    
                    # 3. Drop Test : On la laisse tomber jusqu'au premier point de contact
                    from scipy.spatial import cKDTree
                    tree = cKDTree(m_bot.points[:, :2])
                    
                    # On allège le calcul pour la recherche si la pierre est énorme
                    top_calc = m_top_coulisse.decimate(0.9) if m_top_coulisse.n_points > 50000 else m_top_coulisse
                    _, indices = tree.query(top_calc.points[:, :2], k=1)
                    
                    z_gaps = top_calc.points[:, 2] - m_bot.points[indices, 2]
                    min_gap = np.min(z_gaps)
                    
                    m_top_coulisse.points[:, 2] -= min_gap
                    
                    S["bot_final"] = m_bot
                    S["top_final"] = m_top_coulisse
                    S["chute"] = -min_gap
                    
                else:
                    # --- MODE AUTOMATIQUE ORIGINAL ---
                    S["bot_final"], S["top_final"], S["chute"] = align_pipeline(
                        S["mesh_bottom_raw"], 
                        S["mesh_top_raw"], 
                        pureau_mm, 
                        rotate_z
                    )
                    
                S.update({"last_pureau": pureau_mm, "last_rot": rotate_z})
                st.success(f"Position verrouillée ! Chute verticale effectuée : {S.get('chute', 0):.2f} mm.")

        # --- VISUALISATION ---
        if "bot_final" in S:
            if st.checkbox("👁️ Afficher la vue 3D de l'assemblage (Ralentit la page)", value=False):
                contact_pts = count_contact_points(S["bot_final"], S["top_final"], tolerance=tolerance)
                st.metric("Points d'appui trouvés (Contact strict)", len(contact_pts))
                
                import pyvista as pv
                pl = pv.Plotter(window_size=[700, 400])
                pl.set_background("white")
                pl.add_mesh(S["bot_final"], color="lightgray", label="Lauze Bas")
                pl.add_mesh(S["top_final"], color="sienna", opacity=0.8, label="Lauze Haut (Pureau appliqué)")
                
                if len(contact_pts) > 0:
                    pl.add_mesh(pv.PolyData(contact_pts).glyph(geom=pv.Sphere(radius=tolerance*2)), color="red", label="Points de contact")
                
                pl.add_legend()
                pl.view_isometric()
                st.image(pl.screenshot(return_img=True), use_container_width=True)

# ---------------------------------------------------------------------------------------
# TAB 2 : PROFILS SNIP
# ---------------------------------------------------------------------------------------
with tab_profiles:
    st.subheader("Numérisation en Grille & Filtrage SNIP")
    if "bot_final" not in S and "original_profile_bottom" not in S:
        st.warning("Nécessite des maillages en mémoire.")
    elif "original_profile_bottom" in S and "bot_final" not in S:
        st.info("ℹ️ Profils déjà chargés depuis .npz.")
    else:
        c1, c2 = st.columns(2)
        pas_grille = c1.slider("Résolution d'échantillonnage (mm)", 0.02, 1.0, 0.1, 0.02)
        m_snip     = c2.slider("Seuil de coupure SNIP (m)", 5, 100, 45, 5)

        if st.button("🔄 Décomposer les surfaces (SNIP)", type="primary"):
            grid_x, grid_y, orig_b, orig_t = build_common_grid(
                S["bot_final"].points, S["top_final"].points, pas=pas_grille
            )
            ond_b, rug_b = get_profiles_2D(orig_b, m=m_snip)
            ond_t, rug_t = get_profiles_2D(orig_t, m=m_snip)

            mb, mt = ~np.isnan(orig_b), ~np.isnan(orig_t)
            ond_b[~mb] = np.nan; rug_b[~mb] = np.nan
            ond_t[~mt] = np.nan; rug_t[~mt] = np.nan

            S.update(dict(
                grid_x=grid_x, grid_y=grid_y,
                original_profile_bottom=orig_b, original_profile_top=orig_t,
                ondulation_profile_bottom=ond_b, ondulation_profile_top=ond_t,
                rugosity_profile_bottom=rug_b, rugosity_profile_top=rug_t,
                pas_mm=pas_grille
            ))
            (S["mask_bottom"], S["mask_top"], S["mask_pureau"],
             S["mask_overlap"], S["ligne_infiltration_1d"]) = compute_masks(orig_b, orig_t)
            st.success("Surfaces filtrées prêtes.")

    if "original_profile_bottom" in S:
        st.divider()
        with st.expander("📊 Visualiser les profils SNIP"):
            Ny, Nx = S["original_profile_bottom"].shape
            c1, c2 = st.columns(2)
            cut_y_axis = c1.slider("Ligne de coupe Y", 0, Ny - 1, min(1600, Ny - 1))
            x_range = c2.slider("Zoom sur X", 0, Nx - 1, (min(1000, Nx - 1), min(1100, Nx - 1)))
            start_x, end_x = x_range

            for lauze_name, orig, ond, rug in [
                ("inférieure", S["original_profile_bottom"], S["ondulation_profile_bottom"], S["rugosity_profile_bottom"]),
                ("supérieure", S["original_profile_top"],    S["ondulation_profile_top"],    S["rugosity_profile_top"]),
            ]:
                st.markdown(f"### Profil de la lauze {lauze_name}")
                p_brut = orig[cut_y_axis, start_x:end_x]
                p_ond  = ond[cut_y_axis,  start_x:end_x]
                p_rug  = rug[cut_y_axis,  start_x:end_x]

                nz_ond = int(np.count_nonzero(~np.isnan(p_ond) & (p_ond != 0)))
                nz_rug = int(np.count_nonzero(~np.isnan(p_rug) & (p_rug != 0)))
                cm1, cm2 = st.columns(2)
                cm1.metric("Points actifs - Ondulation", f"{nz_ond:,} px")
                cm2.metric("Points actifs - Rugosité",   f"{nz_rug:,} px")

                fig = plot_snip_profile_cuts(p_brut, p_ond, p_rug, cut_y_axis, f"lauze {lauze_name}")
                st.pyplot(fig)

                # FIX : ajout galerie uniquement via bouton explicite
                if st.button(f"📌 Ajouter à la galerie ({lauze_name})", key=f"gal_snip_{lauze_name}"):
                    png = _fig_to_png_bytes(fig)
                    st.session_state["galerie_multiples"].append({
                        "png": png,
                        "label": f"Profil SNIP {lauze_name} (Y={cut_y_axis})"
                    })
                    st.success(f"✅ Ajouté à la galerie : Profil SNIP {lauze_name}")
                else:
                    import matplotlib.pyplot as plt
                    plt.close(fig)

        with st.expander("💾 Exporter les profils en .npz"):
            if st.button("💾 Générer le fichier .npz"):
                buf = io.BytesIO()
                np.savez(buf,
                    orig_b=S["original_profile_bottom"], orig_t=S["original_profile_top"],
                    ond_b=S["ondulation_profile_bottom"], ond_t=S["ondulation_profile_top"],
                    rug_b=S["rugosity_profile_bottom"],   rug_t=S["rugosity_profile_top"],
                    pas_mm=S["pas_mm"]
                )
                buf.seek(0)
                st.download_button("⬇️ Télécharger pipeline_2d.npz", buf, file_name="pipeline_2d.npz")

# ---------------------------------------------------------------------------------------
# TAB 3 : SIMULATION REMONTÉE D'EAU
# ---------------------------------------------------------------------------------------
with tab_sim:
    st.subheader("Modélisation Hydrodynamique")
    if "rugosity_profile_bottom" not in S:
        st.warning("⚠️ En attente de données d'analyse (onglet 2 ou fichier .npz).")
    else:
        simulate_default = st.checkbox("💧 Simuler les scénarios par défaut", value=False)

        profile_choice = st.radio("Sélecteur :", [
            "Profil Brut (Original complet)", "Ondulation", "Rugosité", "Surface lisse"
        ])

        if profile_choice == "Profil Brut (Original complet)":
            p_b, p_t = S["original_profile_bottom"], S["original_profile_top"]
        elif profile_choice == "Ondulation":
            p_b, p_t = S["ondulation_profile_bottom"], S["ondulation_profile_top"]
        elif profile_choice == "Rugosité":
            p_b, p_t = S["rugosity_profile_bottom"], S["rugosity_profile_top"]
        else:
            p_b = np.zeros_like(S["original_profile_bottom"])
            p_t = np.zeros_like(S["original_profile_top"])

        st.markdown("💡 Surface lisse = matrice de zéros de même forme que l'original.")

        if simulate_default:
            st.markdown("#### Définition des Scénarios Météorologiques par Défaut")
            table_md = "| Scénario | Vent (m/s) | Pluie (mm/h) | Description |\n| :--- | :---: | :---: | :--- |\n"
            for k, v in SCENARIOS_RECHERCHE.items():
                table_md += f"| **{k}** | {v['vent']} | {v['pluie']} | {v['desc']} |\n"
            st.markdown(table_md)

            if st.button("🌧️ Lancer la simulation par défaut", type="primary"):
                results = []
                W_lauzes = create_W_from_superposition(p_b, p_t, min_spacing=0.1, max_spacing=5.0, W_target=2.0)
                st.markdown(
                    f"✅ Espacement — max : {np.max(W_lauzes):.2f} mm | "
                    f"min : {np.min(W_lauzes):.2f} mm | moy : {np.mean(W_lauzes):.2f} mm"
                )
                Ny, Nx = S["rugosity_profile_bottom"].shape
                for sc in SCENARIOS_RECHERCHE.values():
                    water_matrix = simulate_capillary_infiltration(
                        sc["vent"], sc["pluie"], Ny, Nx, W_lauzes, p_b, p_t,
                        S["mask_pureau"].astype(float), S["ligne_infiltration_1d"],
                        pas_mm=S["pas_mm"]
                    )
                    results.append((sc["desc"], water_matrix))

                fig = visualize_multiple_infiltration_maps(
                    S["original_profile_bottom"], S["mask_pureau"], S["mask_overlap"],
                    results, pas_mm=S["pas_mm"]
                )
                st.pyplot(fig)
                png = _fig_to_png_bytes(fig)
                st.session_state["galerie_multiples"].append({"png": png, "label": "Simulation par défaut"})
                st.success("✅ Ajouté à la galerie : Simulation par défaut")

        else:
            c1, c2, c3, c4 = st.columns(4)
            c_vent  = c1.number_input("Vent (m/s)",  value=5.0,    min_value=0.0,    max_value=100.0,   step=0.1)
            c_pluie = c2.number_input("Pluie (mm/h)", value=0.0001, min_value=0.0001, max_value=50.0001, step=0.0001, format="%f")
            beta_d  = c3.slider("Inclinaison β (°)", 0.0, 60.0, 20.0, 1.0)
            theta_d = c4.slider("Mouillage θ (°)",   0.0, 90.0, 75.0, 1.0)

            c5, c6, c7 = st.columns(3)
            W_t   = c5.slider("Écartement cible (mm)", 0.1, 10.0, 2.0,  0.1)
            min_s = c6.slider("Entrefer min",           0.01, 2.0, 0.1,  0.01)
            max_s = c7.slider("Entrefer max",           1.0, 20.0, 5.0,  0.5)

            if st.button("🌧️ Lancer la simulation", type="primary"):
                W_lauzes = create_W_from_superposition(p_b, p_t, min_spacing=min_s, max_spacing=max_s, W_target=W_t)
                Ny, Nx = S["rugosity_profile_bottom"].shape
                S["water"]    = simulate_capillary_infiltration(
                    c_vent, c_pluie, Ny, Nx, W_lauzes, p_b, p_t,
                    S["mask_pureau"].astype(float), S["ligne_infiltration_1d"],
                    beta_deg=beta_d, theta_deg=theta_d, pas_mm=S["pas_mm"]
                )
                S["W_lauzes"] = W_lauzes
                S["last_sim_params"] = {"vent": c_vent, "pluie": c_pluie}
                st.success("Simulation terminée.")

            if "water" in S:
                fig = visualize_infiltration_map(
                    S["original_profile_bottom"], S["mask_pureau"], S["mask_overlap"],
                    S["water"], pas_mm=S["pas_mm"]
                )
                st.pyplot(fig)
                # FIX : ajout galerie uniquement via bouton explicite
                params = S.get("last_sim_params", {})
                if st.button("📌 Ajouter ce résultat à la galerie", key="gal_sim_custom"):
                    png = _fig_to_png_bytes(fig)
                    label = "Simulation Vent={vent}m/s, Pluie={pluie}mm/h".format(**params)
                    st.session_state["galerie_multiples"].append({"png": png, "label": label})
                    st.success("✅ Ajouté à la galerie")
                else:
                    import matplotlib.pyplot as plt
                    plt.close(fig)

# ---------------------------------------------------------------------------------------
# TAB 4 : COUPES & VISU 3D
# ---------------------------------------------------------------------------------------
with tab_cross:
    st.subheader("Vue 3D Interne (Coupe Dynamique)")
    if "water" not in S or "bot_final" not in S or "grid_x" not in S:
        st.warning("⚠️ Nécessite l'alignement 3D (Tab 1), les profils (Tab 2) ET la simulation (Tab 3).")
        if "water" in S:
            st.info("ℹ️ La simulation est prête, mais les maillages 3D ne sont pas encore chargés.")
    else:
        c1, c2 = st.columns(2)
        activer_coupe = c1.checkbox("✂️ Activer le plan de coupe 3D", value=True)
        bounds = S["bot_final"].bounds
        xmin, xmax, ymin, ymax = bounds[0], bounds[1], bounds[2], bounds[3]

        clip_normal, clip_origin = (0, 1, 0), (0, (ymin + ymax) / 2, 0)  # défaut
        if activer_coupe:
            axe_coupe = c1.radio("Axe de coupe", ["Y (Sens de la pente)", "X (Largeur)"], horizontal=True)
            if axe_coupe.startswith("Y"):
                val_coupe = c2.slider("Position Y (mm)", float(ymin), float(ymax), float((ymin+ymax)/2))
                clip_normal, clip_origin = (0, 1, 0), (0, val_coupe, 0)
            else:
                val_coupe = c2.slider("Position X (mm)", float(xmin), float(xmax), float((xmin+xmax)/2))
                clip_normal, clip_origin = (1, 0, 0), (val_coupe, 0, 0)

        if st.button("👁️ Construire la Scène 3D", type="primary"):
            import pyvista as pv
            pv.OFF_SCREEN = True
            with st.spinner("Modélisation du volume et découpe..."):
                pl = pv.Plotter(window_size=[900, 600])
                water_z = np.nan_to_num(
                    S["original_profile_bottom"].copy() + 0.5,
                    nan=np.nanmean(S["original_profile_bottom"])
                )
                grid = pv.StructuredGrid(S["grid_x"], S["grid_y"], water_z)
                grid.point_data["water_level"] = S["water"].ravel(order="C")
                water_mesh = grid.threshold(0.5, scalars="water_level")

                if activer_coupe:
                    bot_disp   = S["bot_final"].clip(normal=clip_normal, origin=clip_origin, invert=False)
                    top_disp   = S["top_final"].clip(normal=clip_normal, origin=clip_origin, invert=False)
                    water_disp = water_mesh.clip(normal=clip_normal, origin=clip_origin, invert=False) if water_mesh.n_points > 0 else water_mesh
                else:
                    bot_disp, top_disp, water_disp = S["bot_final"], S["top_final"], water_mesh

                pl.add_mesh(bot_disp, color="gray",   opacity=1.0, label="Lauze Bas")
                pl.add_mesh(top_disp, color="sienna", opacity=0.4, label="Lauze Haut")
                if water_disp.n_points > 0:
                    pl.add_mesh(water_disp, color="blue", opacity=0.9, label="Eau")
                pl.add_legend()
                img_bytes = pl.screenshot(transparent_background=True, return_img=True)
                st.image(img_bytes, caption="Vue 3D Interne", use_container_width=True)
                pl.close()

# ---------------------------------------------------------------------------------------
# TAB 5 : GALERIE
# ---------------------------------------------------------------------------------------
with tab_gallery:
    st.subheader("🖼️ Galerie des Résultats")
    st.markdown("Cette galerie conserve les résultats des simulations précédentes.")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🗑️ Vider la galerie"):
            st.session_state["galerie_multiples"] = []
            st.success("Galerie vidée !")

    with c2:
        galerie = st.session_state.get("galerie_multiples", [])
        if st.button("📦 Créer un ZIP de la galerie") and galerie:
            import shutil
            with tempfile.TemporaryDirectory() as tmpdirname:
                for idx, entry in enumerate(galerie):
                    png_path = os.path.join(tmpdirname, f"result_{idx + 1}.png")
                    with open(png_path, "wb") as f_out:
                        f_out.write(entry["png"])
                zip_base = os.path.join(tmpdirname, "galerie_resultats")
                shutil.make_archive(zip_base, "zip", tmpdirname, ".")
                with open(zip_base + ".zip", "rb") as fz:
                    st.download_button(
                        "⬇️ Télécharger la galerie (.zip)", fz,
                        file_name="galerie_resultats.zip", mime="application/zip"
                    )
        elif not galerie:
            st.warning("La galerie est vide.")

    galerie = st.session_state.get("galerie_multiples", [])
    if galerie:
        for idx, entry in enumerate(galerie):
            st.markdown(f"### {idx + 1}. {entry['label']}")
            # FIX : on affiche les PNG bytes directement — aucune figure matplotlib en mémoire
            st.image(entry["png"], use_container_width=True)
    else:
        st.info("Aucun résultat dans la galerie. Lance une simulation pour en ajouter.")

# ---------------------------------------------------------------------------------------
# TAB 6 : GUIDE
# ---------------------------------------------------------------------------------------
with tab_guide:
    st.subheader("📚 Documentation Scientifique & Subtilités Techniques")
    st.markdown(
        "Ce guide rassemble la théorie physique sous-jacente au simulateur ainsi que "
        "les règles de gestion des données critiques pour assurer la reproductibilité de tes calculs."
    )

    with st.expander("🔬 1. Physique du Pipeline : Moteurs & Forces en compétition"):
        st.markdown("""
L'infiltration d'eau entre deux pierres naturelles superposées est régie par un équilibre strict de **4 forces macroscopiques** :

1. **La Pression Dynamique du Vent ($F_v$)** :  
   $$F_v = 0.5 \\cdot \\rho_{air} \\cdot C_p \\cdot V_{vent}^2 \\cdot L$$

2. **La Pression Hydrostatique de la Pluie ($F_{pluie}$)** :  
   $$F_{pluie} = \\rho_{eau} \\cdot g \\cdot (I_{pluie} \\cdot t) \\cdot L$$

3. **La Tension Capillaire ($F_c$)** :  
   $$F_c = 0.6 \\cdot \\left(\\frac{2 \\cdot L \\cdot \\gamma \\cdot \\cos(\\theta)}{W \\cdot 10^{-3}}\\right)$$

4. **Le Poids du Front d'Eau ($W_y$)** :  
   $$W_y = \\rho_{eau} \\cdot g \\cdot \\sin(\\beta) \\cdot d_{relative} \\cdot L$$

**L'algorithme de routage (Dijkstra/HeapQueue)** : L'eau se propage pixel par pixel tant que la somme des forces motrices dépasse les forces résistantes ($R \\le 0$).
""")

    with st.expander("📐 2. L'Algorithme SNIP & La Gestion Critique du Vide (NaN)"):
        st.markdown("""
L'algorithme **SNIP** est un filtre de lissage morphologique itératif.

**Le piège de la 'Contagion du Vide'** : toute opération avec un `NaN` produit un `NaN`. Sans précaution, les bords de la lauze 'fondent' sur 45 pixels.

**La solution (Technique du Masque)** :
1. Sauvegarde de l'empreinte exacte (`~np.isnan(Z)`)
2. Remplissage temporaire par la moyenne
3. Réapplication du masque après filtrage
""")

    with st.expander("💾 3. Subtilités des Fichiers : Pourquoi utiliser le format .npz ?"):
        st.markdown("""
* **`.vtp`** : Format 3D natif pour maillages triangulés. Parfait pour Blender/CloudCompare.
* **`.vtk`** : Grille structurée, idéal pour ParaView.
* **`.npz`** *(Le raccourci)* : Sauvegarde les matrices numpy pures dans un binaire compressé ultra-léger.  
  En le chargeant, **tu sautes 100% des étapes géométriques lourdes** et l'app s'allume instantanément sur l'onglet simulation.
""")

    with st.expander("📺 4. Secrets de PyVista (Off-Screen) & Inversions Matplotlib"):
        st.markdown("""
* **`pv.OFF_SCREEN = True`** : Évite le crash sur serveur (pas d'écran physique). PyVista calcule en mémoire et renvoie une image.
* **Le conflit des axes** : Matplotlib utilise `origin='lower'` pour aligner l'axe Y avec la scène 3D physique (origine en bas à gauche comme en mathématiques, pas en haut à gauche comme en imagerie).
""")