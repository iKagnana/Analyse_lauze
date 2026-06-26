"""
=======================================================================================
 PAGE : RECHERCHE SCIENTIFIQUE (1 LAUZE - RUGOSITÉ WENZEL) — VERSION OPTIMISÉE
=======================================================================================
"""

import os
import io
import tempfile
import heapq

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

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
# FONCTIONS NOYAU (CACHÉES)
# =======================================================================================

# FIX CRITIQUE : cache_data (sérialisable) au lieu de cache_resource (fuite mémoire prod)
@st.cache_data(show_spinner="Lecture du maillage 3D...", max_entries=4)
def load_and_orient_mesh(file_bytes: bytes, file_suffix: str):
    """Charge + oriente un maillage 3D. cache_data = libérable, safe en prod."""
    import pyvista as pv
    pv.OFF_SCREEN = True
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    mesh = pv.read(tmp_path)
    os.unlink(tmp_path)  # Nettoyage immédiat

    pts = mesh.points - mesh.points.mean(axis=0)
    cov = np.cov(pts, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvectors = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
    mesh = mesh.copy()
    mesh.points = np.dot(pts, eigenvectors)
    mesh.rotate_z(90, inplace=True)
    if mesh.points[:, 2].mean() < 0:
        mesh.points[:, 2] *= -1
    return mesh


@st.cache_data(show_spinner="Création de la grille 2D...", max_entries=8)
def build_single_grid(points: np.ndarray, pas: float = 0.1):
    """Interpole un nuage de points sur une grille régulière."""
    x_val, y_val, z_val = points[:, 0], points[:, 1], points[:, 2]
    grid_x, grid_y = np.meshgrid(
        np.arange(x_val.min(), x_val.max(), pas),
        np.arange(y_val.min(), y_val.max(), pas)
    )
    grid_z = griddata((x_val, y_val), z_val, (grid_x, grid_y), method="linear")
    return grid_x, grid_y, grid_z


@st.cache_data(show_spinner="Filtrage SNIP (Ondulation / Rugosité)...", max_entries=8)
def get_profiles_2D(original_profile: np.ndarray, m: int = 45):
    """
    OPTIMISÉ :
    - Suppression du tableau W intermédiaire
    - Early-stop si convergence
    """
    original_profile_local = np.nan_to_num(original_profile, nan=np.nanmean(original_profile))
    Z_ondulation = original_profile_local.copy()
    Ny, Nx = Z_ondulation.shape

    for p in range(1, m + 1):
        P1 = Z_ondulation[2*p:Ny,   2*p:Nx]
        P2 = Z_ondulation[0:Ny-2*p, 2*p:Nx]
        P3 = Z_ondulation[2*p:Ny,   0:Nx-2*p]
        P4 = Z_ondulation[0:Ny-2*p, 0:Nx-2*p]
        S1 = np.maximum(Z_ondulation[2*p:Ny,   p:Nx-p],  (P1+P3)/2) - (P1+P3)/2
        S2 = np.maximum(Z_ondulation[p:Ny-p,   2*p:Nx],  (P1+P2)/2) - (P1+P2)/2
        S3 = np.maximum(Z_ondulation[p:Ny-p,   0:Nx-2*p],(P3+P4)/2) - (P3+P4)/2
        S4 = np.maximum(Z_ondulation[0:Ny-2*p, p:Nx-p],  (P2+P4)/2) - (P2+P4)/2

        a2 = (S1+S4)/2 + (S2+S3)/2 + (P1+P2+P3+P4)/4
        center = Z_ondulation[p:Ny-p, p:Nx-p]
        new_vals = np.minimum(center, a2)

        if np.allclose(center, new_vals, atol=1e-8):
            break
        Z_ondulation[p:Ny-p, p:Nx-p] = new_vals

    return Z_ondulation, original_profile_local - Z_ondulation


@st.cache_data(show_spinner="Simulation physique de l'eau (Modèle Wenzel)...", max_entries=16)
def simulate_capillary_single(wind_speed, rain_intensity, Ny, Nx,
                               W_matrix, p_matrix,
                               pas_mm=0.2, beta_deg=20, theta_deg=75):
    """Simule la remontée d'eau sur une seule lauze avec le facteur de rugosité r (Wenzel)."""
    gamma, rho, g, L, Cp = 72.8e-3, 1000, 9.81, 1.0, 0.8
    beta  = np.radians(beta_deg)
    theta = np.radians(theta_deg)

    dy, dx = np.gradient(p_matrix, pas_mm)
    r_bot  = np.sqrt(1 + dx**2 + dy**2)
    r_top  = 1.0  # Plaque virtuelle lisse

    W_safe   = np.clip(W_matrix, 0.01, 100.0)
    Fc       = 0.6 * (L * gamma * (r_bot + r_top) * np.cos(theta) / (W_safe * 1e-3))
    Fv       = 0.5 * 1.225 * Cp * (wind_speed ** 2) * L
    F_pluie  = rho * g * (rain_intensity * 15) * L

    dist_rel = np.arange(Ny)[:, None] * pas_mm * 1e-3
    Weight_y = rho * g * np.sin(beta) * dist_rel * L

    R = Weight_y - Fc - Fv - F_pluie

    water   = np.zeros((Ny, Nx), dtype=int)
    invaded = np.zeros((Ny, Nx), dtype=bool)
    pq      = []
    for x in range(Nx):
        heapq.heappush(pq, (R[0, x], 0, x))
        invaded[0, x] = True

    while pq:
        res, y, x = heapq.heappop(pq)
        if res > 0:
            continue
        water[y, x] = 1
        for dy_dir, dx_dir in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            ny, nx = y + dy_dir, x + dx_dir
            if 0 <= ny < Ny and 0 <= nx < Nx and not invaded[ny, nx]:
                invaded[ny, nx] = True
                heapq.heappush(pq, (R[ny, nx], ny, nx))
    return water


def plot_water_maps(title: str, results: list, mask: np.ndarray, pas_mm: float):
    """
    BUG FIX : signature corrigée — title (str) en premier, mask et pas_mm en fin.
    La version originale passait `title` à la place de `mask`, causant un crash runtime.
    """
    num = len(results)
    fig, axes = plt.subplots(1, num, figsize=(5 * num, 5))
    if num == 1:
        axes = [axes]
    Ny, Nx = mask.shape
    extent = [0, Nx * pas_mm, 0, Ny * pas_mm]

    for ax, (label, water) in zip(axes, results):
        img = np.ones((Ny, Nx, 3))
        img[mask]                    = [0.85, 0.85, 0.85]
        img[(water > 0) & mask]      = [0.0,  0.5,  1.0]
        ax.imshow(img, origin="lower", extent=extent)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xlabel("Largeur (mm)")
        if ax is axes[0]:
            ax.set_ylabel("Longueur (mm)")

    fig.suptitle(title, fontsize=10, style='italic', y=1.01)
    plt.tight_layout()
    return fig


def _fig_to_png_bytes(fig) -> bytes:
    """Convertit une figure matplotlib en PNG bytes pour stockage léger en session_state."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()


# =======================================================================================
# INTERFACE UTILISATEUR
# =======================================================================================
st.title("🔬 Recherche Scientifique - Modèle de Wenzel sur 1 Lauze")

st.sidebar.header("📁 Fichier & Résolution")
uploaded_file = st.sidebar.file_uploader(
    "Modèle 3D (.obj, .vtp...)",
    type=["obj", "ply", "vtp", "stl", "vtk"]
)
pas_grille = st.sidebar.slider("Résolution (pas_mm)", 0.05, 1.0, 0.2, 0.05)

if uploaded_file is not None:
    mesh = load_and_orient_mesh(uploaded_file.getvalue(), os.path.splitext(uploaded_file.name)[1])
    grid_x, grid_y, z_brut = build_single_grid(mesh.points, pas=pas_grille)
    z_ond, z_rug = get_profiles_2D(z_brut)

    mask = ~np.isnan(z_brut)
    z_ond[~mask] = np.nan
    z_rug[~mask] = np.nan
    Ny, Nx = z_brut.shape

    st.success(f"Maillage analysé : Matrice {Ny}×{Nx} pixels.")

    # -----------------------------------------------------------------------------------
    # EXPANDER 1 : PROFILS SNIP
    # -----------------------------------------------------------------------------------
    with st.expander("📊 1. Analyse des Profils SNIP 2D", expanded=False):
        c1, c2, c3 = st.columns(3)
        m_snip = c1.slider("Itérations SNIP", 5, 100, 45, 5)
        cut_y  = c2.slider("Ligne de coupe Y", 0, Ny - 1, Ny // 2)
        sx, ex = c3.slider("Plage X", 0, Nx - 1, (0, Nx - 1))

        fig_snip, ax = plt.subplots(3, 1, figsize=(12, 8))
        ax[0].plot(z_brut[cut_y, sx:ex],  label='Brut',       color='blue')
        ax[0].set_title(f"Profil Original (Y={cut_y})")
        ax[1].plot(z_ond[cut_y,  sx:ex],  label='Ondulation', color='orange')
        ax[1].set_title("Ondulation (Forme globale)")
        ax[2].plot(z_rug[cut_y,  sx:ex],  label='Rugosité',   color='red')
        ax[2].set_title("Rugosité (Micro-aspérités)")
        for a in ax:
            a.legend(); a.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig_snip)
        plt.close(fig_snip)

    # -----------------------------------------------------------------------------------
    # EXPANDER 2 : SIMULATION WENZEL
    # -----------------------------------------------------------------------------------
    with st.expander("💧 2. Simulation d'Infiltration (Tests de Scénarios)", expanded=True):
        c_prof, c_phys = st.columns(2)

        with c_prof:
            st.markdown("#### Topographie étudiée")
            profile_choice = st.radio("Sélecteur :", [
                "Profil Brut (Original complet)",
                "Ondulation",
                "Rugosité",
                "Surface lisse"
            ])
            st.markdown("💡 Surface lisse = matrice de zéros de même forme que l'original.")
            if "lisse" in profile_choice:
                p_matrix = np.zeros_like(z_brut)
            elif "Brut" in profile_choice:
                p_matrix = z_brut
            elif "Ondulation" in profile_choice:
                p_matrix = z_ond
            else:
                p_matrix = z_rug

        with c_phys:
            st.markdown("#### Mode & Environnement")
            mode_simu = st.radio("Condition géométrique :", ["Confiné (Entrefer virtuel)", "À l'air libre (Ruissellement)"])
            W_target  = st.slider("Écart cible (mm)", 0.5, 10.0, 2.0) if "Confiné" in mode_simu else 10000.0
            beta_deg  = st.slider("Pente du toit β (°)", 0.0, 60.0, 20.0)

        st.divider()
        st.markdown("#### Définition des Scénarios Météorologiques")
        table_md = "| Scénario | Vent (m/s) | Pluie (mm/h) | Description |\n| :--- | :---: | :---: | :--- |\n"
        for k, v in SCENARIOS_RECHERCHE.items():
            table_md += f"| **{k}** | {v['vent']} | {v['pluie']} | {v['desc']} |\n"
        st.markdown(table_md)

        mode_lancement = st.radio(
            "Méthode de test :",
            ["Lancer les scénarios prédéfinis", "Créer un scénario sur mesure"],
            horizontal=True
        )

        scenarios_to_run = []
        if "prédéfinis" in mode_lancement:
            for k, v in SCENARIOS_RECHERCHE.items():
                scenarios_to_run.append({"v": v["vent"], "p": v["pluie"], "lbl": v["desc"]})
        else:
            colA, colB = st.columns(2)
            c_vent  = colA.number_input("Vent (m/s)",  value=5.0,    min_value=0.0,    max_value=100.0,   step=0.1)
            c_pluie = colB.number_input("Pluie (mm/h)", value=0.0001, min_value=0.0001, max_value=50.0001, step=0.0001, format="%f")
            scenarios_to_run.append({
                "v": c_vent, "p": c_pluie,
                "lbl": f"Scénario sur mesure : Vent = {c_vent:.1f} m/s, Pluie = {c_pluie:.4f} mm/h"
            })

        simu_title = f"Simulation Wenzel — {profile_choice} — {mode_simu} — β={beta_deg}° — pas={W_target:.1f} mm"

        if st.button("🚀 Lancer l'analyse capillaire", type="primary"):
            with st.spinner("Calcul des gradients de Wenzel et remontées capillaires..."):
                if "Confiné" in mode_simu:
                    W_matrix = np.clip(
                        W_target + np.abs(np.nan_to_num(p_matrix, nan=50.0)), 0.1, 10.0
                    )
                else:
                    W_matrix = np.full_like(p_matrix, W_target)

                results = []
                for sc in scenarios_to_run:
                    water = simulate_capillary_single(
                        wind_speed=sc["v"], rain_intensity=sc["p"],
                        Ny=Ny, Nx=Nx,
                        W_matrix=W_matrix, p_matrix=p_matrix,
                        pas_mm=pas_grille, beta_deg=beta_deg
                    )
                    results.append((sc["lbl"], water))

                # BUG FIX : ordre des arguments corrigé (title, results, mask, pas_mm)
                fig = plot_water_maps(simu_title, results, mask, pas_grille)
                st.pyplot(fig)
                st.success("Analyse de la topographie locale terminée avec succès !")

                # Stockage PNG bytes (léger) au lieu de l'objet Figure
                png = _fig_to_png_bytes(fig)
                if "galerie" not in st.session_state:
                    st.session_state["galerie"] = []
                st.session_state["galerie"].append({"title": simu_title, "results": results, "mask": mask, "pas_mm": pas_grille, "png": png})
                st.success("✅ Résultat ajouté à la galerie !")

    # -----------------------------------------------------------------------------------
    # EXPANDER 3 : GALERIE
    # -----------------------------------------------------------------------------------
    with st.expander("🖼️ 3. Galerie des Résultats", expanded=False):
        st.markdown("Cette galerie conserve les résultats des simulations précédentes.")

        if "galerie" not in st.session_state:
            st.session_state["galerie"] = []

        galerie = st.session_state["galerie"]
        if galerie:
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🗑️ Vider la galerie"):
                    st.session_state["galerie"] = []
                    st.success("Galerie vidée !")
                    st.rerun()

            with c2:
                if st.button("📦 Créer un ZIP de la galerie"):
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

            # Affichage des PNG bytes — aucune figure matplotlib en mémoire
            for idx, entry in enumerate(galerie):
                st.markdown(f"### {idx + 1}. {entry['title']}")
                st.image(entry["png"], use_container_width=True)
        else:
            st.info("Aucun résultat dans la galerie pour le moment. Lance une simulation pour en ajouter.")

else:
    st.info("👈 Charge un maillage 3D dans la barre latérale pour démarrer la recherche scientifique.")