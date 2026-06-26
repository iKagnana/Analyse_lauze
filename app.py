"""
=======================================================================================
 APPLICATION STREAMLIT - PIPELINE LAUZES (Version ULTRA-OPTIMISÉE 🚀)
=======================================================================================
Lancer avec :
    streamlit run app.py
"""
import os
import streamlit as st

# Configuration globale de la page
st.set_page_config(page_title="Analyse des lauzes", page_icon=":bar_chart:", layout="wide")

# Définition des pages
recherche_scientifique = st.Page("recherche-scientifique/recherche-scientifique.py", title="Recherche scientifique - 1 lauze", icon="🔬")
multiples_lauzes = st.Page("multiples-lauzes/multiples-lauzes.py", title="Analyse de lauzes superposées", icon="📚")
process = st.Page("process/alignement-manuel.py", title="Alignement manuel", icon="🛠️")




# =======================================================================================
# ÉLÉMENTS GLOBAUX DE LA BARRE LATÉRALE (SIDEBAR)
# =======================================================================================
# Tout ce qui est écrit ici apparaîtra sur absolument toutes les pages !

st.sidebar.divider()
st.sidebar.subheader("🧹 Nettoyage à la fin de la session")

if st.sidebar.button("Vider les mémoires (RAM & Disque)", use_container_width=True):
    # 1. Nettoyage du dossier 'data' (Si tu utilises l'architecture asynchrone)
    dossier_data = "data"
    fichiers_supprimes = 0
    if os.path.exists(dossier_data):
        for fichier in os.listdir(dossier_data):
            chemin_fichier = os.path.join(dossier_data, fichier)
            if os.path.isfile(chemin_fichier):
                os.remove(chemin_fichier)
                fichiers_supprimes += 1
                
    # 2. Nettoyage de la mémoire vive (Session State de Streamlit)
    if "S" in st.session_state:
        st.session_state.S.clear()
        
    st.sidebar.success(f"Nettoyage terminé ! {fichiers_supprimes} fichier(s) temporaire(s) supprimé(s).")


# CRÉATION DE LA NAVIGATION
pg = st.navigation({
    "Mode Recherche": [recherche_scientifique],
    "Mode Process": [process],
    "Mode Simulation": [multiples_lauzes]
})

# Lancement de la page sélectionnée
pg.run()