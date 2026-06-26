"""
=======================================================================================
 APPLICATION STREAMLIT - PIPELINE LAUZES (Version ULTRA-OPTIMISÉE 🚀)
=======================================================================================
Lancer avec :
    streamlit run app.py
"""

import streamlit as st

# Configuration globale de la page
st.set_page_config(page_title="Analyse des lauzes", page_icon=":bar_chart:", layout="wide")

# Définition des pages
recherche_scientifique = st.Page("recherche-scientifique/recherche-scientifique.py", title="Recherche scientifique - 1 lauze", icon="🔬")
multiples_lauzes = st.Page("multiples-lauzes/multiples-lauzes.py", title="Analyse de lauzes superposées", icon="📚")

# CRÉATION DE LA NAVIGATION
pg = st.navigation({
    "Mode Recherche": [recherche_scientifique],
    "Mode Simulation": [multiples_lauzes]
})

# Lancement de la page sélectionnée
pg.run()