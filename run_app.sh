#!/usr/bin/env bash
# =======================================================================================
# Lancement sans Docker - Linux / macOS
# Crée un environnement virtuel Python isolé, installe les dépendances si besoin,
# puis lance l'application Streamlit.
# =======================================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"
PYTHON_BIN="python3"

# Vérifie que Python 3 est disponible
if ! command -v $PYTHON_BIN &> /dev/null; then
    echo "❌ Python 3 n'est pas trouvé. Installez-le avant de continuer."
    exit 1
fi

# Crée le venv s'il n'existe pas encore
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Création de l'environnement virtuel..."
    $PYTHON_BIN -m venv "$VENV_DIR"
fi

# Active le venv
source "$VENV_DIR/bin/activate"

# Installe/met à jour les dépendances seulement si requirements.txt a changé
HASH_FILE="$VENV_DIR/.requirements.hash"
CURRENT_HASH=$(shasum -a 256 requirements.txt 2>/dev/null | cut -d' ' -f1 || sha256sum requirements.txt | cut -d' ' -f1)

if [ ! -f "$HASH_FILE" ] || [ "$(cat "$HASH_FILE")" != "$CURRENT_HASH" ]; then
    echo "📥 Installation des dépendances (peut prendre quelques minutes la première fois)..."
    pip install --upgrade pip --quiet
    pip install -r requirements.txt
    echo "$CURRENT_HASH" > "$HASH_FILE"
else
    echo "✅ Dépendances déjà installées et à jour."
fi

# macOS : VTK/PyVista peuvent nécessiter ce flag pour le rendu off-screen
export PYVISTA_OFF_SCREEN=true

echo "🚀 Lancement de l'application sur http://localhost:8501"
streamlit run app.py
