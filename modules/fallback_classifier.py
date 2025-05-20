import numpy as np
from unidecode import unidecode
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# --- 🔧 Nova função normalizadora, fora de treinar_classificador ---
def _normalize_text(X):
    """Minúsculas + remoção de acentuação."""
    return [unidecode(x.lower()) for x in X]

def treinar_classificador(entradas, rotulos):
    normalizador = FunctionTransformer(_normalize_text, validate=False)

    classes = np.unique(rotulos)                 # ndarray ✔
    pesos   = compute_class_weight(
        class_weight="balanced", classes=classes, y=rotulos
    )
    pesos_map = dict(zip(classes, pesos))

    modelo = Pipeline([
        ("norm", normalizador),
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ("clf",   LogisticRegression(
                    max_iter=2000,
                    class_weight=pesos_map,
                    n_jobs=-1,
                 )
        ),
    ])
    return modelo
