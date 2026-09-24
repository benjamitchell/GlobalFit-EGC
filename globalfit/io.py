"""Descarga de modelos desde BiGG.

`cobra.io.load_model` pide los modelos a http://bigg.ucsd.edu, que hoy
responde con una redirección a https que COBRApy no sigue; solo funciona si el
modelo ya estaba en su caché. Aquí se descarga directo por https.
"""

from pathlib import Path
from urllib.request import urlretrieve

from cobra import Model
from cobra.io import read_sbml_model

BIGG_URL = "https://bigg.ucsd.edu/static/models/{}.xml.gz"


def load_bigg_model(model_id: str, cache_dir: str | Path = "modelos") -> Model:
    """Carga un modelo de BiGG, descargándolo a `cache_dir` la primera vez."""
    path = Path(cache_dir) / f"{model_id}.xml.gz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(BIGG_URL.format(model_id), path)
    return read_sbml_model(str(path))
