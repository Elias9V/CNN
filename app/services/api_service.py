import glob
import os
import requests
import zipfile
from app.utils.logger import get_logger
from app.config import BASE_URL

logger = get_logger(__name__)

UPLOAD_DIR = "data/uploads"
PATCH_DIR = os.path.join(UPLOAD_DIR, "patches")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PATCH_DIR, exist_ok=True)

def limpiar_parches():
    """
    Eliminar los archivos de parches (`.tif`) y ZIPs antes de un nuevo procesamiento.
    Mantiene los archivos `.pt` de salida del encoder.
    """
    logger.info("Limpiando parches anteriores y archivos ZIP...")

    # Eliminar los parches `.tif`
    parche_files = glob.glob(os.path.join(PATCH_DIR, "*.tif"))
    for file in parche_files:
        os.remove(file)
        logger.info(f"Eliminado parche: {file}")

    # Eliminar los archivos ZIP
    zip_files = glob.glob(os.path.join(UPLOAD_DIR, "*.zip"))
    for file in zip_files:
        os.remove(file)
        logger.info(f"Eliminado ZIP: {file}")


def descargar_parches(image_id: int) -> str:
    """
    Descargar un archivo ZIP de parches desde la API externa.
    """
    url = f"{BASE_URL}/descargar_parches_zip/{image_id}"
    zip_path = os.path.join(UPLOAD_DIR, f"parches_{image_id}.zip")

    try:
        logger.info(f"Solicitando descarga de parches con ID: {image_id}")
        response = requests.get(url)

        if response.status_code != 200:
            logger.error(f"Error al descargar parches {image_id}: {response.status_code}")
            raise Exception(f"Error al descargar parches {image_id}: {response.status_code}")

        if not response.content:
            logger.error(f"Contenido vacío al descargar parches {image_id}")
            raise Exception(f"El archivo ZIP está vacío para los parches {image_id}")

        # Guardar el archivo ZIP
        with open(zip_path, "wb") as file:
            file.write(response.content)

        logger.info(f"Parches {image_id} descargados y guardados en {zip_path}")
        return zip_path

    except Exception as e:
        logger.error(f"Error al descargar los parches {image_id}: {e}")
        raise Exception(f"Error al descargar los parches {image_id}: {e}")

def extraer_parches(zip_path: str) -> str:
    """
    Extraer los parches del archivo ZIP al directorio de parches.
    """
    try:
        logger.info(f"Extrayendo parches del archivo: {zip_path}")

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(PATCH_DIR)

        logger.info(f"Parches extraídos en {PATCH_DIR}")
        return PATCH_DIR

    except zipfile.BadZipFile as e:
        logger.error(f"Error al descomprimir {zip_path}: {e}")
        raise Exception(f"Error al descomprimir {zip_path}: {e}")

def listar_parches() -> list:
    """
    Listar todos los parches disponibles.
    """
    parches = []
    for file in os.listdir(PATCH_DIR):
        if file.endswith(".tif"):
            parches.append({
                "filename": file,
                "path": os.path.join(PATCH_DIR, file)
            })

    logger.info(f"Se encontraron {len(parches)} parches disponibles.")
    return parches
