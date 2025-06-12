from fastapi import APIRouter, HTTPException
from app.services.api_service import (
    descargar_parches, 
    extraer_parches, 
    listar_parches
)
from app.services.visualizar_encoder import visualizar_feature_maps
from app.services.preparar_tensor_input import preparar_tensor_input
from app.services.run_pipeline import run_pipeline

router = APIRouter()

# ---------------------------------------
# DESCARGAR Y EXTRAER PARCHES
# ---------------------------------------
@router.get("/descargar_parches/{image_id}", summary="Descargar ZIP, extraer y preparar tensor")
async def descargar_y_extraer(image_id: int):
    """
    Descarga un ZIP desde Google Drive, extrae los parches .tif y
    genera el archivo patches_input.pt automáticamente.
    """
    try:
        zip_path = descargar_parches(image_id)
        patch_dir = extraer_parches(zip_path)
        tensor_path = preparar_tensor_input()  # ⬅ se ejecuta justo después

        return {
            "message": "Parches descargados, extraídos y tensor preparado correctamente",
            "zip_path": zip_path,
            "patch_dir": patch_dir,
            "tensor_path": tensor_path
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------
# LISTAR PARCHES (.tif encontrados)
# ---------------------------------------
@router.get("/listar_parches/", summary="Listar Parches")
async def listar_parches_endpoint():
    try:
        parches = listar_parches()
        return {"parches": parches}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------
# VISUALIZAR FEATURE MAPS DEL ENCODER
# ---------------------------------------
@router.get("/visualizar/", summary="Visualizar Salida del Encoder")
async def visualizar():
    try:
        visualizar_feature_maps()
        return {"message": "Visualización completada"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------
# EJECUTAR PIPELINE COMPLETO (encoder + decoder)
# ---------------------------------------
@router.post("/run_pipeline/", summary="Ejecutar encoder + decoder")
async def ejecutar_pipeline():
    try:
        salida = run_pipeline()
        return {"message": "Pipeline ejecutado", "archivos_generados": salida}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
