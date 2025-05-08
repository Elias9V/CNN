from fastapi import APIRouter, HTTPException
from app.services.api_service import (
    descargar_parches, 
    extraer_parches, 
    listar_parches
)
from app.services.run_encoder import process_patches
from app.services.visualizar_encoder import visualizar_feature_maps

router = APIRouter()

# ---------------------------------------
# PROCESAR PARCHES
# ---------------------------------------
@router.get("/procesar_parches/{image_id}", summary="Procesar Parches")
async def procesar_parches(image_id: int):
    """
    Descargar un ZIP de parches, descomprimir y procesar.
    """
    try:
        # Descargar y limpiar antes de extraer
        zip_path = descargar_parches(image_id)

        # Extraer los parches
        patch_dir = extraer_parches(zip_path)

        # Procesar los parches
        output_path = process_patches(patch_dir)

        return {"message": "Parches procesados", "output_path": output_path}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------
# LISTAR PARCHES
# ---------------------------------------
@router.get("/listar_parches/", summary="Listar Parches")
async def listar_parches_endpoint():
    """
    Listar todos los parches disponibles en el sistema.
    """
    try:
        parches = listar_parches()
        return {"parches": parches}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------
# VISUALIZAR SALIDA DEL ENCODER
# ---------------------------------------
@router.get("/visualizar/", summary="Visualizar Salida del Encoder")
async def visualizar():
    """
    Visualizar la salida del Encoder CNN.
    """
    try:
        visualizar_feature_maps()
        return {"message": "Visualización completada"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
