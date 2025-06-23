from fastapi import APIRouter, HTTPException
import os
import torch
from app.services.api_service import (
    descargar_parches, 
    extraer_parches, 
    listar_parches
)
from app.services.visualizar_encoder import visualizar_feature_maps
from app.services.preparar_tensor_input import preparar_tensor_input
from app.services.run_pipeline import run_pipeline

router = APIRouter()

@router.post("/sanear_input/")
def sanear_input():
    path = "data/uploads/patches_input.pt"

    if not os.path.exists(path):
        return {"status": "❌ ERROR", "message": "No se encontró patches_input.pt"}

    try:
        x = torch.load(path)
    except Exception as e:
        return {"status": "❌ ERROR", "message": f"No se pudo cargar el archivo: {str(e)}"}

    # Contadores iniciales
    nan_count = torch.isnan(x).sum().item()
    inf_count = (~torch.isfinite(x)).sum().item() - nan_count

    if nan_count == 0 and inf_count == 0:
        return {"status": "✅ OK", "message": "El archivo ya está limpio. No se aplicaron cambios."}

    # Limpiar tensor
    x_clean = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)

    # Sobrescribir el archivo original
    torch.save(x_clean, path)

    return {
        "status": "✅ LIMPIADO",
        "message": f"Reemplazado archivo. NaN: {nan_count}, Infinitos: {inf_count}",
        "archivo_corregido": path
    }


@router.get("/validar_input/")
def validar_input():
    path = "data/uploads/patches_input.pt"

    if not os.path.exists(path):
        return {"status": "❌ ERROR", "message": "No se encontró patches_input.pt"}

    try:
        x = torch.load(path)
    except Exception as e:
        return {"status": "❌ ERROR", "message": f"No se pudo cargar el archivo: {str(e)}"}

    # Validaciones
    if x.dtype != torch.float32:
        return {"status": "❌ ERROR", "message": f"El tipo de datos debe ser float32, no {x.dtype}"}

    if len(x.shape) != 4 or x.shape[1:] != (10, 128, 128):
        return {
            "status": "❌ ERROR",
            "message": f"Forma inválida: se esperaba (B,10,128,128), pero se obtuvo {tuple(x.shape)}"
        }

    if torch.isnan(x).any() or not torch.isfinite(x).all():
        return {"status": "❌ ERROR", "message": "El tensor contiene NaN o infinitos"}

    if x.min() < 0 or x.max() > 1:
        return {
            "status": "⚠️ ADVERTENCIA",
            "message": f"Los valores deben estar entre 0 y 1. Min: {x.min().item():.4f}, Max: {x.max().item():.4f}"
        }

    return {
        "status": "✅ OK",
        "shape": tuple(x.shape),
        "dtype": str(x.dtype),
        "min": float(x.min()),
        "max": float(x.max())
    }


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
