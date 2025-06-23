import os
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

from app.models.encoder_cnn import EncoderCNN
from app.models.decoder_lstm import DecoderLSTM
from app.utils.logger import get_logger

def run_pipeline() -> dict:
    logger = get_logger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"📦 Dispositivo: {device}")

    # 1. Cargar entrada
    input_path = "data/uploads/patches_input.pt"
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {input_path}")
    x_input = torch.load(input_path).to(device)  # (B, 10, 128, 128)

    # 2. Cargar modelos y pesos entrenados
    encoder = EncoderCNN().to(device)
    decoder = DecoderLSTM(feature_dim=128).to(device)

    model_path = "data/models/best_model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError("❌ No se encontró el modelo entrenado best_model.pth")

    checkpoint = torch.load(model_path, map_location=device)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])

    encoder.eval()
    decoder.eval()
    logger.info("🧠 Pesos del modelo cargados correctamente")

    # 3. Ejecutar Encoder
    with torch.no_grad():
        encoded = encoder(x_input)  # (B, 128, 32, 32)
    logger.info(f"🧠 Salida del encoder: {encoded.shape}")

    encoder_path = "data/outputs/encoder/salida_encoder.pt"
    os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
    torch.save(encoded, encoder_path)

    # 4. Preparar secuencia para LSTM
    B, C, H, W = encoded.shape
    x_seq = encoded.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, 1024, 128)

    # 5. Ejecutar Decoder
    with torch.no_grad():
        raw_output = decoder(x_seq)  # (B, 2, 128, 128)
        output = F.softmax(raw_output, dim=1)

    logger.info(f"✅ Salida del decoder: {output.shape}")
    logger.info(f"🧪 NaN en raw_output: {torch.isnan(raw_output).any()}")
    logger.info(f"🧪 NaN en output: {torch.isnan(output).any()}")

    if torch.isnan(output).any():
        logger.error("❌ La salida del decoder contiene NaN. Se aborta la visualización.")
        raise ValueError("La salida del modelo contiene NaN. Revisa el entrenamiento o usa datos de prueba.")

    decoder_path = "data/outputs/decoder/salida_decoder.pt"
    os.makedirs(os.path.dirname(decoder_path), exist_ok=True)
    torch.save(output, decoder_path)

    # 6. Visualizar clase 1 (riesgo) - solo imagen 0
    pred_class_1 = output[:, 1, :, :][0].detach().cpu().numpy()
    logger.info(f"📊 Mínimo: {pred_class_1.min()}, Máximo: {pred_class_1.max()}")
    logger.info(f"🔍 Valores únicos: {torch.unique(output).shape[0]} aproximadamente")

    output_img_path = "data/outputs/mapas/mapa_riesgo_0.png"
    os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
    plt.imshow(pred_class_1, cmap="hot", vmin=0, vmax=1)
    plt.colorbar()
    plt.title("Mapa de riesgo")
    plt.savefig(output_img_path)
    plt.close()
    logger.info(f"🖼️ Imagen guardada: {output_img_path}")

    return {
        "message": "Pipeline ejecutado con modelo entrenado",
        "encoder_output": encoder_path,
        "decoder_output": decoder_path,
        "map_image": output_img_path
    }

# (opcional)
if __name__ == "__main__":
    run_pipeline()
