#!/usr/bin/env python3
"""
VisionEdge Web AR — Exportador y Validador Oficial de YOLO11n + YOLO11n-Pose a ONNX Runtime Web
Compatible con Google Colab y entornos locales (Python 3.9+ / 3.10+ / 3.11+).

Instalación en Google Colab / Terminal local:
    !pip install -U ultralytics onnx onnxruntime onnxslim

Requisitos técnicos garantizados para WebGPU y WASM SIMD (iOS Safari + Chrome Android):
    1. opset=17: Soporte universal de operadores en ONNX Runtime Web (WebGPU EP y WASM SIMD EP).
    2. imgsz=640, dynamic=False: Dimensiones estáticas [1, 3, 640, 640] requeridas por WebGPU shaders.
    3. simplify=True: Fusión de nodos constantes (onnxslim) para reducir latencia en el navegador.
    4. half=False (FP32): Obligatorio para evitar fallos de operadores FP16 no soportados en WASM/Safari iOS.
"""

import json
import os
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from ultralytics import YOLO

# ==============================================================================
# CONFIGURACIÓN DE EXPORTACIÓN ONNX RUNTIME WEB (WEBGPU & WASM SIMD)
# ==============================================================================
OUTPUT_DIR = Path("public/models")
ZIP_FILENAME = Path("models_onnx_web.zip")
INPUT_SIZE = 640
OPSET_VERSION = 17

MODELS_CONFIG = [
    {
        "pt_name": "yolo11n.pt",
        "onnx_name": "yolo11n.onnx",
        "task": "detect",
        "expected_output_shape": (1, 84, 8400),  # 4 bbox (cx,cy,w,h) + 80 clases COCO
    },
    {
        "pt_name": "yolo11n-pose.pt",
        "onnx_name": "yolo11n-pose.onnx",
        "task": "pose",
        "expected_output_shape": (1, 56, 8400),  # 4 bbox + 1 conf + 17*3 keypoints (x,y,vis)
    },
]


def export_yolo_to_onnx_web(pt_model_name: str, target_onnx_path: Path) -> Path:
    """
    Descarga el modelo oficial de Ultralytics (.pt) y lo exporta a .onnx con
    opset=17, imgsz=640, simplify=True, dynamic=False y half=False (FP32 estricto).
    """
    print(f"\n{'=' * 78}")
    print(f"[1/3] Descargando y exportando {pt_model_name} -> {target_onnx_path.name} (FP32)...")
    print(f"{'=' * 78}")

    model = YOLO(pt_model_name)

    exported_path_str = model.export(
        format="onnx",
        opset=OPSET_VERSION,
        imgsz=INPUT_SIZE,
        simplify=True,
        dynamic=False,
        half=False,  # CRÍTICO: FP32 obligatorio para WebGPU + WASM en iOS Safari y Chrome Android
        nms=False,   # Salida cruda [1, 84, 8400] / [1, 56, 8400] para Class-Aware NMS + ByteTrack en JS
    )

    exported_path = Path(exported_path_str)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported_path, target_onnx_path)

    # Verificar integridad del grafo ONNX
    onnx_model = onnx.load(str(target_onnx_path))
    onnx.checker.check_model(onnx_model)

    size_mb = target_onnx_path.stat().st_size / (1024 * 1024)
    print(f"✔ Exportado y verificado: {target_onnx_path} ({size_mb:.2f} MB, opset={OPSET_VERSION}, FP32)")
    return target_onnx_path


def validate_onnx_runtime_session(onnx_path: Path, expected_shape: tuple) -> dict:
    """
    Carga el archivo .onnx con onnxruntime.InferenceSession, ejecuta una inferencia real
    con un tensor Float32 aleatorio en formato Planar NCHW [1, 3, 640, 640] (rango [0.0, 1.0])
    e imprime los nombres exactos de los nodos de entrada/salida y sus dimensiones.
    """
    print(f"\n[2/3] Validando tensores con onnxruntime.InferenceSession: {onnx_path.name}")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    # Inspección de nodos de entrada
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_shape = tuple(input_meta.shape)
    input_type = input_meta.type

    print(f"  ├─ Nodo de Entrada : name='{input_name}' | shape={list(input_shape)} | type={input_type}")

    # Inspección de nodos de salida
    outputs_info = []
    for idx, out_meta in enumerate(session.get_outputs()):
        print(f"  ├─ Nodo de Salida {idx}: name='{out_meta.name}' | shape={list(out_meta.shape)} | type={out_meta.type}")
        outputs_info.append({
            "name": out_meta.name,
            "shape": list(out_meta.shape),
            "type": out_meta.type,
        })

    # Crear tensor de prueba aleatorio Float32 Planar NCHW [1, 3, 640, 640] normalizado en [0.0, 1.0]
    dummy_input = np.random.uniform(0.0, 1.0, size=(1, 3, INPUT_SIZE, INPUT_SIZE)).astype(np.float32)

    t0 = time.perf_counter()
    outputs = session.run(None, {input_name: dummy_input})
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    primary_out = outputs[0]
    actual_out_shape = tuple(primary_out.shape)

    print(f"  ├─ Inferencia Test : {elapsed_ms:.2f} ms | output[0].shape={list(actual_out_shape)} | dtype={primary_out.dtype}")
    print(f"  └─ Rango de Salida : min={float(np.min(primary_out)):.4f}, max={float(np.max(primary_out)):.4f}")

    assert input_shape == (1, 3, INPUT_SIZE, INPUT_SIZE), (
        f"Shape de entrada inválido: se esperaba (1, 3, 640, 640) y se obtuvo {input_shape}"
    )
    assert primary_out.dtype == np.float32, (
        f"Tipo de dato inválido: se esperaba float32 para WebGPU/WASM y se obtuvo {primary_out.dtype}"
    )
    assert actual_out_shape == expected_shape, (
        f"Shape de salida inesperado: se esperaba {expected_shape} y se obtuvo {actual_out_shape}"
    )

    return {
        "file": onnx_path.name,
        "opset": OPSET_VERSION,
        "precision": "float32",
        "input": {
            "name": input_name,
            "shape": list(input_shape),
            "type": input_type,
            "layout": "Planar NCHW [1, 3, 640, 640] RGB normalizado [0.0, 1.0]",
        },
        "outputs": outputs_info,
        "verified_output_shape": list(actual_out_shape),
    }


def package_models_for_web(exported_files: list[Path], manifest: dict, zip_path: Path) -> None:
    """
    Empaqueta yolo11n.onnx, yolo11n-pose.onnx y el manifiesto de metadatos en
    models_onnx_web.zip listos para descomprimir en /public/models/.
    Si se ejecuta en Google Colab, inicia la descarga automática al navegador.
    """
    print(f"\n[3/3] Empaquetando modelos en {zip_path}...")

    manifest_path = OUTPUT_DIR / "onnx_web_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for file_path in exported_files:
            arcname = f"public/models/{file_path.name}"
            zf.write(file_path, arcname=arcname)
            print(f"  ├─ Añadido al ZIP: {arcname}")
        zf.write(manifest_path, arcname="public/models/onnx_web_manifest.json")
        print("  └─ Añadido al ZIP: public/models/onnx_web_manifest.json")

    zip_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"\n✅ Paquete listo: {zip_path.resolve()} ({zip_mb:.2f} MB)")
    print("   Extrae el contenido en la raíz de tu proyecto web (/public/models/yolo11n.onnx).")

    # Descarga automática si el script corre dentro de Google Colab
    try:
        from google.colab import files  # type: ignore
        print("\n[Google Colab] Iniciando descarga automática de models_onnx_web.zip...")
        files.download(str(zip_path))
    except ImportError:
        pass


def main() -> None:
    exported_paths: list[Path] = []
    manifest_models: dict[str, dict] = {}

    for cfg in MODELS_CONFIG:
        target_path = OUTPUT_DIR / cfg["onnx_name"]
        onnx_file = export_yolo_to_onnx_web(cfg["pt_name"], target_path)
        meta = validate_onnx_runtime_session(onnx_file, cfg["expected_output_shape"])
        exported_paths.append(onnx_file)
        manifest_models[cfg["task"]] = meta

    manifest = {
        "runtime": "onnxruntime-web",
        "execution_providers": ["webgpu", "wasm"],
        "wasm_simd": True,
        "letterbox_pad_color_rgb": [114, 114, 114],
        "models": manifest_models,
    }

    package_models_for_web(exported_paths, manifest, ZIP_FILENAME)


if __name__ == "__main__":
    main()
