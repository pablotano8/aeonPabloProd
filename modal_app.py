"""Modal deployment for the Brain Tumor Segmentation Playground.

Deploys the playground as a GPU-powered web app with auto-wake.
The container stays alive for 1 hour after the last request.

Deploy:   modal deploy modal_app.py
Dev:      modal serve modal_app.py
"""

import os
import modal

app = modal.App("brain-tumor-playground")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "nnunetv2",
        "nibabel",
        "scipy",
        "numpy",
        "scikit-survival",
        "lifelines",
        "scikit-learn",
        "monai",
        "SimpleITK",
        "pandas",
        "einops",
        "fastapi",
        "python-multipart",
    )
    .run_commands(
        "mkdir -p /app/nnunet/raw /app/nnunet/preprocessed /app/nnunet/results"
    )
    .env({
        "nnUNet_raw": "/app/nnunet/raw",
        "nnUNet_preprocessed": "/app/nnunet/preprocessed",
        "nnUNet_results": "/app/nnunet/results",
    })
    .add_local_dir("checkpoints", remote_path="/app/checkpoints")
    .add_local_dir("src", remote_path="/app/src")
    .add_local_file("playground/index.html", remote_path="/app/playground/index.html")
    .add_local_dir("playground/lib", remote_path="/app/playground/lib")
)


@app.cls(
    image=image,
    gpu="T4",
    scaledown_window=3600,  # stay alive 1 hour after last request
    timeout=600,            # 10 min max per inference request
)
class Playground:
    @modal.enter()
    def load_model(self):
        """Load nnU-Net model once when the container starts."""
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        print("Loading nnU-Net model...", flush=True)
        self.predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False,
            perform_everything_on_device=True,
            device=torch.device("cuda"),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True,
        )
        self.predictor.initialize_from_trained_model_folder(
            "/app/checkpoints/segmentation",
            use_folds=("all",),
            checkpoint_name="checkpoint_final.pth",
        )
        os.makedirs("/app/playground/output", exist_ok=True)
        print("Model loaded!", flush=True)

    @modal.asgi_app()
    def serve(self):
        import json
        import shutil
        import sys
        import tempfile
        import traceback
        import numpy as np
        import nibabel as nib
        from fastapi import FastAPI, File, Form, UploadFile
        from fastapi.responses import FileResponse, JSONResponse

        web_app = FastAPI()
        predictor = self.predictor
        OUTPUT_DIR = "/app/playground/output"
        HIGH_SENS_THRESHOLD = 0.001

        @web_app.get("/")
        def index():
            return FileResponse("/app/playground/index.html", media_type="text/html")

        @web_app.get("/lib/{filename}")
        def lib_file(filename: str):
            path = f"/app/playground/lib/{filename}"
            if os.path.exists(path):
                return FileResponse(path, media_type="text/javascript")
            return JSONResponse({"error": "not found"}, status_code=404)

        @web_app.get("/output/{filename}")
        def output_file(filename: str):
            path = f"{OUTPUT_DIR}/{filename}"
            if os.path.exists(path):
                return FileResponse(path, media_type="application/octet-stream")
            return JSONResponse({"error": "not found"}, status_code=404)

        @web_app.post("/api/infer")
        def infer(
            file: UploadFile = File(...),
            age: float = Form(None),
            eor: str = Form("GTR"),
        ):
            try:
                ext = ".nii.gz" if file.filename.endswith(".gz") else ".nii"
                upload_path = os.path.join(OUTPUT_DIR, f"input_flair{ext}")
                with open(upload_path, "wb") as f:
                    f.write(file.file.read())

                print(f"Running inference on {file.filename} ({os.path.getsize(upload_path) // 1024} KB)...", flush=True)

                # Run nnU-Net in temp directory
                with tempfile.TemporaryDirectory() as tmpdir:
                    in_dir = os.path.join(tmpdir, "input")
                    out_dir = os.path.join(tmpdir, "output")
                    os.makedirs(in_dir)
                    os.makedirs(out_dir)
                    shutil.copy2(upload_path, os.path.join(in_dir, f"INFER_0000{ext}"))

                    predictor.predict_from_files(
                        in_dir, out_dir,
                        save_probabilities=True,
                        overwrite=True,
                        num_processes_preprocessing=1,
                        num_processes_segmentation_export=1,
                    )

                    pred_nii = nib.load(os.path.join(out_dir, f"INFER{ext}"))
                    seg_data = pred_nii.get_fdata().astype(np.float32)
                    affine = pred_nii.affine

                    prob_npz = np.load(os.path.join(out_dir, "INFER.npz"))
                    key = "probabilities" if "probabilities" in prob_npz else "softmax"
                    tumor_prob = prob_npz[key][1].transpose(2, 1, 0).astype(np.float32)

                # Compute derived outputs
                uncertainty = 1.0 - np.abs(2.0 * tumor_prob - 1.0)
                seg_normal = (seg_data > 0.5).astype(np.float32)
                seg_high_sens = (tumor_prob >= HIGH_SENS_THRESHOLD).astype(np.float32)

                # Save output NIfTI files
                nib.save(nib.Nifti1Image(seg_normal, affine),
                         os.path.join(OUTPUT_DIR, "segmentation.nii.gz"))
                nib.save(nib.Nifti1Image(seg_high_sens, affine),
                         os.path.join(OUTPUT_DIR, "segmentation_high_sens.nii.gz"))
                nib.save(nib.Nifti1Image(tumor_prob, affine),
                         os.path.join(OUTPUT_DIR, "tumor_probability.nii.gz"))
                nib.save(nib.Nifti1Image(uncertainty, affine),
                         os.path.join(OUTPUT_DIR, "uncertainty.nii.gz"))

                stats = {
                    "tumor_volume_cm3": float(seg_normal.sum() / 1000.0),
                    "max_tumor_prob": float(tumor_prob.max()),
                    "mean_uncertainty": float(uncertainty[seg_normal > 0].mean()) if seg_normal.sum() > 0 else 0.0,
                    "tumor_voxels": int(seg_normal.sum()),
                }

                survival = None
                if age is not None:
                    survival = _predict_survival(upload_path, seg_normal, affine, age, eor)

                print(f"Inference complete. Tumor volume: {stats['tumor_volume_cm3']:.1f} cm³", flush=True)

                return {
                    "status": "ok",
                    "files": {
                        "input": f"/output/input_flair{ext}",
                        "segmentation": "/output/segmentation.nii.gz",
                        "segmentation_high_sens": "/output/segmentation_high_sens.nii.gz",
                        "probability": "/output/tumor_probability.nii.gz",
                        "uncertainty": "/output/uncertainty.nii.gz",
                    },
                    "stats": stats,
                    "survival": survival,
                }

            except Exception:
                traceback.print_exc()
                return JSONResponse(
                    {"status": "error", "message": traceback.format_exc()[-500:]},
                    status_code=500,
                )

        if "/app/src/survival" not in sys.path:
            sys.path.insert(0, "/app/src/survival")
        from predict import predict_survival

        def _predict_survival(flair_path, seg_data, affine, age, eor):
            return predict_survival(
                flair_path, seg_data, affine, age, eor,
                checkpoint_dir="/app/checkpoints/survival",
            )

        return web_app
