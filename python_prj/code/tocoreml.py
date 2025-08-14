import torch, coremltools as ct
from pathlib import Path
from src.Models import Unet

# ckpt = "./models/[DAMAGE][Scratch_0]Unet.pt"
# ckpt = "./models/[DAMAGE][Seperated_1]Unet.pt"
# ckpt = "./models/[DAMAGE][Crushed_2]Unet.pt"
ckpt = "./models/[DAMAGE][Breakage_3]Unet.pt"
num_classes = 2
H, W = 256, 256

# 1) 모델 만들고 가중치 로드
model = Unet(encoder="resnet34", pre_weight="imagenet", num_classes=num_classes)
state = torch.load(ckpt, map_location="cpu")
# 다양한 저장 포맷 대응(필요시 키 정리)
if isinstance(state, dict) and "state_dict" in state: state = state["state_dict"]
clean = {k.replace("module.","").replace("model.",""): v for k,v in state.items()}
model.model.load_state_dict(clean, strict=False)
model.eval()

# 2) TorchScript
dummy = torch.randn(1,3,H,W)
with torch.no_grad():
    scripted = torch.jit.trace(model, dummy)
    scripted = torch.jit.freeze(scripted)

# 3) Core ML 변환 (입력: RGB 이미지, /255 스케일)
try:
    # Prefer ML Program backend (iOS 15+) so we can use compute_precision (FP16)
    mlmodel = ct.convert(
        scripted,
        inputs=[ct.ImageType(name="input", shape=(1,3,H,W), color_layout="RGB", scale=1/255.0)],
        outputs=[ct.TensorType(name="logits")],
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS15,
        convert_to="mlprogram",
    )
except ValueError as e:
    print(f"[WARN] ML Program conversion with FP16 failed ({e}). Retrying with NeuralNetwork backend (no precision override)...")
    # Fallback: NeuralNetwork backend (older iOS). Remove compute_precision.
    mlmodel = ct.convert(
        scripted,
        inputs=[ct.ImageType(name="input", shape=(1,3,H,W), color_layout="RGB", scale=1/255.0)],
        outputs=[ct.TensorType(name="logits")],
        compute_units=ct.ComputeUnit.ALL,
    )
# Choose proper file extension depending on backend
# out_path = Path("Damage_Scratch0_Unet_256.mlmodel")
# out_path = Path("Damage_Seperated1_Unet_256.mlmodel")
# out_path = Path("Damage_Crushed2_Unet_256.mlmodel")
out_path = Path("Damage_Breakage3_Unet_256.mlmodel")
try:
    spec = mlmodel.get_spec()
    is_mlprogram = spec.WhichOneof("Type") == "mlProgram"
except Exception:
    # Fallback: if convert_to was set to mlprogram, assume package
    is_mlprogram = False
if is_mlprogram and out_path.suffix != ".mlpackage":
    out_path = out_path.with_suffix(".mlpackage")

mlmodel.save(str(out_path))
print(f"Saved: {out_path}")