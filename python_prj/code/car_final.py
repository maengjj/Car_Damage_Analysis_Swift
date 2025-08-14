from pathlib import Path
import sys
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.max_open_warning": 0})

# ── 디바이스 선택: cuda → mps(Apple GPU) → cpu
def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    # 일부 버전에선 torch.backends.mps가 없을 수 있어 안전 체크
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = pick_device()
print(f"[INFO] device = {DEVICE}")

# ── 프로젝트 루트 기준 경로 안전화
#   이 스크립트를 Car_Damage_Analysis_AI/scripts 같은 폴더에서 실행한다고 가정
#   models/, image/는 프로젝트 루트에 있다고 가정
CUR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ROOT = (CUR / "..").resolve()

MODELS_DIR = ROOT / "models"
IMG_PATH   = ROOT / "image" / "111.jpg"

# ── 모델 임포트
from src.Models import Unet
from enlighten_inference import EnlightenOnnxModel

labels = ["Breakage_3", "Crushed_2", "Scratch_0", "Seperated_1"]
n_classes_damage = 2

# ── 모델 로드 헬퍼: map_location을 안전하게 처리
def load_unet(model_path: Path, num_classes: int):
    model = Unet(encoder="resnet34", pre_weight="imagenet", num_classes=num_classes)

    # mac에서 GPU 가속이 없으면 CPU로 안전 로드
    map_loc = torch.device("cpu") if DEVICE in ("cpu", "mps") else torch.device("cuda")
    state = torch.load(str(model_path), map_location=map_loc)
    model.model.load_state_dict(state)

    # MPS는 일부 버전(PyTorch 1.12+ 권장)에서만 동작.
    # 1.10.x처럼 미지원이면 그냥 cpu에 남겨둠.
    try:
        model = model.to(DEVICE)
    except Exception as e:
        print(f"[WARN] to({DEVICE}) 실패 -> cpu로 폴백: {e}")
        model = model.to("cpu")
    model.eval()
    return model

# ── 손상 모델 4개 로드
models = []
for label in labels:
    path = MODELS_DIR / f"[DAMAGE][{label}]Unet.pt"
    if not path.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {path}")
    models.append(load_unet(path, n_classes_damage))

print("[INFO] Loaded pretrained models!")

# ── 이미지 로드 & 전처리
if not IMG_PATH.exists():
    raise FileNotFoundError(f"이미지 파일이 없습니다: {IMG_PATH}")

img = cv2.imread(str(IMG_PATH))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (256, 256))

# ── 조명 보정(EnlightenGAN onnx inference)
model_light = EnlightenOnnxModel()
img = model_light.predict(img)  # uint8, HxWxC

# ── 입력 텐서 준비
def to_input_tensor(img_rgb_uint8: np.ndarray, device: str):
    x = img_rgb_uint8.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))           # C,H,W
    t = torch.from_numpy(x).unsqueeze(0)     # 1,C,H,W
    # MPS 미지원 시 에러 방지
    try:
        return t.to(device)
    except Exception as e:
        print(f"[WARN] Tensor to({device}) 실패 -> cpu 폴백: {e}")
        return t.to("cpu")

img_input = to_input_tensor(img, DEVICE)

# ── 추론 & 시각화
fig, ax = plt.subplots(1, 5, figsize=(24, 10))
ax[0].imshow(img)
ax[0].axis("off")

outputs = []
for i, model in enumerate(models):
    with torch.no_grad():
        output = model(img_input)  # [1, num_classes, H, W]

    # argmax → [H, W]
    img_output = torch.argmax(output, dim=1).detach().cpu().numpy()[0]  # (H,W)

    outputs.append(img_output)

    ax[i + 1].set_title(labels[i])
    ax[i + 1].imshow(img.astype("uint8"), alpha=0.9)
    ax[i + 1].imshow(img_output, cmap="jet", alpha=0.6)
    ax[i + 1].axis("off")

fig.set_tight_layout(True)
plt.show()
plt.pause(0.1)

# ── 면적/가격 계산
price_table = [
    120,  # Breakage_3
    90,   # Crushed_2
    60,   # Scratch_0
    90,   # Seperated_1
]

total = 0
for i, price in enumerate(price_table):
    area = int(np.sum(outputs[i]))  # bool/label mask의 합
    total += area * price
    print(f"{labels[i]}:\t영역: {area}\t가격:{area * price}원")

print(f"고객님, 총 수리비는 {total}원 입니다!")

# ── 전체 파트 세그멘테이션: 16클래스
part_model_path = MODELS_DIR / "[PART]Unet.pt"
if not part_model_path.exists():
    raise FileNotFoundError(f"파트 모델 파일이 없습니다: {part_model_path}")

n_classes_part = 16
part_model = load_unet(part_model_path, n_classes_part)

img_input_part = to_input_tensor(img, DEVICE)

with torch.no_grad():
    part_output = part_model(img_input_part)
part_pred = torch.argmax(part_output, dim=1).detach().cpu().numpy()[0]  # (H,W)

area_sum = int(np.sum(part_pred))

area_breakage  = int(np.sum(outputs[0]))
area_crushed   = int(np.sum(outputs[1]))
area_scratch   = int(np.sum(outputs[2]))
area_seperated = int(np.sum(outputs[3]))

print("[DEBUG] areas:", area_sum, area_breakage, area_crushed, area_scratch, area_seperated)

# ── 심각도 계산 (0~100 기준 → 등급 1~4)
if area_sum == 0:
    severity = 0.0
else:
    severity = (
        area_breakage * 3.0
        + area_crushed * 2.0
        + area_seperated * 1.2
        + area_scratch * 1.0
    ) * 100.0 / (3.0 * area_sum)

if 0 <= severity < 11:
    grade = 4
elif severity < 41:
    grade = 3
elif severity < 81:
    grade = 2
else:
    grade = 1

print("손상심각도 :", grade, "등급")