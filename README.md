# DLP Light Correction Training Guide

이 문서는 `dlp-light-correction` 레포지토리에서
**forward model 학습**, **inverse model 학습**, **physics-informed inverse 학습**,
그리고 **TrainPack 연결 / benchmark / W&B 설정**까지의 전체 흐름을 정리한 가이드입니다.
이 레포는 단순히 `python main.py`만 실행하는 구조가 아니라,
**Hydra config 조합(`train.yaml + task + model + data + loss`)으로 실험을 구성하는 방식**입니다.

---

## 1. Repository Clone & Environment Setup

### 1-1. Repository clone

```bash
cd path/to/your/workspace
git clone https://github.com/sawo0150/dlp-light-correction.git
cd dlp-light-correction
```

### 1-2. Conda environment 생성

```bash
conda env create -f environment.yml
conda activate dlp_learn
```

## 2. Overall Structure

이 레포의 핵심 구조는 다음과 같습니다.

* `main.py`
  Hydra + W&B 진입점
* `configs/train.yaml`
  공통 실험 기본 설정
* `configs/task/*.yaml`
  어떤 학습을 할지 정의
* `configs/data/*.yaml`
  어떤 TrainPack을 읽을지 정의
* `configs/model/*.yaml`
  UNet / BandUNet 등 모델 정의
* `configs/LossFunction/*.yaml`
  BCE+L1 / SigmoidL1 / PhysicsInformedLoss 정의
* `trainingScripts/*.sh`
  단일 실험 실행용 스크립트
* `sweeps/*.sh`
  비교 실험 / 도메인 스윕용 스크립트

즉, 학습은 보통 아래처럼 이해하면 됩니다.

```text
train.yaml (기본값)
 + task/*.yaml
 + data/*.yaml
 + model/*.yaml
 + LossFunction/*.yaml
 + CLI override
 --> main.py
 --> train(args)
```
---

## 3. main.py의 역할

`main.py`는 단순 실행 파일이 아니라,
이 레포에서 **config를 실제 학습 인자로 바꿔주는 허브** 역할을 합니다.
핵심 역할은 아래와 같습니다.

### 3-1. Hydra config 로드

`@hydra.main(config_path="configs", config_name="train")`로 시작합니다.
즉 기본값은 `configs/train.yaml`입니다.

### 3-2. config → args 변환

`_flatten_cfg_to_args()`에서 `cfg`를 평탄화하여 `SimpleNamespace(args)`로 바꿉니다.
이때 `model`, `data`, `task`, `LossFunction`, `optimizer`, `LRscheduler`, `wandb`, `evaluation` 등은
하위 키를 유지하면서도, 기존 코드가 기대하는 alias도 함께 생성합니다.

### 3-3. 중요한 alias 매핑

특히 아래 매핑이 중요합니다.

* `trainpack_root` → `data_trainpack_root`
* `manifest_csv` → `data_manifest_csv`
* `splits_dir` → `data_splits_dir`

즉 `configs/data/*.yaml`의 경로가 틀리면,
학습은 거의 바로 깨진다고 봐도 됩니다.

### 3-4. 결과 폴더 생성

결과는 기본적으로 아래 구조로 저장됩니다.

```text
<PROJECT_ROOT>/result/<exp_name>/
    checkpoints/
    reconstructions_val/
    wandb_logs/
    hydra_logs/
```

이 경로는 `data.PROJECT_ROOT`와 `exp_name`을 기반으로 생성됩니다. ([GitHub][3])

---

## 4. train.yaml의 의미

`configs/train.yaml`은 전체 실험의 기본 뼈대입니다.
기본값 기준으로는 아래 조합이 들어 있습니다.

* `task: inverse_physics_binary`
* `model: unet_small`
* `data: local_trainpack`
* `LossFunction: physics_informedL1`
* `optimizer: adamw`
* `LRscheduler: CosineAnnealingLR`

즉, 아무 override 없이 실행하면
기본적으로 **physics-informed inverse binary 실험**이 시작됩니다. ([GitHub][4])

### 주요 하이퍼파라미터

* `batch_size: 1`
* `training_accum_steps: 4`
* `training_amp: true`
* `num_epochs: 5`
* `lr: 1e-3`
* `num_workers: 8`

즉 실제 optimizer update는
`batch_size × accum_steps` 효과를 가지도록 설계되어 있습니다.
실제 train loop에서도 gradient accumulation과 AMP(`autocast`, `GradScaler`)를 사용합니다. ([GitHub][4])

---

## 5. data/*.yaml 설정 방법

학습 전에 **가장 먼저 수정해야 하는 파일**은 `configs/data/local_trainpack.yaml` 또는
`configs/data/local_trainpack_binary.yaml`입니다.

### 5-1. local_trainpack.yaml

```yaml
PROJECT_ROOT: /home/...
trainpack_root: /home/.../trainpacks/TrainPack_allinone_COPY_1280
manifest_csv: ${data.trainpack_root}/manifest.csv
splits_dir: ${data.trainpack_root}/splits
split_source: "manifest"
split: "train"
modes: ["binary", "gray"]
```

### 5-2. local_trainpack_binary.yaml

```yaml
PROJECT_ROOT: /home/...
trainpack_root: /home/.../trainpacks/TrainPack_binary_maskonly_COPY
manifest_csv: ${data.trainpack_root}/manifest.csv
splits_dir: ${data.trainpack_root}/splits
split_source: "manifest"
split: "train"
modes: ["binary", "gray"]
```

### 5-3. 반드시 확인할 것

#### 1) `PROJECT_ROOT`

결과 저장 기준 경로입니다.
`result/<exp_name>`가 이 아래에 생성됩니다. ([GitHub][3])

#### 2) `trainpack_root`

실제 TrainPack 폴더 위치입니다.
이 안에 `manifest.csv`, `splits/`, 이미지 폴더들이 있어야 합니다. ([GitHub][5])

#### 3) `split_source`

주석상으로는 `"manifest"` 또는 `"txt"`를 쓸 수 있어 보이지만,
현재 구현은 `manifest`만 지원합니다.
`txt`를 쓰면 `NotImplementedError`가 발생합니다. ([GitHub][5])

#### 4) `modes`

`local_trainpack_binary.yaml`은 binary-only pack을 가리키지만
현재 파일 내용상 `modes: ["binary", "gray"]`로 남아 있습니다.
binary-only pack을 쓸 때는 아래처럼 고치는 편이 안전합니다.

```yaml
modes: ["binary"]
```

이건 현재 config 내용과 파일 이름이 완전히 일치하지 않기 때문에,
명시적으로 고쳐두는 것이 좋습니다. ([GitHub][6])

---

## 6. Task별 의미

이 레포에는 크게 **forward**, **1-stage inverse**, **2-stage chain inverse**,
**3-stage chain inverse**, **physics-informed inverse**가 있습니다. ([GitHub][7])

---

### 6-1. `forward_1`: Mask → LD

Forward model은 **mask를 입력으로 받아 light distribution(LD)를 예측하는 모델**입니다.

* input: `mask_160`
* target: `ld_1280_aligned`
* out_size: `640`
* modes: `["binary", "gray"]`

즉, 기본 forward baseline은
**저해상도 mask를 입력받아 정렬된 LD를 회귀하는 모델**입니다. ([GitHub][7])

### 6-2. `forward_1_1280`

이 버전은 input을 `mask_1280`으로 바꾼 full-resolution 계열입니다.
메모리 부담은 크지만, 1280 기준 실험에 더 가깝습니다. ([GitHub][8])

---

### 6-3. `inverse_1_binary`: Thresholded LD → Binary Mask

* input: `thr_random`
* target: `mask_160`
* modes: `["binary"]`

즉, **thresholded LD만 보고 binary mask를 바로 복원하는 1-stage baseline**입니다. ([GitHub][9])

---

### 6-4. `inverse_1_gray`: Thresholded LD → Gray/Binary Mask

* input: `thr_random`
* target: `mask_160`
* modes: `["binary", "gray"]`

즉, **thresholded LD → mask** 구조는 같지만
학습 데이터에 gray sample도 포함하고, soft target도 다룰 수 있는 버전입니다.
또 이 task에는 domain mix 관련 옵션이 추가되어 있습니다. ([GitHub][10])

---

### 6-5. `inverse_chain_2_binary` / `inverse_chain_2_gray`

이 task들은 네가 설명한 **2-stage correction pipeline**에 해당합니다.

```text
thresholded LD -> LD -> mask
```

구체적으로는 stage가 다음과 같습니다.

* `thr2ld`
* `ld2mask`

기본 설정에서는 `ld2mask` 입력을 `pred`가 아니라 `gt`로 넣도록 되어 있어서,
처음부터 완전 end-to-end라기보다는 **teacher-forced 2-stage baseline**에 가깝습니다. ([GitHub][11])

---

### 6-6. `inverse_chain_3_gray`

이 task는 3-stage 구조입니다.

```text
thresholded LD -> LD -> DoC -> mask
```

stage는 다음과 같습니다.

* `thr2ld`
* `ld2doc`
* `doc2mask`

즉, LD와 curing representation(또는 pseudo DoC)을 분리해서 다루는 확장형 실험입니다. ([GitHub][12])

---

### 6-7. `inverse_physics_binary`

이 task는 일반 inverse와 다르게
입력이 thresholded LD가 아니라 **target mask 자체**입니다.

```text
target mask
 -> corrected mask prediction
 -> frozen forward model
 -> simulated LD
 -> soft curing
 -> target mask와 비교
```

즉, **forward model을 differentiable simulator처럼 이용하는 physics-informed inverse**입니다.
이 설정은 `configs/task/inverse_physics_binary.yaml`과
`configs/LossFunction/physics_informedL1.yaml`을 함께 봐야 정확히 이해됩니다. ([GitHub][13])

---

## 7. Model Config

### 7-1. `unet_small`

기본 U-Net입니다.

```yaml
_target_: utils.model.unet.Unet
in_chans: 1
out_chans: 1
chans: 32
num_pool_layers: 4
drop_prob: 0.0
```

### 7-2. `band_unet_small`

Band constraint가 들어간 UNet입니다.

```yaml
_target_: utils.model.unet.BandUnet
in_chans: 1
out_chans: 1
band_width: 11
fixed_logit_val: 20.0
chans: 32
num_pool_layers: 4
```

따라서 physics-informed inverse는 단순 UNet뿐 아니라,
**band-constrained corrected mask generator**로도 돌릴 수 있습니다.
실제로 `run_inverse_1_forwardLoss.sh`는 `band_unet_small`을 사용합니다. ([GitHub][14])

---

## 8. Loss Function Config

### 8-1. `BCEWithLogitsL1Loss`

binary segmentation 계열 inverse baseline에 주로 사용됩니다. ([GitHub][15])

### 8-2. `SigmoidL1Loss`

forward model처럼 연속값 회귀가 필요한 경우에 사용됩니다.
예를 들어 mask → LD regression에 적합합니다. ([GitHub][16])

### 8-3. `physics_informedL1`

이 loss는 내부에서 frozen forward model을 로드해서
예측 corrected mask를 forward simulation한 뒤, curing 결과와 target mask를 비교합니다.
DoC stage와 curing mode도 config로 제어할 수 있습니다. ([GitHub][17])

---

## 9. Forward Checkpoint와 Proxy Checkpoint

이 부분은 매우 중요합니다.

### 9-1. `train.yaml`의 benchmark용 checkpoint

`configs/train.yaml`에는 아래 값이 있습니다.

```yaml
evaluation:
  benchmark:
    proxy_checkpoint: "../result/forward_1/checkpoints/best_model.pt"
```

이 값은 **평가/benchmark용 forward simulator** 경로입니다. ([GitHub][4])

### 9-2. `physics_informedL1.yaml`의 loss 내부 checkpoint

반면 `configs/LossFunction/physics_informedL1.yaml`에는 아래 값이 있습니다.

```yaml
forward_checkpoint: "../result/forward_2/checkpoints/best_model.pt"
```

이 값은 **physics-informed loss 내부에서 실제로 로드하는 frozen forward model** 경로입니다. ([GitHub][17])

### 9-4. 추천 방법

forward 모델을 먼저 학습한 뒤, 생성된 `best_model.pt`를 기준으로
아래 두 경로를 **같은 파일로 맞춰주는 것**이 가장 안전합니다.

* `evaluation.benchmark.proxy_checkpoint`
* `LossFunction.forward_checkpoint`

예:

```bash
python main.py \
  task=inverse_physics_binary \
  model=band_unet_small \
  LossFunction=physics_informedL1 \
  evaluation.benchmark.proxy_checkpoint="../result/dlp_forward_mask2ld_baseline_v0/checkpoints/best_model.pt" \
  LossFunction.forward_checkpoint="../result/dlp_forward_mask2ld_baseline_v0/checkpoints/best_model.pt"
```

---

## 10. 먼저 해야 하는 것: Forward Model 학습

physics-informed inverse를 돌리려면,
먼저 **forward simulator checkpoint**가 있어야 합니다.

### 10-1. 기본 forward 실행

```bash
sh trainingScripts/run_forward.sh
```

이 스크립트는 대략 아래 설정으로 실행됩니다.

* `task=forward_1`
* `LossFunction=SigmoidL1Loss`
* `exp_name=dlp_forward_mask2ld_baseline_v0`
* `image.binarize_target=false`
* `evaluation.benchmark.enable=false`
* `evaluation.benchmark.proxy_checkpoint="none"`

즉, forward 학습 자체는 benchmark를 끄고 진행합니다.
학습이 끝나면 생성된 checkpoint를 inverse 쪽에 연결해야 합니다. ([GitHub][18])

### 10-2. 1280 버전 forward

```bash
sh trainingScripts/run_forward_1280.sh
```

이 스크립트는 `task=forward_1_1280`을 사용하고,
batch size를 1로 줄이고 accumulation을 8로 늘려서 메모리 부담을 완화합니다. ([GitHub][20])

---

## 11. Inverse Model 실행 방법

### 11-1. Binary baseline

```bash
bash trainingScripts/run_inverse_1_binary.sh
```

핵심 설정:

* `task=inverse_1_binary`
* `LossFunction=BCEWithLogitsL1Loss`
* `image.binarize_target=true`
* `evaluation.benchmark.inverse_post.binarize=true`

즉, 예측 결과도 benchmark에서 binary로 후처리해서 봅니다. ([GitHub][21])

---

### 11-2. Gray baseline

```bash
bash trainingScripts/run_inverse_1_gray.sh
```

핵심 설정:

* `task=inverse_1_gray`
* `LossFunction=BCEWithLogitsL1Loss`
* `image.binarize_target=false`
* `evaluation.benchmark.inverse_post.binarize=false`

즉, target과 prediction을 soft mask 관점에서 유지합니다. ([GitHub][22])

---

### 11-3. Physics-informed inverse

```bash
bash trainingScripts/run_inverse_1_forwardLoss.sh
```

핵심 설정:

* `task=inverse_physics_binary`
* `model=band_unet_small`
* `LossFunction=physics_informedL1`
* `image.binarize_target=true`
* `evaluation.benchmark.inverse_post.binarize=false`

이 실험은 corrected mask를 직접 정답과 비교하는 게 아니라,
forward + curing을 거친 결과를 target과 비교하는 실험입니다. ([GitHub][23])

---

## 12. W&B 설정

이 레포는 W&B를 조건부로 사용합니다.

### 12-1. `main.py`에서의 동작

`wandb.use_wandb=true`이면 `wandb.init()`가 호출되고,
로그 경로는 `result/<exp_name>/wandb_logs` 아래로 들어갑니다.
또 `WANDB_DIR` 환경변수도 함께 지정합니다. ([GitHub][3])

### 12-2. train.yaml에서 설정할 값

```yaml
wandb:
  project: dlp_inverse_Debugging
  entity: swpants05-seoul-national-university
  use_wandb: true
  log_every_n_iters: 1000
  log_images_per_epoch: 5
  log_images_split: "val"
  use_visLogging: true
```

([GitHub][4])

### 12-3. 실행 전 해야 할 것

```bash
wandb login
```

그리고 자신의 계정/팀에 맞게 아래를 수정해야 합니다.

* `wandb.project`
* `wandb.entity`

### 12-4. 이미지 로깅

W&B logger는 train metric뿐 아니라
예측/정답/error heatmap 형태의 시각화도 저장하도록 구현되어 있습니다. ([GitHub][24])

---

## 13. Benchmark 설정

`train.yaml`의 `evaluation.benchmark`는 inverse 결과를
forward simulator와 함께 재해석하는 평가 설정입니다. ([GitHub][4])

핵심 옵션은 아래와 같습니다.

* `proxy_checkpoint`
  평가용 forward model 경로
* `inverse_post.apply_sigmoid`
  inverse 출력이 logits이면 sigmoid 적용
* `inverse_post.binarize`
  inverse 출력 mask를 binary로 만들지 여부
* `curing.threshold`
  curing threshold
* `mask_pixelize.enable`
  coarse DMD grid 효과를 평가 시점에 반영할지 여부
* `doc.enable`
  LD → DoC stage를 추가할지 여부
* `bandopt.enable`
  band-only optimization/constraint를 평가 시점에 반영할지 여부

즉 benchmark는 단순 validation loss만 보는 것이 아니라,
**예측 mask가 실제 광학/경화 관점에서 어떻게 해석되는지**까지 함께 보기 위한 설정입니다. ([GitHub][4])

---

## 14. Sweep Scripts의 의미

`sweeps/` 아래 스크립트들은 단순 반복 실행이 아니라,
**모델군 비교**와 **domain composition 비교**를 위한 실험 묶음입니다. ([GitHub][25])

### 14-1. `run_inverse_model_sweeps.sh`

이 스크립트는 아래 계열을 한 번에 비교합니다.

* `inverse_1_gray` + BCEWithLogitsL1
* `inverse_1_gray` + SigmoidL1
* `inverse_chain_2_gray` + BCEWithLogitsL1
* `inverse_chain_2_gray` + SigmoidL1
* `inverse_physics_binary`
* `inverse_chain_2_binary`

즉, **1-stage / 2-stage / physics-informed**를 한꺼번에 비교하는 스크립트입니다. ([GitHub][26])

### 14-2. domain sweep 계열

* `run_inverse_1_forwardLoss_sweep.sh`
* `run_inverse_2_binary_domain_sweep.sh`
* `run_inverse_2_gray_domain_sweep.sh`

이 스크립트들은 B1~B5 데이터셋 비중을 바꿔가며
성능이 domain composition에 얼마나 민감한지 보는 용도입니다.
`inverse_1_gray.yaml`, `inverse_chain_2_*`에도 실제로 `domain.weights`, `max_samples`, `max_samples_unit`, `with_replacement` 같은 옵션이 들어 있습니다. ([GitHub][27])

---


## 15. Recommended Execution Order

재현/정리 관점에서 추천 순서는 아래와 같습니다.

### Step 1. TrainPack 경로 설정

`configs/data/local_trainpack.yaml` 또는 `local_trainpack_binary.yaml` 수정

### Step 2. Forward model 학습

```bash
bash trainingScripts/run_forward.sh
```

### Step 3. 생성된 checkpoint 위치 확인

예:

```text
<PROJECT_ROOT>/result/dlp_forward_mask2ld_baseline_v0/checkpoints/best_model.pt
```

### Step 4. Physics-informed config에 forward checkpoint 연결

* `evaluation.benchmark.proxy_checkpoint`
* `LossFunction.forward_checkpoint`

둘 다 같은 파일로 맞추기

### Step 5. Inverse baseline / chain / physics-informed 학습

```bash
bash trainingScripts/run_inverse_1_binary.sh
bash trainingScripts/run_inverse_1_gray.sh
bash trainingScripts/run_inverse_1_forwardLoss.sh
```

### Step 6. 필요 시 sweep 실행

```bash
bash sweeps/run_inverse_model_sweeps.sh
```

---

## 16. Example Commands

### Forward baseline

```bash
python main.py \
  task=forward_1 \
  LossFunction=SigmoidL1Loss \
  exp_name=dlp_forward_mask2ld_baseline_v0 \
  image.binarize_target=false \
  training_accum_steps=4 \
  num_epochs=4 \
  batch_size=2 \
  evaluation.benchmark.enable=false \
  evaluation.benchmark.proxy_checkpoint="none"
```

### Binary inverse baseline

```bash
python main.py \
  task=inverse_1_binary \
  LossFunction=BCEWithLogitsL1Loss \
  exp_name=dlp_inverse_binary_baseline_v1 \
  image.binarize_target=true \
  num_epochs=10 \
  training_accum_steps=4 \
  batch_size=2
```

### Gray inverse baseline

```bash
python main.py \
  task=inverse_1_gray \
  LossFunction=BCEWithLogitsL1Loss \
  exp_name=dlp_inverse_gray_baseline_v1 \
  image.binarize_target=false \
  evaluation.benchmark.inverse_post.binarize=false
```

### Physics-informed inverse

```bash
python main.py \
  data=local_trainpack_binary \
  task=inverse_physics_binary \
  model=band_unet_small \
  LossFunction=physics_informedL1 \
  exp_name=dlp_inverse_physics_bandopt_v1 \
  image.binarize_target=true \
  evaluation.benchmark.proxy_checkpoint="../result/dlp_forward_mask2ld_baseline_v0/checkpoints/best_model.pt" \
  LossFunction.forward_checkpoint="../result/dlp_forward_mask2ld_baseline_v0/checkpoints/best_model.pt"
```

---

## 17. Summary

이 레포는 크게 아래 흐름으로 이해하면 됩니다.

```text
TrainPack 준비
  ↓
configs/data/*.yaml 에서 경로 연결
  ↓
forward model 학습 (mask -> LD)
  ↓
best_model.pt 확보
  ↓
proxy_checkpoint / forward_checkpoint 연결
  ↓
inverse baseline / chain / physics-informed 학습
  ↓
benchmark + W&B로 비교
  ↓
sweep으로 domain / model family 비교
```

즉 핵심은
**(1) TrainPack 경로를 올바르게 연결하고**,
**(2) forward simulator checkpoint를 먼저 만든 뒤**,
**(3) inverse task와 physics-informed task에 그 checkpoint를 정확히 연결하는 것**입니다. ([GitHub][5])
