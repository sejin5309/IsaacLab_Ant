# IsaacLab Ant 험지 보행

Height Scanner로 주변 높이를 관측하는 Ant의 **추론·자체 평가용 최소 배포본**입니다. 1000회 업데이트를 완료한 체크포인트 하나와 필요한 환경 코드만 포함합니다. IsaacLab의 원본 `classic/ant` 폴더를 수정할 필요가 없습니다.

1000회 모델 자체 평가: **원본 Ant 누적 보상 36.864746 ± 4.010567**, `seed=24`, 100개 첫 에피소드. 97개는 16초 도달, 0개는 넘어짐, 3개는 지도 경계 종료였습니다. 자세한 조건과 한계는 [자체 평가 보고서](docs/self_evaluation.md)에 있습니다.

400회 체크포인트에서 같은 조건으로 600회 추가 학습했습니다. 파일 내부 iteration은 999(0부터 시작)이며 이 학습 단계의 총 업데이트 수는 1000입니다. 모델명은 `ant_rough_1000.pt`, 기본 task는 `Isaac-Ant-Rough-SelfEval-v0`입니다. 이전 명칭의 task는 더 이상 등록하지 않습니다.

[외부 저장소 3곳 환경 평가와 항목별 점수](docs/cross_repository_evaluation.md)도 함께 제공합니다.

## 실행 환경

Isaac Sim과 IsaacLab이 설치되고 IsaacLab의 RSL-RL 예제가 실행되는 Linux/NVIDIA 환경이 필요합니다. 이 저장소에 시뮬레이터나 Ant USD 자산은 포함하지 않습니다. 최초 실행 시 IsaacLab의 자산 서버에 접근할 수 있어야 합니다.

| 구성 | 검증 환경 |
|---|---|
| Isaac Sim | 5.1 |
| IsaacLab 소스 기준 | `3c6e67bb5c7ada942a6d1884ab69338f57596f77` |
| 설치된 `isaaclab` 패키지 | 0.47.2 |
| Python | 3.11 |
| PyTorch | 2.7.0+cu128 |
| rsl-rl-lib | 3.0.1 |
| GPU | RTX 4060 Laptop, 8 GB |

다른 버전에서는 API·물리 엔진 차이가 있을 수 있습니다. 먼저 [IsaacLab 설치 안내](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)에 따라 호환되는 환경을 준비하세요. 기존 설치의 버전을 무조건 업데이트할 필요는 없습니다.

배포 전 이 코드로 `seed=24`, 100개 환경을 다시 실행하여 1000회 모델의 과제 실행기 평가와 100개 개별 누적 보상이 모두 일치함을 확인했습니다. [패키지 검증 기록](results/package_validation.json)은 추가 일반화 실험이 아니라 파일 분리 후 동작이 유지되는지 확인한 결과입니다. 다른 하드웨어·버전에서 소수점까지 동일함을 보장하지 않습니다.

## 빠른 시작

IsaacLab을 실행할 때 사용하던 Python/Conda 환경을 활성화한 상태에서 실행합니다.

```bash
git clone https://github.com/sejin5309/IsaacLab_Ant.git
cd IsaacLab_Ant
ANT_REPO="$(pwd)"
ISAACLAB_PATH="$HOME/IsaacLab"  # 본인의 설치 경로로 변경
export TERM=xterm
# Conda 설치를 사용한다면 런타임 라이브러리 경로 설정:
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
```

GUI로 16초 보행을 확인합니다. 실행이 끝나면 창이 닫힙니다.

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p "$ANT_REPO/scripts/play_one_episode.py" \
  --num_envs 16 --seed 24 --real-time \
  --output "$ANT_REPO/outputs/preview_01"
```

과제 조건인 100개 환경의 첫 에피소드를 평가합니다.

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p "$ANT_REPO/scripts/play_one_episode.py" \
  --headless --seed 24 --num_envs 100 \
  --checkpoint "$ANT_REPO/checkpoints/ant_rough_1000.pt" \
  --output "$ANT_REPO/outputs/eval_01"
```

결과는 터미널과 `outputs/eval_01/summary.json`에 저장됩니다. 종료 순간 보상을 포함하고 자동 reset 이후의 두 번째 에피소드는 제외합니다. `std`는 100개 에피소드 누적 보상의 모집단 표준편차입니다. 중간에 창을 닫으면 미완료 평가로 표시하고 오류 종료합니다. 결과 덮어쓰기를 막기 위해 재실행할 때는 새로운 `--output` 경로를 사용하세요.

영상까지 기록하려면 평가 명령에 `--video --video_length 960`을 추가합니다. 영상은 해당 출력 폴더의 `videos/`에 저장됩니다. 영상 길이를 줄여도 보상 집계는 100개 첫 에피소드가 끝날 때까지 계속합니다. 영상은 Ant 0을 추적하므로 전체 100개 결과를 대표하는 성공 증거로 해석하지 않습니다.

## 선택 가능한 환경

| `--task` | 지도 | 용도 |
|---|---|---|
| `Isaac-Ant-Rough-SelfEval-v0` (기본값) | seed 71, 새로운 배치와 치수 조합 | 기록된 자체 평가 조건 |
| `Isaac-Ant-Rough-Medium-v0` | seed 42, 경사 0.20·계단/블록 6 cm | 학습에 사용한 중간 난이도 형상에서 재생 |

두 환경 모두 **평가 점수는 원본 Ant의 7개 보상 항목**으로 계산합니다. `--seed`는 에피소드/초기 상태 seed이고 지도 생성 seed는 환경 설정에 별도로 고정되어 있습니다. `--show-scanner`로 스캔 지점을 표시할 수 있습니다. 관측 순서나 스캔 범위를 바꾸면 저장된 정책의 입력 의미가 달라지므로 이 체크포인트 그대로의 성능을 기대할 수 없습니다.

## 파일과 코드 읽는 순서

| 파일 | 역할 |
|---|---|
| `checkpoints/ant_rough_1000.pt` | 1000회 학습 RSL-RL 체크포인트 약 5.9 MB |
| `scripts/play_one_episode.py` | 시뮬레이터 시작 → 환경 생성 → 정책 로딩 → 첫 에피소드 평가/영상 저장 |
| `ant_rough/env_cfg.py` | 센서·382차원 관측·행동·원본 평가 보상·종료·두 가지 지도 설정 |
| `ant_rough/terrain_cfg.py` | 10×10 m 타일, 30×24 배열, 6가지 지형의 기본 정의 |
| `ant_rough/mdp/` | 관측 계산, reset 시 타일 재선택, 경계 종료, 보행 진단 |
| `ant_rough/agent_cfg.py` | 체크포인트와 호환되는 PPO 네트워크 설정 |
| `results/self_evaluation.json` | 1000회 모델 자체 평가와 100개 에피소드 원자료 |

작은 코드·문서 외에 여러 실험 모델, 학습 로그, 영상, 캐시, 로컬 절대 경로는 넣지 않았습니다. 재학습 전체 이력을 재현하는 저장소는 아닙니다.

## 정책에서 알아둘 점

- 관측은 기존 Ant의 60차원에서 몸체 높이 1개를 빼고 높이 스캔 323개를 넣은 **382차원**입니다. 스캐너 출력은 실제 정책 입력으로 사용됩니다.
- 스캔은 yaw 정렬, 간격 0.1 m, 범위 1.8×1.6 m, 전방 offset 0.5 m입니다. 몸통 기준 지형 높이 차를 ±1 m로 제한합니다. yaw 정렬은 roll/pitch에 따라 광선이 기울어지지 않도록 하며 방향 회전인 yaw는 따라갑니다.
- 8개 관절에 `JointEffortActionCfg(scale=7.5)`를 사용합니다. 관절 위치 목표를 내는 PD 제어 정책과 혼동하지 마세요.
- 학습에서는 원본 `progress`만 목표 방향 속도 `v`에 대한 `2 - (v - 2)^2 / 2`로 바꿨습니다. 2 m/s는 보상 최적점이며 물리적인 속도 제한이나 입력 속도 command가 아닙니다. **추론 시 목표 속도 숫자만 바꿔 3 m/s 정책으로 전환할 수 없습니다.**
- 학습 지형은 중간 난이도 혼합 종류이며 커리큘럼은 사용하지 않았습니다. 매 reset에는 미리 만들어진 지도의 내부 타일을 재선택합니다.
- 이 배포본은 학습 보상 대신 원본 평가 보상을 직접 import합니다. 목표 위치는 `(1000, 0, 0)`이며 16초 안에 목표 도착을 요구하는 과제가 아닙니다.

체크포인트 SHA-256:

```text
5fd3a893a7913c0aa82a0785f95bc31305c89128db7045b136cf6c83d18e2d93
```

## 과제에서 제공한 실행 스크립트를 그대로 쓰는 경우

기존 보고 점수는 과제의 `IsaacLab/scripts/reinforcement_learning/rsl_rl/play_one_episode.py`로 얻었습니다. 이 저장소의 동명 스크립트는 외부 패키지에서 바로 실행하도록 정리한 배포용 실행기입니다. 두 스크립트를 동일 파일이라고 주장하지 않습니다.

과제 실행기와 연동하려면 같은 환경에서 패키지를 설치하고, 과제 실행기의 **AppLauncher 시작 이후** task import 구간에 `import ant_rough` 한 줄을 추가하면 됩니다. 원본 `classic/ant` 코드는 변경하지 않습니다.

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p -m pip install -e "$ANT_REPO" --no-deps --no-build-isolation
"$ISAACLAB_PATH/isaaclab.sh" -p \
  "$ISAACLAB_PATH/scripts/reinforcement_learning/rsl_rl/play_one_episode.py" \
  --task Isaac-Ant-Rough-SelfEval-v0 --seed 24 --num_envs 100 \
  --checkpoint "$ANT_REPO/checkpoints/ant_rough_1000.pt" --headless
```

## 문제 해결

- `ModuleNotFoundError: isaaclab`: 일반 시스템 Python이 아닌 IsaacLab의 `isaaclab.sh -p`와 설치에 사용한 환경으로 실행하세요.
- CUDA 정책은 동작하지만 PhysX가 CPU로 fallback한다면 실행 로그와 NVIDIA 드라이버, Conda의 라이브러리 경로를 확인하세요. CPU fallback 실행은 기록된 GPU 평가와 동일 조건이 아닙니다.
- GPU 메모리가 부족하면 영상 없이 GUI 미리보기를 16개 환경으로 먼저 확인하세요. 공식 비교 점수는 다시 100개로 평가해야 합니다.
- 이 저장소에는 USD 자산을 복사하지 않았습니다. 자산 다운로드 오류는 IsaacLab 자산 서버 연결부터 확인하세요.

IsaacLab에서 파생된 코드는 [BSD-3-Clause 라이선스](LICENSE)와 원 저작권 표시를 유지했습니다. Isaac Sim 및 외부 자산의 라이선스는 각 제공자의 조건을 따릅니다.
