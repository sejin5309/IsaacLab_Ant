# 1000회 Ant 모델 — 세 저장소 환경 전이 평가

학습을 추가하거나 테스트 환경의 난이도를 조절하지 않고 같은 `ant_rough_1000.pt`로 평가했습니다. **다른 팀의 모델끼리 비교한 표가 아니라, 우리 모델 하나를 서로 다른 환경에서 실행한 결과**입니다.

과제 기준에 맞춰 모든 주 결과는 원본 `Isaac-Ant-v0`의 7개 reward와 weight를 사용합니다. `seed=24`, 환경 100개씩, 첫 에피소드만, 최대 16초/960 step, deterministic inference입니다. 마지막 종료 step의 보상도 포함하며 auto-reset 뒤 에피소드는 제외합니다. ±는 population std입니다.

## 원본 보상 기준 결과

| 테스트 환경 | return mean ± std | step mean ± std | 해당 환경의 실패 종료 | 16초까지 도달 |
|---|---:|---:|---:|---:|
| sejin5309/IsaacLab_Ant | **36.864746 ± 4.010567** | 956.05 ± 28.596984 | 0/100 | 97/100 |
| zxro-z/IsaacLab_RS_Assignment1 | **19.296631 ± 11.834774** | 645.43 ± 324.400594 | 60/100 | 40/100 |
| dding222/ant_rl | **31.133374 ± 14.306796** | 786.08 ± 337.458729 | 24/100 | 76/100 |

본인 환경은 지도 경계 종료가 별도로 3개입니다. timeout은 코스 완주를 뜻하지 않으며, 저장소마다 실패 정의와 시작 조건이 다르므로 실패율을 같은 의미로 순위화하지 않습니다.

## 항목별 평균 누적 기여

| 원본 보상 항목 | 본인 혼합 지형 | Final-Unseen 코스 | ant_rl 혼합 지형 |
|---|---:|---:|---:|
| progress | +27.463116 | +13.741506 | +23.355128 |
| alive | +7.967083 | +5.373583 | +6.548667 |
| upright | +1.528317 | +0.952667 | +1.267983 |
| move_to_target | +7.942057 | +4.863706 | +6.499056 |
| action_l2 | -0.098038 | -0.066815 | -0.081321 |
| energy | -4.599823 | -2.594680 | -3.668687 |
| joint_pos_limits | -3.337966 | -2.973337 | -2.787451 |
| **합계** | **36.864746** | **19.296631** | **31.133374** |

각 항목은 `raw term × weight × 1/60 s`의 에피소드 누적값입니다. RewardManager의 실제 step 결과를 읽어 집계했으며 reward 함수를 재호출하지 않았습니다. 모든 주 평가에서 항목 합과 실제 return의 최대 개별 오차는 7e-7 미만입니다. `progress`는 목표까지 거리의 감소, `move_to_target`은 방향 정렬, `energy`는 원본 proxy이며 실제 Joule이 아닙니다.

## 사용한 환경과 유지한 조건

| 항목 | 본인 저장소 | Assignment1 | ant_rl |
|---|---|---|---|
| 선택 환경 | SelfEval / map71 | README의 Isaac-Ant-Final-Unseen-v0 | Ant-rl-v0 |
| 지도 | 10×10 m 타일, 30×24 | 40×8 m 코스, 10×10 반복 | 10×10 m 타일, 20×10 |
| 지형 seed | 71 | 2404 | 42 |
| 내용 | 평지·경사/역경사·계단/역계단·블록 | 비대칭 ridge·연결 terraces·basin·smooth landing | 계단/역계단·경사/역경사·블록 |
| 마찰 | 고정 1.0, multiply | 고정 1.0, average | robot static U(0.3,1.0), dynamic=0.8×static, multiply |
| 시작 | 내부 타일 재선택, z 추가 15cm | 코스 local (2,4), 원본 reset | 원래 tile 배정, z 추가 없음 |
| 실패 | 몸통 기울기 >90° | 지면 대비 몸통 높이 <0.31m 또는 ray miss | 몸통 기울기 >90° |
| 지도 경계 종료 | 끝 2m 이전 truncation | 별도 없음 | 별도 없음 |

각 저장소 코드는 아래 commit으로 고정했습니다. 테스트 도중 지형·마찰·reset·실패 기준은 변경하지 않았습니다.

- [sejin5309/IsaacLab_Ant @ 0ecbfb9](https://github.com/sejin5309/IsaacLab_Ant/tree/0ecbfb95bac7750f56cda67a8bf334d24f1ada46)
- [zxro-z/IsaacLab_RS_Assignment1 @ b312c33](https://github.com/zxro-z/IsaacLab_RS_Assignment1/tree/b312c336161b986bd0db5ef506fb000d4007a397)
- [dding222/ant_rl @ dd35ebc](https://github.com/dding222/ant_rl/tree/dd35ebc11b01ea57e79cc6dfa03ac6ffcbfac557)

## 입력 호환을 위해 바꾼 부분

우리 정책은 **382차원 입력·8차원 effort action(scale 7.5)**을 요구합니다. Assignment1의 기본 60차원/다른 모델의 123·127차원, ant_rl의 59차원+depth CNN 입력은 그대로 넣을 수 없습니다.

- 세 환경에 같은 학습 관측 순서와 323-ray Height Scanner를 사용했습니다. 0 채우기·차원 잘라내기를 하지 않았습니다.
- Assignment1의 몸통 아래 종료 판정용 ray는 유지하고, 정책용 스캐너만 우리 설정으로 교체했습니다.
- ant_rl에서는 우리 모델이 사용하지 않는 Depth Camera를 제거하고 Height Scanner를 연결했습니다. 접촉 센서와 마찰 무작위화는 유지했습니다.
- 로봇·관절·effort scale·reset·지형·종료는 각 환경 설정을 유지했습니다. 계측용 종료 항목은 항상 False이며 보행에 개입하지 않습니다.
- 공통 런타임은 본인 설치의 Isaac Sim 5.1/IsaacLab입니다. 상대 저장소가 기록한 Isaac Sim 5.0 등과 물리 엔진까지 동일하게 재현한 시험은 아닙니다. 외부 task 모듈만 읽으며 상대 저장소 전체 IsaacLab을 설치하거나 덮어쓰지 않습니다.

따라서 이는 **관측 호환 어댑터를 사용한 환경 전이 평가**입니다. 상대 저장소의 원래 관측/모델로 얻은 논문식 baseline 재현 결과가 아닙니다.

## 해석과 한계

본인 지형에서 0/100 실패라는 결과만으로 외부 환경에서도 안정적이라고 말할 수 없습니다. Final-Unseen에서는 낮은 몸통 clearance 때문에 종료될 수 있고, 본인 환경은 뒤집힐 때까지 종료하지 않습니다. 이번 Final-Unseen의 실패 60개 중 10개는 첫 3초 이내입니다. ant_rl은 실패 24개 중 12개가 첫 1초, 14개가 첫 3초 이내에 발생했습니다. 시작 높이·중앙 평탄부·마찰 차이를 후속 통제 실험으로 분리할 필요가 있지만, 이 숫자만으로 특정 원인이 확정되지는 않습니다.

Final-Unseen에서 낮은 progress와 짧은 episode가 함께 관측됩니다. 누적 페널티가 작다고 제어가 더 효율적이라는 뜻도 아닙니다. 생존 시간이 짧으면 페널티를 받을 시간 자체가 줄어듭니다. 다른 팀 모델의 README 점수와 우리 점수를 직접 우열 비교하려면 센서 입력, 런타임, 정책 목표와 평가 절차까지 통제해야 합니다.

한 모델·한 seed·각 한 지도 실현의 결과입니다. 본인 map71은 반복 평가했고, ant_rl은 개발 중 참고한 지형 계열입니다. 세 환경 모두 완전히 독립적인 미공개 최종 시험이라는 표현은 쓰지 않습니다.

## 재실행

README의 런타임 설정 후 실행합니다. 외부 저장소는 별도로 clone하며, 이 저장소에는 그 전체 소스/모델을 중복해서 넣지 않습니다.

```bash
git clone https://github.com/zxro-z/IsaacLab_RS_Assignment1.git /tmp/ant-assignment-test
git -C /tmp/ant-assignment-test checkout b312c336161b986bd0db5ef506fb000d4007a397
git clone https://github.com/dding222/ant_rl.git /tmp/ant-rl-test
git -C /tmp/ant-rl-test checkout dd35ebc11b01ea57e79cc6dfa03ac6ffcbfac557

"$ISAACLAB_PATH/isaaclab.sh" -p "$ANT_REPO/scripts/play_one_episode.py" \
  --headless --seed 24 --num_envs 100 --output "$ANT_REPO/outputs/own_test"
"$ISAACLAB_PATH/isaaclab.sh" -p "$ANT_REPO/scripts/evaluate_transfer.py" \
  --source assignment --repository /tmp/ant-assignment-test \
  --headless --seed 24 --num_envs 100 --output "$ANT_REPO/outputs/assignment_test"
"$ISAACLAB_PATH/isaaclab.sh" -p "$ANT_REPO/scripts/evaluate_transfer.py" \
  --source ant_rl --repository /tmp/ant-rl-test \
  --headless --seed 24 --num_envs 100 --output "$ANT_REPO/outputs/ant_rl_test"
```

외부 실행기는 검사한 commit만 허용합니다. 각 실행에서 native/adapted config, 초기 관측·물리 상태, 100개 return 및 항목별 값을 저장합니다. 기본 자체 실행기는 summary를 저장합니다. 모든 실행은 새로운 output 폴더를 사용하세요.

[기계 판독 결과와 300개 원자료](../results/cross_repository_evaluation.json) · [원본 보상 300개 에피소드 CSV](../results/cross_repository_episodes.csv)

## 보충: ant_rl 자체 학습 보상

원본 7항목 점수와 다른 수치입니다. `play_one_episode.py`에서 사용하는 등록 환경의 학습 보상 10항목으로 같은 모델을 별도 실행했습니다. `play.py`의 6항목 재생 보상은 이 표의 기준이 아닙니다.

**자체 보상 mean ± std: 61.561737 ± 29.369378**

| 항목 | 평균 누적 기여 |
|---|---:|
| progress | +58.387828 |
| alive | +6.548640 |
| upright | +0.633991 |
| move_to_target | +19.497002 |
| foot_contact | +3.819838 |
| action_l2 | -0.081321 |
| energy | -11.006064 |
| joint_pos_limits | -13.937248 |
| joint_velocity | -1.076948 |
| foot_slip | -1.224162 |

보상 객체의 계산 전후 누적값 차이를 reset 전에 읽어 계측했습니다. float32 누적에서 오는 항목 합 오차의 최댓값은 0.000307입니다. 큰 progress weight(2.5), contact bonus 등 때문에 원본 보상 점수와 직접 비교할 수 없습니다. 초기 위치·관절·재질·관측 배열이 모두 일치했고 100개 종료 step·종료 사유·전진량도 공통 보상 실행과 전부 일치했습니다([검증 기록](../results/reward_replay_validation.json)). `--native-reward`로 재실행할 수 있습니다.
