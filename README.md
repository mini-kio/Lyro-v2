# LYRO v2

오디오·가사 정렬과 텍스트 조건 음악 생성을 실험하는 연구 코드입니다. 텍스트/오디오 인코더, Flow Matching 생성기, 학습기, DCAE·보코더 구성 요소를 포함합니다.

## 구성

| 경로 | 역할 |
| --- | --- |
| [models/](models/) | 인코더·생성기·정렬·샘플링·손실 |
| [training/](training/) | 학습 설정과 트레이너 |
| [data/](data/) | 데이터셋과 전처리 |
| [inference/](inference/) | 추론 파이프라인 |
| [dcae_vocoder/](dcae_vocoder/) | 오디오 표현과 복원 모듈 |
| [examples/](examples/) | 실험용 예제 |

설정은 [training/config.py](training/config.py)에서 확인할 수 있습니다. 가사·오디오 메타데이터 경로와 생성기 설정을 실행 환경에 맞게 조정해야 합니다.

## 현재 코드 상태

일부 예제는 저장소에 없는 `models.multimodal_lyro`를 참조합니다. 예제를 바로 실행하기 전에 현재 모듈과 import 경로를 맞춰야 합니다. DCAE·보코더 폴더에는 설정 파일이 있으며, 학습된 가중치는 별도로 준비해야 합니다.

관련 초기 실험: [lyro](https://github.com/mini-kio/lyro).
