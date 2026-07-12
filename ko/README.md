# LLM Math Book Exercises

직접 실행해 볼 수 있는 공개 실습 및 실험 코드입니다.

언어: [영어](../en/), [일본어](../jp/)

## 구성

- `notebooks/`: 한국어 실행 노트북
- `solutions/`: Ch01-Ch32 한국어 해설 노트북
- `benchmarks/`: 한국어 CPU/GPU 성능 비교 스크립트
- `../src/llm_math/`: 공통 유틸리티
- `../data/`: 실습용 작은 데이터

## 공개 범위

Ch00-Ch32 실습 노트북과 Ch01-Ch32 해설 노트북을 한국어, 영어, 일본어로
공개합니다. 언어별 해설 32개, 전체 해설 96개를 포함해 총 195개입니다.

## 빠른 시작

```bash
pip install -e ".[notebook]"
jupyter lab
```

Colab에서는:

```bash
bash colab_setup.sh
```

## 라이선스

코드는 MIT License로 배포합니다.
