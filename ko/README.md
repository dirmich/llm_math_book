# LLM Math Book Exercises

직접 실행해 볼 수 있는 공개 실습 및 실험 코드입니다.

언어: [영어](../en/), [일본어](../jp/)

## 구성

- `notebooks/`: 한국어 실행 노트북
- `solutions/`: Ch01-Ch05 한국어 해설 노트북
- `benchmarks/`: 한국어 CPU/GPU 성능 비교 스크립트
- `../src/llm_math/`: 공통 유틸리티
- `../data/`: 실습용 작은 데이터

## 공개 범위

현재 공개된 해설 노트북은 Ch01-Ch05뿐입니다. 완성되지 않은 Ch06-Ch32
해설 placeholder는 공개 목록에서 제거했습니다.

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
