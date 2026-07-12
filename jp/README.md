# LLM Math Book Exercises

実際に実行できる公開用の演習および実験コードです。

言語: [英語](../en/), [韓国語](../ko/)

## 内容

- `notebooks/`: 日本語の実行可能なノートブック
- `solutions/`: Ch01-Ch32 の日本語解答ノートブック
- `benchmarks/`: 日本語の CPU/GPU ベンチマークスクリプト
- `../src/llm_math/`: 共通ユーティリティ
- `../data/`: 小規模な練習用データ

## 公開範囲

Ch00-Ch32 の演習ノートブックと Ch01-Ch32 の解答ノートブックを日本語、
英語、韓国語で公開しています。各言語 32 冊、解答 96 冊を含む合計 195 冊です。

## クイックスタート

```bash
pip install -e ".[notebook]"
jupyter lab
```

Colab の場合:

```bash
bash colab_setup.sh
```

## ライセンス

コードは MIT License で配布されます。
