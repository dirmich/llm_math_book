# LLM Math Book Exercises

実際に実行できる公開用の演習および実験コードです。

言語: [英語](../en/), [韓国語](../ko/)

## 内容

- `notebooks/`: 日本語の実行可能なノートブック
- `solutions/`: Ch01-Ch05 の日本語解答ノートブック
- `benchmarks/`: 日本語の CPU/GPU ベンチマークスクリプト
- `../src/llm_math/`: 共通ユーティリティ
- `../data/`: 小規模な練習用データ

## 公開範囲

現在公開している解答ノートブックは Ch01-Ch05 のみです。未完成の
Ch06-Ch32 解答 placeholder は、章に対応した完成版が用意できるまで
公開対象から削除しました。

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
