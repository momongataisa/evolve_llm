# Contributing to Evolve LLM

Evolve LLMへの貢献を歓迎します！

## 開発環境のセットアップ

```bash
# リポジトリのクローン
git clone <repository-url>
cd evolve_llm

# 仮想環境の作成
python -m venv venv
source venv/bin/activate  # Windowsの場合: venv\Scripts\activate

# 開発用依存関係のインストール
pip install -r requirements.txt
pip install -e .
```

## コーディング規約

- PEP 8に準拠
- Black でコードフォーマット
- Flake8 でリント
- 型ヒントを使用
- Docstringを記述（Google style）

```bash
# フォーマット
black .

# リント
flake8 .
```

## テスト

```bash
# テストの実行
pytest

# カバレッジ付き
pytest --cov=. --cov-report=html
```

## プルリクエストのガイドライン

1. 新しい機能や修正用のブランチを作成
2. コードを記述し、テストを追加
3. フォーマットとリントを実行
4. コミットメッセージは明確に
5. プルリクエストを作成

## 報告とフィードバック

バグ報告や機能要望はGitHub Issuesでお願いします。

## ライセンス

貢献したコードはMITライセンスの下で公開されます。
