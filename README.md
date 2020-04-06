
# Sudoku Solver with OpenCV

- 数独の画像0〜8に現れるマスを使って数字認識モデルを作成
- 数独の画像9〜はモデルにとって未知のデータ

```bash
python main.py 9
```

### Dev

- 流れ

    1. 直線（数独の枠）を検出する
    2. 直線の交点を求める
    3. 左上、左下、右上、右下の交点を求める
    4. 射影変換する
    5. 9x9のマスで画像切り取り
    6. 数字の認識
    7. SAT・SMTソルバで数独を解く


- 画像データのExif削除

    ```bash
    mogrify -auto-orient data/img.orig/*
    exiftool -all= data/img.orig/*
    ```

- モデルの作成

    ```bash
    python digitrecog.py
    ```
