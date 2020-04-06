
# Sudoku Solver with OpenCV

- 数独の画像0〜8に現れるマスを使って数字認識モデルを作成
- 数独の画像9〜はモデルにとって未知のデータ

```bash
python main.py 9
```

### Dev

- 画像データのExif削除

    ```bash
    mogrify -auto-orient data/img.orig/*
    exiftool -all= data/img.orig/*
    ```

- モデルの作成

    ```bash
    python digitrecog.py
    ```
