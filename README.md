# BlackMask
# pythonのコード
bg_diff_line.py	: 修正のマスクをすpythonソースコード
使い方: python3 bg_diff_line.py (ただし、source .venv/bin/activate が必要)
出力先 -> 05_raf_image_mask (マスク画像), 05_raf_image_json (マスクデータ)

overlap.py : 修正のマスクと物体検出済みの箱を重ね、ラベルを抽出する。
judge_score = (ink_ratio * score) が閾値judge_thresh以上なら、そのラベルがtrueになる
使い方: python3 overlap.py > (リダイレクトしたいファイル)
出力 -> 06_overlap_out(詳細情報), 標準出力(trueのラベルのみ上位から)

eval_tags.py : 手動のラベルと検出ラベルの正誤率判定のソースコード
使い方: python3 eval_tags.py <手動のタグファイル> <検出ラベルのタグファイル>
出力 -> 標準出力(各評価指標とそのデータ)

#フォルダ
03_sub_cell : 元の彩色済みセル画像
05_raf_image : 元のラフ画像
05_raf_image_mask : 修正のマスク画像
05_raf_image_json : 修正のマスクデータ
06_over_lap : 検出ラベルの詳細情報

#ファイル
correct_label.txt : 手動で作った正解のタグデータ

#物体検出
v1 prompt : face . eyes . mouth . arm . head . leg . nose . person
v2 prompt : face . eyes . mouth . arm . head . leg . nose . person . ears . clothes . eyebrows .

