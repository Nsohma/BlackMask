import json
import os
import re
from pathlib import Path

def parse_ground_truth(file_path):
    """correct_qwenvl.txt を解析して辞書形式に変換する"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = re.split(r'\n(?=C\d{3}_)', content)
    gt_data = {}

    for block in blocks:
        lines = block.strip().split('\n')
        if not lines: continue
        
        img_id = lines[0].replace(':', '').strip()
        data = {'subjects': [], 'props': set(), 'id_count': 0}
        current_sub = None

        for line in lines[1:]:
            line = line.strip()
            if not line: continue
            
            if '(' in line or '{メインキャラ}' in line:
                if current_sub: data['subjects'].append(current_sub)
                current_sub = {'id_label': line}
                continue
            
            if ':' in line:
                parts = line.split(':', 1)
                key = parts[0].strip()
                val = parts[1].strip().strip('{}').replace('、', ',') if len(parts) > 1 else ""
                
                if val.lower() in ['none', '']: val = None

                if key == 'id_count':
                    data['id_count'] = int(val) if val else 0
                elif key == 'props':
                    data['props'] = {v.strip() for v in val.split(',')} if val else set()
                elif current_sub is not None:
                    if key in ['pose', 'expression']:
                        current_sub[key] = {v.strip() for v in val.split(',')} if val else set()
                    else:
                        current_sub[key] = val
        
        if current_sub: data['subjects'].append(current_sub)
        gt_data[img_id] = data
    return gt_data

def calculate_set_metrics(gt_set, pred_set):
    """集合間の Precision, Recall, F1 を計算する"""
    if not gt_set and not pred_set: return 1.0, 1.0, 1.0
    if not gt_set or not pred_set: return 0.0, 0.0, 0.0
    
    tp = len(gt_set.intersection(pred_set))
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def evaluate(gt_file, json_dir):
    gt_dict = parse_ground_truth(gt_file)
    results = {
        'cropping': [], 'camera_angle': [], 
        'pose_f1': [], 'expr_f1': [], 'props_f1': [], 'id_acc': []
    }

    print(f"{'ID':<15} | ID_Cnt | Crop | Angle | PoseF1 | ExprF1 | PropF1")
    print("-" * 80)

    for img_id, gt in gt_dict.items():
        json_file = Path(json_dir) / f"{img_id}_all.png.json"
        if not json_file.exists():
            continue

        with open(json_file, 'r', encoding='utf-8') as f:
            pred = json.load(f)

        # 初期値 (その画像で比較対象がない場合のデフォルト値)
        cur_crop = 0.0
        cur_angle = 0.0
        cur_pose_f1 = 0.0
        cur_expr_f1 = 0.0
        
        # 1. 人数カウント (id_count)
        pred_count = len(pred.get('subjects', []))
        id_acc = 1.0 if pred_count == gt.get('id_count', 0) else 0.0
        results['id_acc'].append(id_acc)

        # 2. メインキャラ (id:1) の比較
        if gt['subjects'] and pred.get('subjects'):
            gt_main = gt['subjects'][0]
            pred_main = pred['subjects'][0]

            # Cropping & Angle (Exact Match)
            cur_crop = 1.0 if gt_main.get('cropping') == pred_main.get('cropping', {}).get('value') else 0.0
            cur_angle = 1.0 if gt_main.get('camera_angle') == pred_main.get('camera_angle', {}).get('value') else 0.0
            
            results['cropping'].append(cur_crop)
            results['camera_angle'].append(cur_angle)

            # Pose & Expression (Set F1)
            pred_poses = {p['description'] for p in pred_main.get('pose', [])}
            _, _, cur_pose_f1 = calculate_set_metrics(gt_main.get('pose', set()), pred_poses)
            results['pose_f1'].append(cur_pose_f1)

            pred_exprs = {e['description'] for e in pred_main.get('expression', [])}
            _, _, cur_expr_f1 = calculate_set_metrics(gt_main.get('expression', set()), pred_exprs)
            results['expr_f1'].append(cur_expr_f1)
        
        # 3. 小物 (Props Set F1)
        pred_props = {p['name'] for p in pred.get('props', [])}
        _, _, cur_prop_f1 = calculate_set_metrics(gt.get('props', set()), pred_props)
        results['props_f1'].append(cur_prop_f1)

        # 表示
        print(f"{img_id:<15} | {pred_count:>6} | {cur_crop:.2f} | {cur_angle:.2f} | {cur_pose_f1:.2f} | {cur_expr_f1:.2f} | {cur_prop_f1:.2f}")

    # 最終集計
    print("\n" + "="*30)
    print("FINAL SUMMARY (AVERAGE)")
    print("="*30)
    for k, v in results.items():
        avg = sum(v) / len(v) if v else 0
        print(f"{k:<15}: {avg:.4f}")

if __name__ == "__main__":
    # ここに正しいパスを入力してください
    GT_FILE = "correct_qwenvl.txt"
    JSON_DIR = "/Users/niihosoushin/Downloads/Bthesis_Dataset/07_output_prompt_try_0128_1741"
    evaluate(GT_FILE, JSON_DIR)