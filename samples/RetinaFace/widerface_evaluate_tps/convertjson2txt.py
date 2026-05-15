import json
import os

# 增加参数支持，方便外部调用时指定路径
def solve_conversion(json_path="result_widerval.json", output_root="widerface_txt"):
    # --- 配置信息 ---
    if not os.path.exists(json_path):
        print(f"错误：找不到 JSON 文件 {json_path}")
        return False # 返回状态方便外部判断
    else:
        print(f"解析 JSON 文件 {json_path}")

    # 1. 获取已创建的子目录列表
    print("正在读取子目录结构...")
    if not os.path.exists(output_root):
        print(f"错误：输出根目录 {output_root} 不存在")
        return False
        
    sub_folders = [f for f in os.listdir(output_root) if os.path.isdir(os.path.join(output_root, f))]

    # --- 清空子目录内的旧文件 ---
    print("正在清理旧的导出文件...")
    for folder in sub_folders:
        folder_path = os.path.join(output_root, folder)
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"无法删除文件 {file_path}: {e}")

    # 2. 加载 JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"开始处理 {len(data)} 条数据...")
    success_count = 0

    # 3. 解析并分发数据
    for item in data:
        full_image_name = item.get('image_name', '')
        file_only = os.path.basename(full_image_name)
        file_stem = os.path.splitext(file_only)[0]

        target_folder = None
        prefix = file_only.split('_')[0] 
        
        for folder in sub_folders:
            if folder.startswith(f"{prefix}--"):
                target_folder = folder
                break
        
        if not target_folder:
            for folder in sub_folders:
                if folder.split('--')[-1] in file_only:
                    target_folder = folder
                    break

        if target_folder:
            txt_path = os.path.join(output_root, target_folder, f"{file_stem}.txt")
            
            
            with open(txt_path, 'w', encoding='utf-8') as f_out:
                f_out.write(f"{file_stem}\n") 
                bboxes = item.get('bboxes', [])
                f_out.write(f"{len(bboxes)}\n") 
                
                for det in bboxes:
                    b = det['bbox']
                    x, y = b[0], b[1]
                    w, h = b[2] - b[0], b[3] - b[1]
                    score = det['score']
                    #f_out.write(f"{x:.3f} {y:.3f} {w:.3f} {h:.3f} {score:.6f}\n") #float for coordination
                    f_out.write(f"{round(x)} {round(y)} {round(w)} {round(h)} {score:.6f}\n") #int for coordination
            
            '''
            with open(txt_path, 'w', encoding='utf-8') as f_out:
                f_out.write(f"{file_stem}\n") 
                
                # --- 修改核心逻辑：先过滤 ---
                bboxes = item.get('bboxes', [])
                valid_bboxes = []
                for det in bboxes:
                    b = det['bbox']
                    w, h = b[2] - b[0], b[3] - b[1]
                    # 只有符合条件的才放入有效列表
                    if w < 600 and h < 600:
                        valid_bboxes.append(det)
                
                # --- 写入真实的过滤后的数量 ---
                f_out.write(f"{len(valid_bboxes)}\n") 
                
                # --- 遍历有效列表进行写入 ---
                for det in valid_bboxes:
                    b = det['bbox']
                    x, y = b[0], b[1]
                    w, h = b[2] - b[0], b[3] - b[1]
                    score = det['score']
                    # 使用 round 处理并写入
                    f_out.write(f"{round(x)} {round(y)} {round(w)} {round(h)} {score:.6f}\n")
            '''
            success_count += 1



            
    print(f"转换完成！成功写入 {success_count} 个文件。")
    return True

if __name__ == "__main__":
    solve_conversion()