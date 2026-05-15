import evaluation
import convertjson2txt
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--json_source', default='result_widerval.json', help="原始JSON结果路径")
    parser.add_argument('-p', '--predicted_path', default="./widerface_txt/")
    parser.add_argument('-g', '--gt_path', default='./ground_truth/')
    args = parser.parse_args()
 


    # --- 新增：在评估前先调用转换脚本 ---
    print(f"\n步骤 1: 启动 JSON文件 {args.json_source}转换为 TXT 格式...")
    # 这里的 predict_path 对应 solve_conversion 的 output_root
    conversion_success = convertjson2txt.solve_conversion(json_path=args.json_source, output_root=args.predicted_path)

    # 传入 json_source 参数
    if (conversion_success == True):      
        evaluation.evaluation(args.predicted_path, args.gt_path)