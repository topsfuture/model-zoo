
from YOLOv8 import YOLOv8
from argparse import ArgumentParser



def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--onnx_path', type=str, default="../yolov8s.onnx", help="path to onnx model")
    parser.add_argument('--dataset_path', type=str, default="../val2017_1000", help="path to dataset")
    parser.add_argument('--quantize_type', type=str, default="float16", help="quantize data type")
    parser.add_argument('--quantize_batch_size', type=int, default=10, help="size of quantize cali data")    
    
    args = parser.parse_args()
    return args    

def main(args):
    yolov8 = YOLOv8(args.onnx_path)
    if args.quantize_type != "float32":
        yolov8.load_q_net(args.quantize_type)
    
    yolov8.nn.export_ovxlib(yolov8.acuity_net,
                            output_path = './export/',
                            pack_nbg_unify = True,
                            optimize = 'VIP9200O_PID0X10000049',
                            viv_sdk = '/workspace/Unified_Driver/cmdtools')



if __name__ == "__main__":
    args = parse_args()
    main(args)
