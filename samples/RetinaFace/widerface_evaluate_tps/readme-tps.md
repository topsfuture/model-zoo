# widerface_evaluate_tps

## step1
解压widerface_txt.tar.gz到本目录

## step2
把板端推理整个WIDERVAL目录生成的.json(如WIDERVAL.json)到本目录下 

## step3
执行 python3 ./evaluation_tps.py -j ./WIDERVAL.json, 即可得到此次推理的评估结果

## step4
如果执行过程中遇到 tqdm、scipy、bbox、IPython模块找不到，请分别使用pip install安装