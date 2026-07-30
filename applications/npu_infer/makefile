CC = riscv64-unknown-linux-gnu-gcc
CFLAGS = -O3
INCLUDES = -I/tps-future/ta-base/ta-runtime/include \
           -I/tps-future/ta-vsp/ta-unify-9200O/include \
           -I/tps-future/ta-vsp/ta-unify-9200O/include/ovxlib-package-dev \
           -I/tps-future/ta-vsp/ta-unify-9200O/include/Vivante_ML_Toolkit_OVXLIB_dev \
           -I/tps-future/ta-vsp/ta-sys/include \
           -I/tps-future/ta-vsp/others
LIBS = -L/tps-future/ta-base/ta-runtime/lib \
       -L/tps-future/ta-vsp/ta-unify-9200O/lib \
       -L/tps-future/ta-vsp/ta-viplite-9200o/lib \
       -lta_runtime -lVIPhal -lNBGlinker -lpthread -lm

all: npu_infer

npu_infer: npu_infer.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $< $(LIBS)

clean:
	rm -f npu_infer