# Source this file to set up the MESA environment:  source mesa_env.sh
export MESASDK_ROOT=/Applications/mesasdk
export MESA_DIR=/Users/greglaughlin/mesa-26.04.1
export OMP_NUM_THREADS=10          # performance cores on this machine
export SDKROOT=$(xcrun --sdk macosx --show-sdk-path)
source $MESASDK_ROOT/bin/mesasdk_init.sh
export PATH=$PATH:$MESA_DIR/scripts/shmesa
