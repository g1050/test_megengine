g++ lite_infer_cuda.cpp -o bin/lite_infer -I$LITE_INSTALL_DIR/include -llite_shared -L$LITE_INSTALL_DIR/lib/x86_64 \
    -I/usr/local/include/opencv4/ \
    -lopencv_highgui  -lopencv_imgcodecs -lopencv_imgproc -lopencv_core \
    -L/usr/local/lib/ -g