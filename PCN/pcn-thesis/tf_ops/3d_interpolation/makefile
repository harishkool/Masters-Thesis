cuda_inc = /usr/local/cuda-9.0/include/
cuda_lib = /usr/local/cuda-9.0/lib64/
nvcc = /usr/local/cuda-9.0/bin/nvcc
tf_inc = `python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())'`
tf_lib = `python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())'`

all: tf_interpolate_so.so

tf_interpolate_so.so: tf_interpolate.cpp 
	g++ tf_interpolate.cpp -o tf_interpolate_so.so \
	-I $(cuda_inc) -I $(tf_inc) -L $(cuda_lib) -lcudart -L $(tf_lib) -ltensorflow_framework \
    -shared -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -fPIC -O2

clean:
	rm -rf *.o *.so
