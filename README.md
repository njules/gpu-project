# Python implementation
The python script can be run with `python python_implementation/main.py` after the environment has been set up. In its current configuration it loads the pretrained weights and runs the profiler on a single GPU accelerated forward pass on the test image.
# C implementation
Compile main.c and run to print the prediction and function execution times to stdout.
# CUDA implementation
Compile main.cu using nvcc and run.
# Pretrained weights
The pretrained weights can be found in the `weights` folder. For the C and CUDA implementations they can be found in the respective header files as statically defined matrices.
# Sample images
`test_images` contains the test images used for evaluation. They are stored as pictures before and after preprocessing and as .npy and .txt files after processing. Intermediate results after each layer and activation function for the `frog` image can be found under `test_images/frog_debug`. For the C and CUDA implementation the images are stored again in a header file, similar to the weights.