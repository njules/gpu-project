#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "img.h"

#define imgsize 32




int main(){
    
    //convolution();

    return 0;
}

void convolution(int a_width, int b_width, int channel_in, int channel_out,
                int matrix_a[a_width][a_width][channel_in], 
                int matrix_b[b_width][b_width][channel_in][channel_out], 
                int matrix_c[a_width - (b_width / 2)][a_width - (b_width / 2)][channel_out]){

    int kCenter = b_width / 2;
    
    for (int c_out = 0; c_out < channel_out; ++c_out)
    {
        for (int i = 0; i < a_width; ++i)              // rows
        {
            for (int j = 0; j < a_width; ++j)          // columns
            {
                for (int m = 0; m < b_width; ++m)     // kernel rows
                {
                    int mm = b_width - 1 - m;      // row index

                    for (int n = 0; n < b_width; ++n) // kernel columns
                    {
                        int nn = b_width - 1 - n;  // column index

                        // index of input signal, used for checking boundary
                        int ii = i + (m - kCenter);
                        int jj = j + (n - kCenter);

                        // ignore input samples which are out of bound
                        if (ii >= 0 && ii < a_width && jj >= 0 && jj < a_width){
                            for (int c_in = 0; c_in < channel_in; ++c_in){
                                matrix_c[i][j][c_out] += matrix_a[ii][jj][c_in] * matrix_b[m][n][c_in][c_out];
                            }
                        }
                    }
                }
            }
        }
    }

    return ;
}

void pool(int a_width, int amount,int channel_in, int channel_out,
                int matrix_a[a_width][a_width][channel_in],
                int matrix_b[a_width/amount][a_width/amount][channel_out]){


    return;
}

void dense(int matrix_a, int matrix_b, int width, int height){


    return;
}

void sigmoid(int a_width, int amount,int channel_in, int channel_out,
                int matrix_a[a_width][a_width][channel_in],
                int matrix_b[a_width/amount][a_width/amount][channel_out]){


    return;
}