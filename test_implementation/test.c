#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "input_vector.h"
#include "input.h"


#define a 32
#define b 3

void test( long double *matrix_a, int i, int j, int k);

int main(){

    int i,j,k;
    i = 1;
    j = 3;
    k = 2;

    test(input_vector, i,j,k);

    printf("%Lf here \n",input[k][j][i]);


    return 0;
}

void test( long double *matrix_a, int i, int j, int k){
    printf("%Lf here \n",matrix_a[i + j * a + k * a * a]);
}