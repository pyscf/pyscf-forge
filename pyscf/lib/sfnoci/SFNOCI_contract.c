#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <stdbool.h>
#include <inttypes.h> //
#include <unistd.h>  //


void SFNOCIcontract_H_spin1(double *erieff, double *ci0, double *ci1, int ncas, int nelecasa, int nelecasb, 
int *conf_info_list, int na, uint64_t *stringsa, int nb, uint64_t *stringsb, int num,
int *t1a, int *t1a_nonzero, int t1a_nonzero_size, 
int *t1b, int *t1b_nonzero, int t1b_nonzero_size,
int *t2aa, int *t2aa_nonzero, int t2aa_nonzero_size,
int *t2bb, int *t2bb_nonzero, int t2bb_nonzero_size,
double *TSc, double *energy_core){

int aa, ia, str1a, str0a;
int ab, ib, str1b, str0b;
int a1, i1, a2, i2;
int p1, p2, p; 

for (int i = 0; i <t1a_nonzero_size; i++){
    aa = t1a_nonzero[i*4+0];
    ia = t1a_nonzero[i*4+1];
    str1a = t1a_nonzero[i*4+2];
    str0a = t1a_nonzero[i*4+3];
    
    for (int j = 0; j < t1b_nonzero_size; j++){
        ab = t1b_nonzero[j*4+0];
        ib = t1b_nonzero[j*4+1];
        str1b = t1b_nonzero[j*4+2];
        str0b = t1b_nonzero[j*4+3];
        p1 = conf_info_list[str1a * nb + str1b]; 
        p2 = conf_info_list[str0a * nb + str0b];

        ci1[str1a * nb + str1b] += ci0[str0a * nb + str0b]
        *erieff[p1 * num * ncas * ncas* ncas* ncas 
        + p2 * ncas * ncas * ncas * ncas 
        + aa * ncas * ncas * ncas 
        + ia * ncas *ncas 
        + ab * ncas + ib] 
        * t1a[aa * ncas * (size_t)na * (size_t)na + ia * (size_t)na * (size_t)na  + str1a * na  + str0a]
        * t1b[ab * ncas * (size_t)nb * (size_t)nb + ib * (size_t)nb * (size_t)nb + str1b * nb + str0b]
        * TSc[p1 * num + p2] * 2.0;
    }
}

for (int i =0; i <t2aa_nonzero_size; i++){
    a1 = t2aa_nonzero[i*6+0];
    i1 = t2aa_nonzero[i*6+1];
    a2 = t2aa_nonzero[i*6+2];
    i2 = t2aa_nonzero[i*6+3];
    str1a = t2aa_nonzero[i*6+4];
    str0a = t2aa_nonzero[i*6+5];
    for (int str0b = 0; str0b < nb; str0b++){  
        p1 = conf_info_list[str1a * nb + str0b];
        p2 = conf_info_list[str0a * nb + str0b];
        ci1[str1a * nb + str0b] += ci0[str0a * nb + str0b] 
        * erieff[p1 * num * ncas * ncas* ncas* ncas 
        + p2 * ncas * ncas * ncas * ncas 
        + a1 * ncas * ncas * ncas 
        + i1 * ncas  * ncas 
        + a2 * ncas 
        + i2]
        * t2aa[a1 * ncas * ncas * ncas * (size_t)na * (size_t)na + i1* ncas * ncas * (size_t)na * (size_t)na + a2 * ncas* (size_t)na * (size_t)na + i2* (size_t)na * (size_t)na + str1a * na + str0a]
        * TSc[p1 * num + p2];
    }
}

for (int i =0; i <t2bb_nonzero_size; i++){
    a1 = t2bb_nonzero[i*6+0];
    i1 = t2bb_nonzero[i*6+1];
    a2 = t2bb_nonzero[i*6+2];
    i2 = t2bb_nonzero[i*6+3];
    str1b = t2bb_nonzero[i*6+4];
    str0b = t2bb_nonzero[i*6+5];
    for (int str0a = 0; str0a < na; str0a++){  
        p1 = conf_info_list[str0a * nb + str1b];
        p2 = conf_info_list[str0a * nb + str0b];

        ci1[str0a * nb + str1b] += ci0[str0a * nb + str0b] 
        * erieff[p1 * num * ncas * ncas* ncas* ncas 
        + p2 * ncas * ncas * ncas * ncas 
        + a1 * ncas * ncas * ncas 
        + i1 * ncas  * ncas 
        + a2 * ncas 
        + i2]
        * t2bb[a1 * ncas * ncas * ncas * (size_t)nb * (size_t)nb + i1* ncas * ncas * (size_t)nb * (size_t)nb + a2 * ncas* (size_t)nb * (size_t)nb + i2* (size_t)nb * (size_t)nb + str1b * nb + str0b]
        * TSc[p1 * num + p2];
    }
}
for (int str0a = 0; str0a < na; str0a++) {
        for (int str0b = 0; str0b < nb; str0b++) {
            p = conf_info_list[str0a * nb + str0b];
            ci1[str0a * nb + str0b] += energy_core[p] * ci0[str0a * nb + str0b];
}
}
}