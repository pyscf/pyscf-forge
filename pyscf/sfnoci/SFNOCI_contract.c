#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <stdbool.h>
#include <inttypes.h> //
#include <unistd.h>  //

bool is_matching_row(int *matrix_row, int *target_row, int cols) {
    for (int i = 0; i < cols; i++) {
        if (matrix_row[i] != target_row[i]) {
            return false;
        }
    }
    return true;
}

int find_matching_row(int *matrix, int *occ, int nrows, int ncols) {
    for (int i = 0; i < nrows; i++) {
        int match = 1;
        for (int j = 0; j < ncols; j++) {
            if (matrix[i*ncols+j] != occ[j]) {
                match = 0;
                break;
            }
        }
        if (match) return i;
    }
    return -1;  
}

int num_to_group(int *group, int group_sizes[], int num_groups, int number) {
    int index = 0;
    for (int i = 0; i < num_groups; i++) {
        for (int j = 0; j < group_sizes[i]; j++) {
            if (group[index] == number) {
                return i; 
            }
            index++;
        }
    }
    return -1; 
}

void str2occ(int *occ, uint64_t str, int norb) {
    for (int i = 0; i < norb; i++) {
        occ[i] = (str & (1ULL << i)) ? 1 : 0;
    }
}

void SFNOCIcontract_H_spin1(double *erieff, double *ci0, double *ci1, int ncas, int nelecasa, int nelecasb, 
int *PO, int PO_nrows, 
int na, uint64_t *stringsa, int nb, uint64_t *stringsb,
int *group, int group_sizes[], int num_groups, 
int *t1a, int *t1a_nonzero, int t1a_nonzero_size, 
int *t1b, int *t1b_nonzero, int t1b_nonzero_size,
int *t2aa, int *t2aa_nonzero, int t2aa_nonzero_size,
int *t2bb, int *t2bb_nonzero, int t2bb_nonzero_size,
double *TSc, double *energy_core){

int aa, ia, str1a, str0a;
int ab, ib, str1b, str0b;
int a1, i1, a2, i2;
int* w_occa = (int*) malloc(ncas * sizeof(int));
int* w_occb = (int*) malloc(ncas * sizeof(int));
int* x_occa = (int*) malloc(ncas * sizeof(int));
int* x_occb = (int*) malloc(ncas * sizeof(int));
int* w_occ = (int*) malloc(ncas * sizeof(int));
int* x_occ = (int*) malloc(ncas * sizeof(int));
int p1, p2, p; 
int num;

for (int i = 0; i <t1a_nonzero_size; i++){
    aa = t1a_nonzero[i*4+0];
    ia = t1a_nonzero[i*4+1];
    str1a = t1a_nonzero[i*4+2];
    str0a = t1a_nonzero[i*4+3];
    str2occ(w_occa, stringsa[str0a], ncas);
    str2occ(x_occa, stringsa[str1a], ncas);
    
    for (int j = 0; j < t1b_nonzero_size; j++){
        ab = t1b_nonzero[j*4+0];
        ib = t1b_nonzero[j*4+1];
        str1b = t1b_nonzero[j*4+2];
        str0b = t1b_nonzero[j*4+3];
        str2occ(w_occb, stringsb[str0b], ncas);
        str2occ(x_occb, stringsb[str1b], ncas);
        
        for (int k = 0; k < ncas; k++) {
                x_occ[k] = x_occa[k] + x_occb[k];
                w_occ[k] = w_occa[k] + w_occb[k];
            }
        p1 = find_matching_row(PO, x_occ, PO_nrows, ncas);
        p2 = find_matching_row(PO, w_occ, PO_nrows, ncas);
        
        num = PO_nrows;
        if (group!= NULL){
            p1 = num_to_group(group,group_sizes,num_groups,p1);
            p2 = num_to_group(group,group_sizes,num_groups,p2);
            num = num_groups;
        }

        ci1[str1a * nb + str1b] += ci0[str0a * nb + str0b]
        *erieff[p1 * num * ncas * ncas* ncas* ncas 
        + p2 * ncas * ncas * ncas * ncas 
        + aa * ncas * ncas * ncas 
        + ia * ncas *ncas 
        + ab * ncas + ib] 
        * t1a[aa * ncas * na * na + ia * na * na  + str1a * na  + str0a]
        * t1b[ab * ncas * nb * nb + ib * nb * nb + str1b * nb + str0b]
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
    str2occ(w_occa, stringsa[str0a], ncas);
    str2occ(x_occa, stringsa[str1a], ncas);
    for (int str0b = 0; str0b < nb; str0b++){
            str2occ(w_occb, stringsb[str0b], ncas);
            for (int k = 0; k < ncas; k++) {
                x_occ[k] = x_occa[k] + w_occb[k];
                w_occ[k] = w_occa[k] + w_occb[k];
            }
        p1 = find_matching_row(PO, x_occ, PO_nrows, ncas);
        p2 = find_matching_row(PO, w_occ, PO_nrows, ncas);
        num = PO_nrows;
        
        if (group!= NULL){
            p1 = num_to_group(group,group_sizes,num_groups,p1);
            p2 = num_to_group(group,group_sizes,num_groups,p2);
            num = num_groups;
        }    
        
        ci1[str1a * nb + str0b] += ci0[str0a * nb + str0b] 
        * erieff[p1 * num * ncas * ncas* ncas* ncas 
        + p2 * ncas * ncas * ncas * ncas 
        + a1 * ncas * ncas * ncas 
        + i1 * ncas  * ncas 
        + a2 * ncas 
        + i2]
        * t2aa[a1 * ncas * ncas * ncas * na * na + i1* ncas * ncas * na * na + a2 * ncas* na * na + i2* na * na + str1a * na + str0a]
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
    str2occ(w_occb, stringsb[str0b], ncas);
    str2occ(x_occb, stringsb[str1b], ncas);
    for (int str0a = 0; str0a < na; str0a++){
        str2occ(w_occa, stringsa[str0a], ncas);
        

            for (int k = 0; k < ncas; k++) {
                x_occ[k] = w_occa[k] + x_occb[k];
                w_occ[k] = w_occa[k] + w_occb[k];
            }
        p1 = find_matching_row(PO, x_occ, PO_nrows, ncas);
        p2 = find_matching_row(PO, w_occ, PO_nrows, ncas);
        num = PO_nrows;
      
        if (group!= NULL){
            p1 = num_to_group(group,group_sizes,num_groups,p1);
            p2 = num_to_group(group,group_sizes,num_groups,p2);
            num = num_groups;
        }    
        ci1[str0a * nb + str1b] += ci0[str0a * nb + str0b] 
        * erieff[p1 * num * ncas * ncas* ncas* ncas 
        + p2 * ncas * ncas * ncas * ncas 
        + a1 * ncas * ncas * ncas 
        + i1 * ncas  * ncas 
        + a2 * ncas 
        + i2]
        * t2bb[a1 * ncas * ncas * ncas * nb * nb + i1* ncas * ncas * nb * nb + a2 * ncas* nb * nb + i2* nb * nb + str1b * nb + str0b]
        * TSc[p1 * num + p2];
    }
}
for (int str0a = 0; str0a < na; str0a++) {
        for (int str0b = 0; str0b < nb; str0b++) {

            str2occ(w_occa, stringsa[str0a], ncas);
            str2occ(w_occb, stringsb[str0b], ncas);

            for (int k = 0; k < ncas; k++) {
                w_occ[k] = w_occa[k] + w_occb[k];
            }
            p = find_matching_row(PO, w_occ, PO_nrows, ncas);
            num = PO_nrows;
            if (group!= NULL){
            p = num_to_group(group,group_sizes,num_groups,p);
            num = num_groups;
            }
            ci1[str0a * nb + str0b] += energy_core[p] * ci0[str0a * nb + str0b];
}
}
free(w_occ);
free(w_occa);
free(w_occb);
free(x_occ);
free(x_occa);
free(x_occb);
}