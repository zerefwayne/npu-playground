#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

extern "C" {

void matrix_mul_aie(int32_t *M, int32_t *N, int32_t *RES, int32_t m, int32_t k, int32_t n) {
  for(int i = 0; i < m; i++) {
    for(int j = 0; j < n; j++) {
      for(int rc = 0; rc < k; rc++) {
        RES[i * n + j] += M[i * k + rc] * N[rc * n + j];
      }
    }
  }
}

}
