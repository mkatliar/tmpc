/* This function was automatically generated by CasADi */
#ifdef __cplusplus
extern "C" {
#endif

#ifdef CODEGEN_PREFIX
  #define NAMESPACE_CONCAT(NS, ID) _NAMESPACE_CONCAT(NS, ID)
  #define _NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else /* CODEGEN_PREFIX */
  #define CASADI_PREFIX(ID) pendulum_ode_generated_ ## ID
#endif /* CODEGEN_PREFIX */

#include <math.h>

#ifndef real_t
#define real_t double
#endif /* real_t */

#define to_double(x) (double) x
#define to_int(x) (int) x
/* Pre-c99 compatibility */
#if __STDC_VERSION__ < 199901L
real_t CASADI_PREFIX(fmin)(real_t x, real_t y) { return x<y ? x : y;}
#define fmin(x,y) CASADI_PREFIX(fmin)(x,y)
real_t CASADI_PREFIX(fmax)(real_t x, real_t y) { return x>y ? x : y;}
#define fmax(x,y) CASADI_PREFIX(fmax)(x,y)
#endif

#define PRINTF printf
real_t CASADI_PREFIX(sq)(real_t x) { return x*x;}
#define sq(x) CASADI_PREFIX(sq)(x)

real_t CASADI_PREFIX(sign)(real_t x) { return x<0 ? -1 : x>0 ? 1 : x;}
#define sign(x) CASADI_PREFIX(sign)(x)

void CASADI_PREFIX(copy)(const real_t* x, int n, real_t* y) {
  int i;
  if (y) {
    if (x) {
      for (i=0; i<n; ++i) *y++ = *x++;
    } else {
      for (i=0; i<n; ++i) *y++ = 0.;
    }
  }
}
#define copy(x, n, y) CASADI_PREFIX(copy)(x, n, y)


void CASADI_PREFIX(fill)(real_t* x, int n, real_t alpha) {
  int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = alpha;
  }
}
#define fill(x, n, alpha) CASADI_PREFIX(fill)(x, n, alpha)


void CASADI_PREFIX(trans)(const real_t* x, const int* sp_x, real_t* y, const int* sp_y, int *tmp) {
  int ncol_x = sp_x[1];
  int nnz_x = sp_x[2 + ncol_x];
  const int* row_x = sp_x + 2 + ncol_x+1;
  int ncol_y = sp_y[1];
  const int* colind_y = sp_y+2;
  int k;
  for (k=0; k<ncol_y; ++k) tmp[k] = colind_y[k];
  for (k=0; k<nnz_x; ++k) {
    y[tmp[row_x[k]]++] = x[k];
  }
}
#define trans(x, sp_x, y, sp_y, tmp) CASADI_PREFIX(trans)(x, sp_x, y, sp_y, tmp)

void CASADI_PREFIX(project)(const real_t* x, const int* sp_x, real_t* y, const int* sp_y, real_t* w) {
  int ncol_x = sp_x[1];
  const int *colind_x = sp_x+2, *row_x = sp_x + 2 + ncol_x+1;
  int ncol_y = sp_y[1];
  const int *colind_y = sp_y+2, *row_y = sp_y + 2 + ncol_y+1;
  /* Loop over columns of x and y */
  int i, el;
  for (i=0; i<ncol_x; ++i) {
    /* Zero out requested entries in y */
    for (el=colind_y[i]; el<colind_y[i+1]; ++el) w[row_y[el]] = 0;
    /* Set x entries */
    for (el=colind_x[i]; el<colind_x[i+1]; ++el) w[row_x[el]] = x[el];
    /* Retrieve requested entries in y */
    for (el=colind_y[i]; el<colind_y[i+1]; ++el) y[el] = w[row_y[el]];
  }
}
#define project(x, sp_x, y, sp_y, w) CASADI_PREFIX(project)(x, sp_x, y, sp_y, w)


static const int CASADI_PREFIX(s0)[8] = {2, 2, 0, 1, 3, 1, 0, 1};
#define s0 CASADI_PREFIX(s0)
static const int CASADI_PREFIX(s1)[9] = {2, 2, 0, 2, 4, 0, 1, 0, 1};
#define s1 CASADI_PREFIX(s1)
static const int CASADI_PREFIX(s2)[6] = {2, 1, 0, 2, 0, 1};
#define s2 CASADI_PREFIX(s2)
static const int CASADI_PREFIX(s3)[5] = {2, 1, 0, 1, 1};
#define s3 CASADI_PREFIX(s3)
static const int CASADI_PREFIX(s4)[7] = {2, 2, 0, 1, 2, 1, 0};
#define s4 CASADI_PREFIX(s4)
static const int CASADI_PREFIX(s5)[6] = {2, 2, 0, 1, 1, 0};
#define s5 CASADI_PREFIX(s5)
static const int CASADI_PREFIX(s6)[5] = {1, 1, 0, 1, 0};
#define s6 CASADI_PREFIX(s6)
/* pendulum_ode */
int pendulum_ode(const real_t** arg, real_t** res, int* iw, real_t* w, int mem) {
  int i, j, k, *ii, *jj, *kk;
  const int *cii;
  real_t r, s, t, *rr, *ss, *tt;
  const real_t *cr, *cs, *ct;
  const real_t** arg1=arg+3;
  real_t** res1=res+9;
  real_t w0, w1, w2, w3, w4, *w5=w+7, w6, w7, *w8=w+11, *w9=w+14, *w10=w+17, *w11=w+21;
  /* #0: Input 1 (x0), part 0 (phi) */
  w0 = arg[1] ? arg[1][0] : 0;
  /* #1: @1 = sin(@0) */
  w1 = sin( w0 );
  /* #2: @2 = -9.81 */
  w2 = -9.8100000000000005e+00;
  /* #3: @1 = (@2*@1) */
  w1  = (w2*w1);
  /* #4: Input 1 (x0), part 1 (dphi) */
  w3 = arg[1] ? arg[1][1] : 0;
  /* #5: @4 = (2.*@3) */
  w4 = (2.* w3 );
  /* #6: @1 = (@1-@4) */
  w1 -= w4;
  /* #7: Input 2 (u0), part 0 (u) */
  w4 = arg[2] ? arg[2][0] : 0;
  /* #8: @1 = (@1+@4) */
  w1 += w4;
  /* #9: @5 = vertcat(@3, @1) */
  rr=w5;
  *rr++ = w3;
  *rr++ = w1;
  /* #10: Output 0 (xdot) */
  if (res[0]) copy(w5, 2, res[0]);
  /* #11: @1 = cos(@0) */
  w1 = cos( w0 );
  /* #12: @6 = ones(2x1, 1 nnz) */
  w6 = 1.;
  /* #13: {@7, NULL} = vertsplit(@6) */
  w7 = w6;
  /* #14: @1 = (@1*@7) */
  w1 *= w7;
  /* #15: @2 = (@2*@1) */
  w2 *= w1;
  /* #16: @8 = zeros(2x2, 3 nnz) */
  fill(w8, 3, 0.);
  /* #17: (@8[1] = @2) */
  for (rr=w8+1, ss=(&w2); rr!=w8+2; rr+=1) *rr = *ss++;
  /* #18: @2 = ones(2x1, 1 nnz) */
  w2 = 1.;
  /* #19: {NULL, @1} = vertsplit(@2) */
  w1 = w2;
  /* #20: @2 = (2.*@1) */
  w2 = (2.* w1 );
  /* #21: @2 = (-@2) */
  w2 = (- w2 );
  /* #22: @5 = vertcat(@1, @2) */
  rr=w5;
  *rr++ = w1;
  *rr++ = w2;
  /* #23: (@8[:4:2] = @5) */
  for (rr=w8+0, ss=w5; rr!=w8+4; rr+=2) *rr = *ss++;
  /* #24: @9 = @8' */
  trans(w8, s0, w9, s0, iw);
  /* #25: @10 = dense(@9) */
    project(w9, s0, w10, s1, w);
  /* #26: Output 1 (A) */
  if (res[1]) copy(w10, 4, res[1]);
  /* #27: @1 = zeros(1x2, 1 nnz) */
  w1 = 0.;
  /* #28: @2 = 1 */
  w2 = 1.;
  /* #29: (@1[0] = @2) */
  for (rr=(&w1)+0, ss=(&w2); rr!=(&w1)+1; rr+=1) *rr = *ss++;
  /* #30: @1 = @1' */
  /* #31: @5 = dense(@1) */
    project((&w1), s3, w5, s2, w);
  /* #32: Output 2 (B) */
  if (res[2]) copy(w5, 2, res[2]);
  /* #33: @1 = sq(@3) */
  w1 = sq( w3 );
  /* #34: @5 = vertcat(@1, @0) */
  rr=w5;
  *rr++ = w1;
  *rr++ = w0;
  /* #35: Output 3 (q) */
  if (res[3]) copy(w5, 2, res[3]);
  /* #36: @3 = (2.*@3) */
  w3 = (2.* w3 );
  /* #37: @5 = ones(2x1, dense) */
  fill(w5, 2, 1.);
  /* #38: {@1, @2} = vertsplit(@5) */
  w1 = w5[0];
  w2 = w5[1];
  /* #39: @3 = (@3*@2) */
  w3 *= w2;
  /* #40: @5 = vertcat(@3, @1) */
  rr=w5;
  *rr++ = w3;
  *rr++ = w1;
  /* #41: @11 = zeros(2x2, 2 nnz) */
  fill(w11, 2, 0.);
  /* #42: (@11[:2] = @5) */
  for (rr=w11+0, ss=w5; rr!=w11+2; rr+=1) *rr = *ss++;
  /* #43: @5 = @11' */
  trans(w11, s4, w5, s4, iw);
  /* #44: @10 = dense(@5) */
    project(w5, s4, w10, s1, w);
  /* #45: Output 4 (qA) */
  if (res[4]) copy(w10, 4, res[4]);
  /* #46: @5 = zeros(2x1, dense) */
  fill(w5, 2, 0.);
  /* #47: Output 5 (qB) */
  if (res[5]) copy(w5, 2, res[5]);
  /* #48: @3 = cos(@0) */
  w3 = cos( w0 );
  /* #49: @5 = vertcat(@3, @4) */
  rr=w5;
  *rr++ = w3;
  *rr++ = w4;
  /* #50: Output 6 (r) */
  if (res[6]) copy(w5, 2, res[6]);
  /* #51: @0 = sin(@0) */
  w0 = sin( w0 );
  /* #52: @5 = ones(2x1, dense) */
  fill(w5, 2, 1.);
  /* #53: {@3, NULL} = vertsplit(@5) */
  w3 = w5[0];
  /* #54: @0 = (@0*@3) */
  w0 *= w3;
  /* #55: @0 = (-@0) */
  w0 = (- w0 );
  /* #56: @3 = zeros(2x2, 1 nnz) */
  w3 = 0.;
  /* #57: (@3[0] = @0) */
  for (rr=(&w3)+0, ss=(&w0); rr!=(&w3)+1; rr+=1) *rr = *ss++;
  /* #58: @0 = @3' */
  trans((&w3), s5, (&w0), s5, iw);
  /* #59: @10 = dense(@0) */
    project((&w0), s5, w10, s1, w);
  /* #60: Output 7 (rA) */
  if (res[7]) copy(w10, 4, res[7]);
  /* #61: @0 = zeros(1x2, 1 nnz) */
  w0 = 0.;
  /* #62: @3 = 1 */
  w3 = 1.;
  /* #63: (@0[0] = @3) */
  for (rr=(&w0)+0, ss=(&w3); rr!=(&w0)+1; rr+=1) *rr = *ss++;
  /* #64: @0 = @0' */
  /* #65: @5 = dense(@0) */
    project((&w0), s3, w5, s2, w);
  /* #66: Output 8 (rB) */
  if (res[8]) copy(w5, 2, res[8]);
  return 0;
}

void pendulum_ode_incref(void) {
}

void pendulum_ode_decref(void) {
}

int pendulum_ode_n_in(void) { return 3;}

int pendulum_ode_n_out(void) { return 9;}

const char* pendulum_ode_name_in(int i){
  switch (i) {
  case 0: return "t";
  case 1: return "x0";
  case 2: return "u0";
  default: return 0;
  }
}

const char* pendulum_ode_name_out(int i){
  switch (i) {
  case 0: return "xdot";
  case 1: return "A";
  case 2: return "B";
  case 3: return "q";
  case 4: return "qA";
  case 5: return "qB";
  case 6: return "r";
  case 7: return "rA";
  case 8: return "rB";
  default: return 0;
  }
}

const int* pendulum_ode_sparsity_in(int i) {
  switch (i) {
  case 0: return s6;
  case 1: return s2;
  case 2: return s6;
  default: return 0;
  }
}

const int* pendulum_ode_sparsity_out(int i) {
  switch (i) {
  case 0: return s2;
  case 1: return s1;
  case 2: return s2;
  case 3: return s2;
  case 4: return s1;
  case 5: return s2;
  case 6: return s2;
  case 7: return s1;
  case 8: return s2;
  default: return 0;
  }
}

int pendulum_ode_work(int *sz_arg, int* sz_res, int *sz_iw, int *sz_w) {
  if (sz_arg) *sz_arg = 5;
  if (sz_res) *sz_res = 11;
  if (sz_iw) *sz_iw = 3;
  if (sz_w) *sz_w = 23;
  return 0;
}

/* fwd1_tmp */
static int CASADI_PREFIX(f0)(const real_t** arg, real_t** res, int* iw, real_t* w, int mem) {
  int i, j, k, *ii, *jj, *kk;
  const int *cii;
  real_t r, s, t, *rr, *ss, *tt;
  const real_t *cr, *cs, *ct;
  const real_t** arg1=arg+3;
  real_t** res1=res+1;
  real_t w0, *w1=w+1, w2, w3;
  /* #0: Input 0 (der_i0), part 0 (phi) */
  w0 = arg[0] ? arg[0][0] : 0;
  /* #1: @0 = cos(@0) */
  w0 = cos( w0 );
  /* #2: Input 2 (fwd_i0), part 0 (f_0) */
  if (arg[2])
    copy(arg[2], 2, w1);
  else 
    fill(w1, 2, 0);
  /* #3: {@2, @3} = vertsplit(@1) */
  w2 = w1[0];
  w3 = w1[1];
  /* #4: @0 = (@0*@2) */
  w0 *= w2;
  /* #5: @2 = -9.81 */
  w2 = -9.8100000000000005e+00;
  /* #6: @2 = (@2*@0) */
  w2 *= w0;
  /* #7: @0 = (2.*@3) */
  w0 = (2.* w3 );
  /* #8: @2 = (@2-@0) */
  w2 -= w0;
  /* #9: @1 = vertcat(@3, @2) */
  rr=w1;
  *rr++ = w3;
  *rr++ = w2;
  /* #10: Output 0 (fwd_o0) */
  if (res[0]) copy(w1, 2, res[0]);
  return 0;
}

#define f0(arg, res, iw, w, mem) CASADI_PREFIX(f0)(arg, res, iw, w, mem)

/* fwd1_tmp */
static int CASADI_PREFIX(f1)(const real_t** arg, real_t** res, int* iw, real_t* w, int mem) {
  int i, j, k, *ii, *jj, *kk;
  const int *cii;
  real_t r, s, t, *rr, *ss, *tt;
  const real_t *cr, *cs, *ct;
  const real_t** arg1=arg+3;
  real_t** res1=res+1;
  real_t w1, w2;
  /* #0: @0 = 00 */
  /* #1: Input 2 (fwd_i0), part 0 (f_0) */
  w1 = arg[2] ? arg[2][0] : 0;
  /* #2: @2 = vertcat(@0, @1) */
  rr=(&w2);
  *rr++ = w1;
  /* #3: Output 0 (fwd_o0) */
  if (res[0]) *res[0] = w2;
  return 0;
}

#define f1(arg, res, iw, w, mem) CASADI_PREFIX(f1)(arg, res, iw, w, mem)

/* pendulum_ode_sens */
int pendulum_ode_sens(const real_t** arg, real_t** res, int* iw, real_t* w, int mem) {
  int i, j, k, *ii, *jj, *kk;
  const int *cii;
  real_t r, s, t, *rr, *ss, *tt;
  const real_t *cr, *cs, *ct;
  const real_t** arg1=arg+5;
  real_t** res1=res+2;
  real_t w0, w1, w2, w3, *w4=w+9, *w6=w+11, *w7=w+13;
  /* #0: Input 1 (x0), part 0 (phi) */
  w0 = arg[1] ? arg[1][0] : 0;
  /* #1: @1 = sin(@0) */
  w1 = sin( w0 );
  /* #2: @2 = -9.81 */
  w2 = -9.8100000000000005e+00;
  /* #3: @2 = (@2*@1) */
  w2 *= w1;
  /* #4: Input 1 (x0), part 1 (dphi) */
  w1 = arg[1] ? arg[1][1] : 0;
  /* #5: @3 = (2.*@1) */
  w3 = (2.* w1 );
  /* #6: @2 = (@2-@3) */
  w2 -= w3;
  /* #7: Input 2 (u0), part 0 (u) */
  w3 = arg[2] ? arg[2][0] : 0;
  /* #8: @2 = (@2+@3) */
  w2 += w3;
  /* #9: @4 = vertcat(@1, @2) */
  rr=w4;
  *rr++ = w1;
  *rr++ = w2;
  /* #10: Output 0 (xdot) */
  if (res[0]) copy(w4, 2, res[0]);
  /* #11: @4 = vertcat(@0, @1) */
  rr=w4;
  *rr++ = w0;
  *rr++ = w1;
  /* #12: @5 = zeros(2x1, 0 nnz) */
  /* #13: Input 3 (x_seed), part 0 (x_seed) */
  if (arg[3])
    copy(arg[3], 2, w6);
  else 
    fill(w6, 2, 0);
  /* #14: @7 = fwd1_tmp(@4, @5, @6) */
  arg1[0]=w4;
  arg1[1]=0;
  arg1[2]=w6;
  res1[0]=w7;
  if (f0(arg1, res1, iw, w, 0)) return 1;
  /* #15: @5 = zeros(2x1, 0 nnz) */
  /* #16: Input 4 (u_seed), part 0 (u_seed) */
  w0 = arg[4] ? arg[4][0] : 0;
  /* #17: @1 = fwd1_tmp(@3, @5, @0) */
  arg1[0]=(&w3);
  arg1[1]=0;
  arg1[2]=(&w0);
  res1[0]=(&w1);
  if (f1(arg1, res1, iw, w, 0)) return 1;
  /* #18: @4 = dense(@1) */
    project((&w1), s3, w4, s2, w);
  /* #19: @7 = (@7+@4) */
  for (i=0, rr=w7, cs=w4; i<2; ++i) (*rr++) += (*cs++);
  /* #20: Output 1 (xdot_sens) */
  if (res[1]) copy(w7, 2, res[1]);
  return 0;
}

void pendulum_ode_sens_incref(void) {
}

void pendulum_ode_sens_decref(void) {
}

int pendulum_ode_sens_n_in(void) { return 5;}

int pendulum_ode_sens_n_out(void) { return 2;}

const char* pendulum_ode_sens_name_in(int i){
  switch (i) {
  case 0: return "t";
  case 1: return "x0";
  case 2: return "u0";
  case 3: return "x_seed";
  case 4: return "u_seed";
  default: return 0;
  }
}

const char* pendulum_ode_sens_name_out(int i){
  switch (i) {
  case 0: return "xdot";
  case 1: return "xdot_sens";
  default: return 0;
  }
}

const int* pendulum_ode_sens_sparsity_in(int i) {
  switch (i) {
  case 0: return s6;
  case 1: return s2;
  case 2: return s6;
  case 3: return s2;
  case 4: return s6;
  default: return 0;
  }
}

const int* pendulum_ode_sens_sparsity_out(int i) {
  switch (i) {
  case 0: return s2;
  case 1: return s2;
  default: return 0;
  }
}

int pendulum_ode_sens_work(int *sz_arg, int* sz_res, int *sz_iw, int *sz_w) {
  if (sz_arg) *sz_arg = 10;
  if (sz_res) *sz_res = 5;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 15;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif