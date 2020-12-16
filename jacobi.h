#pragma once

void jacobistep(float *psinew, float *psi, int m, int n);

double deltasq(float *newarr, float *oldarr, int m, int n);

void jacobiiter_gpu(float *psi, int m, int n, int numiter, float &error);