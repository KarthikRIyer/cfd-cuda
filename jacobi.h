#pragma once

void jacobistep(double *psinew, double *psi, int m, int n);

double deltasq(double *newarr, double *oldarr, int m, int n);

void jacobiiter_gpu(double *psi, int m, int n, int numiter, double &error);