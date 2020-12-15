#pragma once

void writedatafiles(float *psi, int m, int n, int scale);

void writeplotfile(int m, int n, int scale);

void hue2rgb(double hue, int *r, int *g, int *b);

double colfunc(double x);
