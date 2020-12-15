#include "cfdio.h"
#include <cmath>
#include <fstream>
#include <iostream>

void writedatafiles(float *psi, int m, int n, int scale) {
    std::ofstream cfile, vfile;
    typedef float vecvel[2];
    typedef int vecrgb[3];

    vecvel *vel;
    vecrgb *rgb;

    double modvsq, hue;
    int nvel, nrgb;

    std::cout << "\nWriting data files.\n";

    vel = new vecvel[m * n];
    rgb = new vecrgb[m * n];

    //calculate velocities and hues

    double v1, v2;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            vel[i * m + j][0] = 0.5f * (psi[(m + 2) * (i + 1) + j + 2] - psi[(m + 2) * (i + 1) + j]);
            vel[i * m + j][1] = -0.5f * (psi[(m + 2) * (i + 2) + j + 1] - psi[(m + 2) * (i) + j + 1]);
            v1 = vel[i * m + j][0];
            v2 = vel[i * m + j][1];

            modvsq = v1 * v1 + v2 * v2;
            hue = pow(modvsq, 0.4);
            hue2rgb(hue, &(rgb[i * m + j][0]), &(rgb[i * m + j][1]), &(rgb[i * m + j][2]));
        }
    }

    //write data
    cfile.open("colormap.dat");
    vfile.open("velocity.dat");

    for (int i = 0; i < m; i++) {
        int ix = i + 1;
        for (int j = 0; j < n; j++) {
            int iy = j + 1;
            cfile << ix << " " << iy << " " << rgb[i * m + j][0] << " " << rgb[i * m + j][1] << " " << rgb[i * m + j][2]
                  << "\n";
            if ((ix - 1) % scale == (scale - 1) / 2 && (iy - 1) % scale == (scale - 1) / 2) {
                vfile << ix << " " << iy << " " << vel[i * m + j][0] << " " << vel[i * m + j][1] << "\n";
            }
        }
    }
    vfile.close();
    cfile.close();

    delete[] rgb;
    delete[] vel;

    std::cout << "...Done!!\n";
}

void writeplotfile(int m, int n, int scale) {
    std::ofstream gnuplot;
    gnuplot.open("cuda-cfd.plt");
    gnuplot << "set size square\n";
    gnuplot << "set key off\n";
    gnuplot << "unset xtics\n";
    gnuplot << "unset ytics\n";
    gnuplot << "set xrange [" << 1 - scale << ":" << m + scale << "]\n";
    gnuplot << "set yrange [" << 1 - scale << ":" << n + scale << "]\n";
    gnuplot << "plot \"colormap.dat\" w rgbimage, \"velocity.dat\" u 1:2:(" << scale << "*0.75*$3/sqrt($3**2+$4**2)):("
            << scale << "*0.75*$4/sqrt($3**2+$4**2)) with vectors lc rgb \"#7F7F7F\"";
    gnuplot.close();
    std::cout << "\nWritten gnuplot script 'cuda-cfd.plt'\n";
}

void hue2rgb(double hue, int *r, int *g, int *b) {
    int rgbmax = 255;

    *r = (int) (rgbmax * colfunc(hue - 1.0));
    *g = (int) (rgbmax * colfunc(hue - 0.5));
    *b = (int) (rgbmax * colfunc(hue));
}

double colfunc(double x) {
    double absx;

    double x1 = 0.2;
    double x2 = 0.5;

    absx = fabs(x);

    if (absx > x2) {
        return 0;
    } else if (absx < x1) {
        return 1.0;
    } else {
        return 1.0 - pow((absx - x1) / (x2 - x1), 2);
    }
}