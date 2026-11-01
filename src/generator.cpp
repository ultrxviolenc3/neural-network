#include "generator.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <random>


void gen_squares(int size, double cx0, double cy0, double a0,
                 double cx1, double cy1, double a1) {
    std::ofstream f("point.txt");
    for (int i = 0; i < size; i++) {
        int class_l = std::rand() % 2;
        if (class_l == 0) {
            f << cx0 - a0 / 2 + a0 * (double)std::rand() / RAND_MAX << " "
              << cy0 - a0 / 2 + a0 * (double)std::rand() / RAND_MAX << " "
              << class_l << std::endl;
        } else {
            f << cx1 - a1 / 2 + a1 * (double)std::rand() / RAND_MAX << " "
              << cy1 - a1 / 2 + a1 * (double)std::rand() / RAND_MAX << " "
              << class_l << std::endl;
        }
    }
}


void gen_circle(int n0, int n1, double R, double a) {
    std::ofstream f("point.txt");
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<> dist(-a, a);
    int count = 0;
    while (count < n0) {
        double x = dist(gen), y = dist(gen);
        if (x * x + y * y <= R * R) {
            f << x << " " << y << " 0" << std::endl;
            count++;
        }
    }
    count = 0;
    while (count < n1) {
        double x = dist(gen), y = dist(gen);
        if (x * x + y * y > R * R) {
            f << x << " " << y << " 1" << std::endl;
            count++;
        }
    }
}


void gen_sine(int n0, int n1, double h, double w, double a) {
    std::ofstream f("point.txt");
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist_x(-2.0, 2.0);
    std::uniform_real_distribution<double> dist_y(-2.0, 2.0);
    int count = 0;
    while (count < n0) {
        double x = dist_x(gen), y = dist_y(gen);
        double upper = h * std::sin(w * x) + a / 2.0;
        double lower = h * std::sin(w * x) - a / 2.0;
        if (y > lower && y < upper) {
            f << x << " " << y << " 0" << std::endl;
            count++;
        }
    }
    count = 0;
    while (count < n1) {
        double x = dist_x(gen), y = dist_y(gen);
        double upper = h * std::sin(w * x) + a / 2.0;
        double lower = h * std::sin(w * x) - a / 2.0;
        if (y <= lower || y >= upper) {
            f << x << " " << y << " 1" << std::endl;
            count++;
        }
    }
}
