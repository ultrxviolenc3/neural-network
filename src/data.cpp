#include "data.h"

#include <algorithm>
#include <iostream>
#include <random>

std::vector<Point> read(const std::string &file, int size) {
    std::vector<Point> vec;
    vec.reserve(size);
    std::ifstream f(file);
    if (!f) {
        std::cout << "ошибка открытия файла" << std::endl;
        return vec;
    }
    Point p;
    while (f >> p.x >> p.y >> p.label) vec.push_back(p);
    return vec;
}

Norm normp(const std::vector<Point> &traindata) {
    Norm obb;
    if (traindata.empty()) return {0, 0, 0, 0};

    obb.xmin = obb.xmax = traindata[0].x;
    obb.ymin = obb.ymax = traindata[0].y;
    for (const auto &p : traindata) {
        if (p.x < obb.xmin) obb.xmin = p.x;
        if (p.x > obb.xmax) obb.xmax = p.x;
        if (p.y < obb.ymin) obb.ymin = p.y;
        if (p.y > obb.ymax) obb.ymax = p.y;
    }
    return obb;
}

void normirovka(double &x, double &y, const Norm &obb) {
    double dx = obb.xmax - obb.xmin;
    double dy = obb.ymax - obb.ymin;
    if (dx == 0.0) dx = 1.0;
    if (dy == 0.0) dy = 1.0;
    x = 2.0 * (x - obb.xmin) / dx - 1.0;
    y = 2.0 * (y - obb.ymin) / dy - 1.0;
}

void split(std::vector<Point> &alldata, std::vector<Point> &traindata,
           std::vector<Point> &validdata) {
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(alldata.begin(), alldata.end(), g);
    int trainsize = static_cast<int>(alldata.size() * 0.8);
    traindata.assign(alldata.begin(), alldata.begin() + trainsize);
    validdata.assign(alldata.begin() + trainsize, alldata.end());
}

void start(std::ofstream &f) {
    f << "set xlabel \"x\"" << std::endl
      << "set ylabel \"y\"" << std::endl
      << "set size ratio -1" << std::endl
      << "set palette defined(0 \"blue\", 1 \"red\")" << std::endl
      << "set key off" << std::endl
      << "plot \"pointnew.txt\" using 1:2:3 with points pt 7 ps 0.1 lc palette" << std::endl;
}
