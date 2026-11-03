#ifndef NEYRO_DATA_H
#define NEYRO_DATA_H

#include <fstream>
#include <string>
#include <vector>

class Point {
public:
    double x, y;
    int label;
};

class Norm {
public:
    double xmin, xmax, ymin, ymax;
};

std::vector<Point> read(const std::string &file, int size);
Norm normp(const std::vector<Point> &traindata);
void normirovka(double &x, double &y, const Norm &obb);
void split(std::vector<Point> &alldata, std::vector<Point> &traindata,
           std::vector<Point> &validdata);
void start(std::ofstream &f);

#endif
