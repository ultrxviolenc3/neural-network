#ifndef NEYRO_NEURALNETWORK_H
#define NEYRO_NEURALNETWORK_H

#include "data.h"
#include <fstream>
#include <vector>

template <typename T>
class Neuralnetwork {
private:
    Norm params;
    std::vector<int> layers;
    std::vector<std::vector<std::vector<T>>> w;
    std::vector<std::vector<T>> b;
    std::vector<std::vector<T>> a;
    std::vector<std::vector<T>> f;
    std::vector<std::vector<T>> delta;
    T lr;

public:
    Neuralnetwork(const std::vector<int> &inner_size, T learning_rate);
    void set_norm(const Norm &p);
    void weights();
    T forward(double x, double y);
    void tofile(const std::vector<Point> &data, std::ofstream &file);
    void backward(int truelabel);
    void train_epoch(std::vector<Point> &traindata);
    void update();
    int predict(const Point &p);
    T evaluate(const std::vector<Point> &data);
};

extern template class Neuralnetwork<float>;
extern template class Neuralnetwork<double>;

#endif
