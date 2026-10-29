#include "neuralnetwork.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>

template <typename T>
T Sigmoida(T x) {
    return T{1} / (T{1} + std::exp(-x));
}

template <typename T>
T Sigmoida2(T x) {
    T s = Sigmoida(x);
    return s * (T{1} - s);
}

template <typename T>
void Neuralnetwork<T>::set_norm(const Norm &p) {
    params = p;
}

template <typename T>
Neuralnetwork<T>::Neuralnetwork(const std::vector<int> &inner_size, T learning_rate) : lr(learning_rate) {
    layers.reserve(inner_size.size() + 2);
    layers.push_back(2);
    for (int h : inner_size) layers.push_back(h);
    layers.push_back(1);
    weights();
}

template <typename T>
void Neuralnetwork<T>::weights() {
    int l = layers.size();
    w.resize(l);
    b.resize(l);
    for (int k = 1; k < l; k++) {
        int prev = layers[k - 1];
        int curr = layers[k];
        b[k].resize(curr);
        for (int i = 0; i < curr; i++) b[k][i] = T{0};
        w[k].resize(curr);
        for (int i = 0; i < curr; i++) {
            w[k][i].resize(prev);
            for (int j = 0; j < prev; j++) {
                T random_number = std::rand();
                T max_number = RAND_MAX;
                w[k][i][j] = random_number / max_number * T{2} - T{1};
            }
        }
    }
}

template <typename T>
T Neuralnetwork<T>::forward(double x, double y) {
    normirovka(x, y, params);
    int l = layers.size();
    a.resize(l);
    f.resize(l);
    a[0].resize(2);
    a[0][0] = x;
    a[0][1] = y;
    for (int k = 1; k < l; k++) {
        int prevn = layers[k - 1];
        int currn = layers[k];
        f[k].resize(currn);
        a[k].resize(currn);
        for (int i = 0; i < currn; i++) {
            f[k][i] = b[k][i];
            for (int j = 0; j < prevn; j++) {
                f[k][i] += w[k][i][j] * a[k - 1][j];
            }
            a[k][i] = Sigmoida(f[k][i]);
        }
    }
    return a[l - 1][0];
}

template <typename T>
void Neuralnetwork<T>::tofile(const std::vector<Point> &data, std::ofstream &file) {
    for (const auto &p : data) {
        int pred = predict(p);
        file << p.x << " " << p.y << " " << pred << std::endl;
    }
}

template <typename T>
void Neuralnetwork<T>::backward(int truelabel) {
    int l = layers.size();
    delta.resize(l);
    delta[l - 1].resize(1);
    T p = a[l - 1][0];
    delta[l - 1][0] = p - truelabel;
    for (int k = l - 2; k >= 1; k--) {
        int currn = layers[k];
        int nextn = layers[k + 1];
        delta[k].resize(currn);
        for (int i = 0; i < currn; i++) {
            delta[k][i] = T{0};
            for (int j = 0; j < nextn; j++) {
                delta[k][i] += w[k + 1][j][i] * delta[k + 1][j];
            }
            delta[k][i] *= Sigmoida2(f[k][i]);
        }
    }
}

template <typename T>
void Neuralnetwork<T>::train_epoch(std::vector<Point> &traindata) {
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(traindata.begin(), traindata.end(), g);
    for (auto &p : traindata) {
        forward(p.x, p.y);
        backward(p.label);
        update();
    }
}

template <typename T>
void Neuralnetwork<T>::update() {
    int l = layers.size();
    for (int k = 1; k < l; k++) {
        int prevn = layers[k - 1];
        int currn = layers[k];
        for (int i = 0; i < currn; i++) {
            for (int j = 0; j < prevn; j++) {
                w[k][i][j] -= lr * delta[k][i] * a[k - 1][j];
            }
            b[k][i] -= lr * delta[k][i];
        }
    }
}

template <typename T>
int Neuralnetwork<T>::predict(const Point &p) {
    T prob = forward(p.x, p.y);
    return (prob >= T{0.5}) ? 1 : 0;
}

template <typename T>
T Neuralnetwork<T>::evaluate(const std::vector<Point> &data) {
    if (data.empty()) return T{0};
    int count = 0;
    for (const auto &p : data) {
        int pred = predict(p);
        if (pred == p.label) count++;
    }
    T correct = count;
    T total = data.size();
    return correct / total;
}

template class Neuralnetwork<float>;
template class Neuralnetwork<double>;
