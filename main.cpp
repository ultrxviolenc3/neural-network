#include <fstream>
#include <iostream>
#include <vector>

#include "data.h"
#include "generator.h"
#include "neuralnetwork.h"

template <typename T>
int run(int mode, int size) {
    int epochs = 15;
    if (mode == 1) {
        gen_squares(size, 0, 1, 2, 2.0, 2.6, 2);
    } else if (mode == 2) {
        gen_circle(size / 2, size - size / 2, 0.5, 1);
    } else {
        gen_sine(size / 2, size - size / 2, 0.4, 2, 1);
        epochs = 24;
    }

    std::vector<Point> alldata = read("point.txt", size);
    if (alldata.empty()) {
        std::cout << "Файл пуст" << std::endl;
        return 1;
    }

    std::vector<Point> traindata, validdata;
    split(alldata, traindata, validdata);

    std::vector<int> inner = {8, 8};
    T lr = 0.05;
    Neuralnetwork<T> net(inner, lr);
    Norm params = normp(traindata);
    net.set_norm(params);

    for (int e = 0; e < epochs; e++) {
        net.train_epoch(traindata);
        T valid_acc = net.evaluate(validdata);
        std::cout << e + 1 << std::endl;
        std::cout << "valid accuracy: " << valid_acc << std::endl;
    }

    std::ofstream file("pointnew.txt");
    std::ofstream plot("point.plt");
    if (!file || !plot) {
        std::cout << "Ошибка создания файлов результата" << std::endl;
        return 1;
    }
    start(plot);
    net.tofile(alldata, file);
    return 0;
}

int main() {
    int mode;
    std::cout << "1 - квадраты, 2 - круг, 3 - синусоида: " << std::endl;
    if (!(std::cin >> mode) || mode < 1 || mode > 3) {
        std::cout << "Неверный вариант" << std::endl;
        return 1;
    }

    int size;
    std::cout << "Введите количество точек: " << std::endl;
    if (!(std::cin >> size) || size < 10) {
        std::cout << "Введите не менее 10 точек" << std::endl;
        return 1;
    }

    int type;
    std::cout << "Выберите тип чисел: 1 - float, 2 - double: " << std::endl;
    if (!(std::cin >> type) || (type != 1 && type != 2)) {
        std::cout << "Неверный тип" << std::endl;
        return 1;
    }

    if (type == 1) return run<float>(mode, size);
    return run<double>(mode, size);
}
