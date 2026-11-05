#include <boost/program_options.hpp>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "data.h"
#include "generator.h"
#include "neuralnetwork.h"

namespace opt = boost::program_options;

struct Settings {
    int mode = 1;
    int size = 1000;
    std::string type = "double";
    double lr = 0.05;
    int epochs = 15;
    int hidden1 = 8;
    int hidden2 = 8;
};

bool read_options(int argc, char* argv[], Settings& settings) {
    opt::options_description desc("Параметры нейронной сети");
    desc.add_options()
        ("help,h", "Показать список параметров")
        ("config,c", opt::value<std::string>(), "Путь к конфигурационному файлу (по умолчанию config.cfg)")
        ("mode,m", opt::value<int>(), "1 - квадраты, 2 - круг, 3 - синусоида")
        ("size,n", opt::value<int>(), "Количество точек")
        ("type,t", opt::value<std::string>(), "float или double")
        ("lr", opt::value<double>(), "Скорость обучения, например 0.05")
        ("epochs,e", opt::value<int>(), "Количество эпох")
        ("hidden1", opt::value<int>(), "Нейронов в первом скрытом слое")
        ("hidden2", opt::value<int>(), "Нейронов во втором скрытом слое");

    opt::variables_map vm;
    opt::store(opt::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
        std::cout << desc << '\n';
        return false;
    }

    const std::string config_path = vm.count("config") ? vm["config"].as<std::string>() : "config.cfg";
    std::ifstream config_file(config_path);
    if (config_file) {
        opt::store(opt::parse_config_file(config_file, desc), vm);
    } else if (vm.count("config")) {
        throw std::runtime_error("Не удалось открыть конфиг: " + config_path);
    } else {
        std::cout << "Файл config.cfg не найден.\n";
    }
    opt::notify(vm);

    if (vm.count("mode")) settings.mode = vm["mode"].as<int>();
    if (vm.count("size")) settings.size = vm["size"].as<int>();
    if (vm.count("type")) settings.type = vm["type"].as<std::string>();
    if (vm.count("lr")) settings.lr = vm["lr"].as<double>();
    if (vm.count("hidden1")) settings.hidden1 = vm["hidden1"].as<int>();
    if (vm.count("hidden2")) settings.hidden2 = vm["hidden2"].as<int>();
    if (vm.count("epochs"))
        settings.epochs = vm["epochs"].as<int>();
    else if (settings.mode == 3)
        settings.epochs = 24;
    else
        settings.epochs = 15;

    return true;
}

template <typename T>
int run(const Settings& s) {
    if (s.mode == 1) {
        gen_squares(s.size, 0, 1, 2, 2.0, 2.6, 2);
    } else if (s.mode == 2) {
        gen_circle(s.size / 2, s.size - s.size / 2, 0.5, 1);
    } else {
        gen_sine(s.size / 2, s.size - s.size / 2, 0.4, 2, 1);
    }

    std::vector<Point> alldata = read("point.txt", s.size);
    if (alldata.empty()) {
        std::cerr << "Файл с точками пуст или не читается\n";
        return 1;
    }

    std::vector<Point> traindata, validdata;
    split(alldata, traindata, validdata);

    std::vector<int> inner = {s.hidden1, s.hidden2};
    T lr = s.lr;
    Neuralnetwork<T> net(inner, lr);
    net.set_norm(normp(traindata));

    for (int e = 0; e < s.epochs; e++) {
        net.train_epoch(traindata);
        T valid_acc = net.evaluate(validdata);
        std::cout << "Эпоха " << e + 1 << ": valid accuracy = " << valid_acc << '\n';
    }

    std::ofstream result("pointnew.txt");
    std::ofstream plot("point.plt");
    if (!result || !plot) {
        std::cerr << "Ошибка создания файлов\n";
        return 1;
    }
    start(plot);
    net.tofile(alldata, result);
    return 0;
}

int main(int argc, char* argv[]) {
    Settings settings;
    try {
        if (!read_options(argc, argv, settings)) return 0;
    } catch (const std::exception& e) {
        return 1;
    }

    std::cout << "Параметры запуска: mode=" << settings.mode
              << ", size=" << settings.size
              << ", type=" << settings.type
              << ", lr=" << settings.lr
              << ", epochs=" << settings.epochs
              << ", hidden=" << settings.hidden1 << ',' << settings.hidden2 << '\n';

    if (settings.type == "float") return run<float>(settings);
    return run<double>(settings);
}
