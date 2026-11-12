#include <boost/program_options.hpp>

#include <fstream>
#include <iostream>
#include <regex>
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

std::string check_value(const opt::variables_map& vm, const std::string& name,
                        const std::regex& pattern) {
    std::string value = vm[name].as<std::string>();
    if (!std::regex_match(value, pattern)) {
        throw std::runtime_error("Неверное значение " + name + ": " + value);
    }
    return value;
}

bool read_options(int argc, char* argv[], Settings& settings) {
    opt::options_description desc("Параметры нейронной сети");
    desc.add_options()
        ("help,h", "Показать список параметров")
        ("config,c", opt::value<std::string>(), "Путь к конфигурационному файлу")
        ("mode,m", opt::value<std::string>(), "1 - квадраты, 2 - круг, 3 - синусоида")
        ("size,n", opt::value<std::string>(), "Количество точек")
        ("type,t", opt::value<std::string>(), "float или double")
        ("lr", opt::value<std::string>(), "Скорость обучения, например 0.05")
        ("epochs,e", opt::value<std::string>(), "Количество эпох")
        ("hidden1", opt::value<std::string>(), "Нейронов в первом скрытом слое")
        ("hidden2", opt::value<std::string>(), "Нейронов во втором скрытом слое");

    opt::variables_map vm;
    opt::store(opt::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
        std::cout << desc << '\n';
        return false;
    }

    std::string config_path = vm.count("config")
        ? vm["config"].as<std::string>() : "config.cfg";
    std::ifstream config_file(config_path);
    if (config_file) {
        opt::store(opt::parse_config_file(config_file, desc), vm);
    } else if (vm.count("config")) {
        throw std::runtime_error("Не удалось открыть конфиг: " + config_path);
    } else {
        std::cout << "Файл config.cfg не найден, используются значения по умолчанию.\n";
    }
    opt::notify(vm);

    const std::regex mode_re(R"([123])");
    const std::regex size_re(R"([1-9]\d+)");
    const std::regex integer_re(R"([1-9]\d*)");
    const std::regex type_re(R"(float|double)");
    const std::regex lr_re(R"((0\.\d*[1-9]\d*|[1-9]\d*(\.\d+)?))");

    if (vm.count("mode"))
        settings.mode = std::stoi(check_value(vm, "mode", mode_re));
    if (vm.count("size"))
        settings.size = std::stoi(check_value(vm, "size", size_re));
    if (vm.count("type"))
        settings.type = check_value(vm, "type", type_re);
    if (vm.count("lr"))
        settings.lr = std::stod(check_value(vm, "lr", lr_re));
    if (vm.count("hidden1"))
        settings.hidden1 = std::stoi(check_value(vm, "hidden1", integer_re));
    if (vm.count("hidden2"))
        settings.hidden2 = std::stoi(check_value(vm, "hidden2", integer_re));
    if (vm.count("epochs"))
        settings.epochs = std::stoi(check_value(vm, "epochs", integer_re));
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
        std::cerr << "Ошибка создания файлов результата\n";
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
        std::cerr << "Ошибка параметров: " << e.what() << '\n';
        std::cerr << "Используйте --help для списка параметров.\n";
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
