
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "data.h"
#include "generator.h"
#include "neuralnetwork.h"

#include <cmath>
#include <cstdlib>
#include <vector>

TEST_CASE("Norm: minimum, maximum and normalization") {
    std::vector<Point> points = {
        {-2.0, 1.0, 0},
        { 4.0, 7.0, 1},
        { 1.0, 4.0, 0}
    };

    Norm n = normp(points);
    CHECK(n.xmin == -2.0);
    CHECK(n.xmax == 4.0);
    CHECK(n.ymin == 1.0);
    CHECK(n.ymax == 7.0);

    SUBCASE("Minimum becomes -1") {
        double x = -2.0, y = 1.0;
        normirovka(x, y, n);
        CHECK(x == doctest::Approx(-1.0));
        CHECK(y == doctest::Approx(-1.0));
    }

    SUBCASE("Maximum becomes 1") {
        double x = 4.0, y = 7.0;
        normirovka(x, y, n);
        CHECK(x == doctest::Approx(1.0));
        CHECK(y == doctest::Approx(1.0));
    }

    SUBCASE("Middle becomes 0") {
        double x = 1.0, y = 4.0;
        normirovka(x, y, n);
        CHECK(x == doctest::Approx(0.0));
        CHECK(y == doctest::Approx(0.0));
    }
}

TEST_CASE("Split: 80 percent training, 20 percent validation") {
    std::vector<Point> points;
    for (int i = 0; i < 10; ++i) {
        points.push_back({double(i), 0.0, i % 2});
    }

    std::vector<Point> train, valid;
    split(points, train, valid);
    REQUIRE(train.size() == 8);
    REQUIRE(valid.size() == 2);

    // При разделении ни одна точка не теряется и не повторяется.
    std::vector<int> count(10, 0);
    for (const auto& p : train) ++count[int(p.x)];
    for (const auto& p : valid) ++count[int(p.x)];
    for (int n : count) CHECK(n == 1);
}

TEST_CASE("Generator: points inside and outside a circle") {
    gen_circle(10, 10, 0.5, 1.0);
    std::vector<Point> points = read("point.txt", 20);
    REQUIRE(points.size() == 20);

    int inside = 0, outside = 0;
    for (const auto& p : points) {
        const double r2 = p.x * p.x + p.y * p.y;
        if (p.label == 0) {
            ++inside;
            CHECK(r2 <= 0.25 + 0.00001);
        } else {
            REQUIRE(p.label == 1);
            ++outside;
            CHECK(r2 >= 0.25 - 0.00001);
        }
    }
    CHECK(inside == 10);
    CHECK(outside == 10);
}

TEST_CASE("Neuralnetwork: prediction and evaluation") {
    SUBCASE("double") {
        Neuralnetwork<double> net({8, 8}, 0.05);
        net.set_norm({-1, 1, -1, 1});
        double p = net.forward(0.2, 0.4);
        CHECK(p >= 0.0);
        CHECK(p <= 1.0);
        Point point{0.2, 0.4, 1};
        int prediction = net.predict(point);
        CHECK((prediction == 0 || prediction == 1));
        CHECK(net.evaluate({}) == 0.0);
    }

    SUBCASE("float") {
        Neuralnetwork<float> net({8, 8}, 0.05f);
        net.set_norm({-1, 1, -1, 1});
        float p = net.forward(0.2, 0.4);
        CHECK(p >= 0.0f);
        CHECK(p <= 1.0f);
        CHECK(net.evaluate({}) == 0.0f);
    }
}

TEST_CASE("Neuralnetwork: copy constructor and assignment") {
    Neuralnetwork<double> original({4, 4}, 0.05);
    original.set_norm({-1, 1, -1, 1});
    double expected = original.forward(0.2, 0.4);

    SUBCASE("Copy constructor") {
        Neuralnetwork<double> copy(original);
        CHECK(copy.forward(0.2, 0.4) == doctest::Approx(expected));

        original.forward(0.2, 0.4);
        original.backward(1);
        original.update();
        CHECK(copy.forward(0.2, 0.4) == doctest::Approx(expected));
    }

    SUBCASE("Assignment and self-assignment") {
        Neuralnetwork<double> assigned({3}, 0.1);
        assigned = original;
        CHECK(assigned.forward(0.2, 0.4) == doctest::Approx(expected));
        assigned = assigned;
        CHECK(assigned.forward(0.2, 0.4) == doctest::Approx(expected));
    }
}
