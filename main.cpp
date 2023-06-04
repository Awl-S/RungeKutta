#include <iostream>
#include <vector>
#include <functional>

class RungeKutta4 {
private:
    std::function<std::vector<double>(double, const std::vector<double>&)> func;

public:
    RungeKutta4(std::function<std::vector<double>(double, const std::vector<double>&)> f) : func(f) {}

    std::vector<double> solve(std::vector<double> y0, double t0, double t, double dt) {
        if (dt <= 0.0 || t0 >= t) {
            throw std::invalid_argument("invalid arguments");
        }

        std::vector<double> y = y0;
        for (double ti = t0; ti < t; ti += dt) {
            std::vector<double> k1 = mul(dt, this->func(ti, y));
            std::vector<double> k2 = mul(dt, this->func(ti + dt / 2.0, add(y, div(2.0, k1))));
            std::vector<double> k3 = mul(dt, this->func(ti + dt / 2.0, add(y, div(2.0, k2))));
            std::vector<double> k4 = mul(dt, this->func(ti + dt, add(y, k3)));

            y = add(y, div(1.0/6.0, add(add(k1, mul(2.0, k2)), add(mul(2.0, k3), k4))));
        }

        return y;
    }

private:
    std::vector<double> mul(double scalar, const std::vector<double>& vec) {
        std::vector<double> result(vec.size());
        for (size_t i = 0; i < vec.size(); ++i) {
            result[i] = scalar * vec[i];
        }
        return result;
    }

    std::vector<double> div(double scalar, const std::vector<double>& vec) {
        return mul(1.0 / scalar, vec);
    }

    std::vector<double> add(const std::vector<double>& vec1, const std::vector<double>& vec2) {
        if (vec1.size() != vec2.size()) {
            throw std::invalid_argument("vectors are not the same size");
        }
        std::vector<double> result(vec1.size());
        for (size_t i = 0; i < vec1.size(); ++i) {
            result[i] = vec1[i] + vec2[i];
        }
        return result;
    }
};

int main() {
    setlocale(LC_ALL, "RU.UTF-8");
    // ОДУ первого порядка: dy/dt = -2y
    {
        std::function<std::vector<double>(double, const std::vector<double>&)> func = [](double t, const std::vector<double>& y) {
            return std::vector<double>{-2.0 * y[0]};
        };
        RungeKutta4 rk4(func);
        std::vector<double> y0 = {1.0};
        double t0 = 0.0;
        double t = 2.0;
        double dt = 0.01;
        std::vector<double> y = rk4.solve(y0, t0, t, dt);
        std::cout << "Решение ОДУ первого порядка в t=" << t << ": " << y[0] << std::endl;
    }

    // Система ОДУ: dy/dt = z, dz/dt = -y
    {
        std::function<std::vector<double>(double, const std::vector<double>&)> func = [](double t, const std::vector<double>& y) {
            return std::vector<double>{y[1], -y[0]};
        };
        RungeKutta4 rk4(func);
        std::vector<double> y0 = {0.0, 1.0};
        double t0 = 0.0;
        double t = 2.0;
        double dt = 0.01;
        std::vector<double> y = rk4.solve(y0, t0, t, dt);
        std::cout << "Решение системы ОДУ в t=" << t << ": y=" << y[0] << ", z=" << y[1] << std::endl;
    }

    // ОДУ второго порядка: d²y/dt² = -y (представленное как система ОДУ первого порядка)
    {
        std::function<std::vector<double>(double, const std::vector<double>&)> func = [](double t, const std::vector<double>& y) {
            return std::vector<double>{y[1], -y[0]};
        };
        RungeKutta4 rk4(func);
        std::vector<double> y0 = {0.0, 1.0};
        double t0 = 0.0;
        double t = 2.0;
        double dt = 0.01;
        std::vector<double> y = rk4.solve(y0, t0, t, dt);
        std::cout << "Решение ОДУ второго порядка в t=" << t << ": y=" << y[0] << ", y'=" << y[1] << std::endl;
    }

    return 0;
}
