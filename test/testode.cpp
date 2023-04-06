#define CATCH_CONFIG_MAIN
#include <string>

#include "../include/ode.hpp"
#include "catch2/catch.hpp"

using namespace tinyode;

TEST_CASE("Simple euler.", "[ODE]") {
    auto ode_function = [](const double t, const Eigen::Matrix<double, 1, 1>& x) -> Eigen::Matrix<double, 1, 1> {
        return Eigen::Matrix<double, 1, 1>(-x(0, 0));
    };

    Eigen::Matrix<double, 1, 1> initial_conditions;
    initial_conditions << 1.0;

    TimeSpan<double> time_span(0.0, 1.0);
    Options<double, 1> options;
    options.fixed_time_step    = 0.01;
    options.integration_method = IntegrationMethod::EULER;

    Result<double, 1> result = ode<double, 1>(ode_function, time_span, initial_conditions, options);

    REQUIRE(result.t.size() == 101);
    REQUIRE(result.x.cols() == 101);

    for (int i = 0; i < result.t.size(); ++i) {
        double expected = std::exp(-result.t(i));
        CHECK(std::abs(result.x(0, i) - expected) < 1e-2);
    }
}

TEST_CASE("Simple RK4.", "[ODE]") {
    auto ode_function = [](const double t, const Eigen::Matrix<double, 1, 1>& x) -> Eigen::Matrix<double, 1, 1> {
        return Eigen::Matrix<double, 1, 1>(-x(0, 0));
    };

    Eigen::Matrix<double, 1, 1> initial_conditions;
    initial_conditions << 1.0;

    TimeSpan<double> time_span(0.0, 1.0);
    Options<double, 1> options;
    options.fixed_time_step    = 0.01;
    options.integration_method = IntegrationMethod::RK4;

    Result<double, 1> result = ode<double, 1>(ode_function, time_span, initial_conditions, options);

    REQUIRE(result.t.size() == 101);
    REQUIRE(result.x.cols() == 101);

    for (int i = 0; i < result.t.size(); ++i) {
        double expected = std::exp(-result.t(i));
        CHECK(std::abs(result.x(0, i) - expected) < 1e-2);
    }
}