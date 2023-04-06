#define CATCH_CONFIG_MAIN
#include <string>

#include "../include/ode.hpp"
#include "catch2/catch.hpp"

using namespace tinyode;

TEST_CASE("Solve exponential decay ode using Euler method.", "[ode]") {
    // Define the ODE function
    auto ode_function = [](const double t, const Eigen::Matrix<double, 1, 1>& x) -> Eigen::Matrix<double, 1, 1> {
        return Eigen::Matrix<double, 1, 1>(-x(0, 0));
    };

    // Set initial conditions
    Eigen::Matrix<double, 1, 1> initial_conditions;
    initial_conditions << 1.0;

    // Set time span and options for Euler method
    TimeSpan<double> time_span(0.0, 1.0);
    Options<double, 1> options;
    options.fixed_time_step    = 0.01;
    options.integration_method = IntegrationMethod::EULER;

    // Solve the ODE
    Result<double, 1> result = ode<double, 1>(ode_function, time_span, initial_conditions, options);

    // Check the size of time and solution vectors
    REQUIRE(result.t.size() == 101);
    REQUIRE(result.x.cols() == 101);

    // Check the computed solution against the analytical solution
    for (int i = 0; i < result.t.size(); ++i) {
        double expected = std::exp(-result.t(i));
        CHECK(std::abs(result.x(0, i) - expected) < 1e-1);
    }
}

TEST_CASE("Solve exponential decay ode using RK4 method.", "[ode]") {
    // Define the ODE function
    auto ode_function = [](const double t, const Eigen::Matrix<double, 1, 1>& x) -> Eigen::Matrix<double, 1, 1> {
        return Eigen::Matrix<double, 1, 1>(-x(0, 0));
    };

    // Set initial conditions
    Eigen::Matrix<double, 1, 1> initial_conditions;
    initial_conditions << 1.0;

    // Set time span and options for RK4 method
    TimeSpan<double> time_span(0.0, 1.0);
    Options<double, 1> options;
    options.fixed_time_step    = 0.01;
    options.integration_method = IntegrationMethod::RK4;

    // Solve the ODE
    Result<double, 1> result = ode<double, 1>(ode_function, time_span, initial_conditions, options);

    // Check the size of time and solution vectors
    REQUIRE(result.t.size() == 101);
    REQUIRE(result.x.cols() == 101);

    // Check the computed solution against the analytical solution
    for (int i = 0; i < result.t.size(); ++i) {
        double expected = std::exp(-result.t(i));
        CHECK(std::abs(result.x(0, i) - expected) < 1e-2);
    }
}

TEST_CASE("Solve exponential decay ode using RK5 method.", "[ode]") {
    // Define the ODE function
    auto ode_function = [](const double t, const Eigen::Matrix<double, 1, 1>& x) -> Eigen::Matrix<double, 1, 1> {
        return Eigen::Matrix<double, 1, 1>(-x(0, 0));
    };

    // Set initial conditions
    Eigen::Matrix<double, 1, 1> initial_conditions;
    initial_conditions << 1.0;

    // Set time span and options for RK5 method
    TimeSpan<double> time_span(0.0, 1.0);
    Options<double, 1> options;
    options.fixed_time_step    = 0.01;
    options.integration_method = IntegrationMethod::RK5;

    // Solve the ODE
    Result<double, 1> result = ode<double, 1>(ode_function, time_span, initial_conditions, options);

    // Check the size of time and solution vectors
    REQUIRE(result.t.size() == 101);
    REQUIRE(result.x.cols() == 101);

    // Check the computed solution against the analytical solution
    for (int i = 0; i < result.t.size(); ++i) {
        double expected = std::exp(-result.t(i));
        CHECK(std::abs(result.x(0, i) - expected) < 1e-3);
    }
}

TEST_CASE("Solve exponential decay ode using RK45 method.", "[ode]") {
    // Define the ODE function
    auto ode_function = [](const double t, const Eigen::Matrix<double, 1, 1>& x) -> Eigen::Matrix<double, 1, 1> {
        return Eigen::Matrix<double, 1, 1>(-x(0, 0));
    };

    // Set initial conditions
    Eigen::Matrix<double, 1, 1> initial_conditions;
    initial_conditions << 1.0;

    // Set time span and options for RK45 method
    TimeSpan<double> time_span(0.0, 1.0);
    Options<double, 1> options;
    options.integration_method = IntegrationMethod::RK45;
    options.absolute_tolerance = 1e-6;
    options.relative_tolerance = 1e-6;

    // Solve the ODE
    Result<double, 1> result = ode<double, 1>(ode_function, time_span, initial_conditions, options);

    // Check the computed solution against the analytical solution
    for (int i = 0; i < result.t.size(); ++i) {
        double expected = std::exp(-result.t(i));
        CHECK(std::abs(result.x(0, i) - expected) < options.absolute_tolerance);
    }
}

TEST_CASE("Solve harmonic oscillator ode using Runge-Kutta-Fehlberg 45 method.", "[ode]") {
    // Define the ODE function
    auto ode_function = [](const double t, const Eigen::Matrix<double, 2, 1>& x) -> Eigen::Matrix<double, 2, 1> {
        Eigen::Matrix<double, 2, 1> dxdt;
        dxdt(0) = x(1);
        dxdt(1) = -x(0);
        return dxdt;
    };

    // Set initial conditions
    Eigen::Matrix<double, 2, 1> initial_conditions;
    initial_conditions << 1.0, 0.0;

    // Set time span and options for Runge-Kutta-Fehlberg 45 method
    TimeSpan<double> time_span(0.0, 10.0);
    Options<double, 2> options;
    options.absolute_tolerance = 1e-6;
    options.relative_tolerance = 1e-6;
    options.integration_method = IntegrationMethod::RK45;

    // Solve the ODE
    Result<double, 2> result = ode<double, 2>(ode_function, time_span, initial_conditions, options);

    // Check the computed solution against the analytical solution
    for (int i = 0; i < result.t.size(); ++i) {
        double expected_x = std::cos(result.t(i));
        double expected_v = -std::sin(result.t(i));
        CHECK(std::abs(result.x(0, i) - expected_x) < 1e-5);
        CHECK(std::abs(result.x(1, i) - expected_v) < 1e-5);
    }
}
