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
    REQUIRE(result.x.size() == 101);

    // Check the computed solution against the analytical solution
    for (int i = 0; i < result.t.size(); ++i) {
        double expected = std::exp(-result.t[i]);
        CHECK(std::abs(result.x[i](0) - expected) < 1e-1);
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
    REQUIRE(result.x.size() == 101);

    // Check the computed solution against the analytical solution
    for (int i = 0; i < result.t.size(); ++i) {
        double expected = std::exp(-result.t[i]);
        CHECK(std::abs(result.x[i](0) - expected) < 1e-2);
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
    REQUIRE(result.x.size() == 101);

    // Check the computed solution against the analytical solution
    for (int i = 0; i < result.t.size(); ++i) {
        double expected = std::exp(-result.t[i]);
        CHECK(std::abs(result.x[i](0) - expected) < 1e-3);
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
        double expected = std::exp(-result.t[i]);
        CHECK(std::abs(result.x[i](0) - expected) < options.absolute_tolerance);
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
        double expected_x = std::cos(result.t[i]);
        double expected_v = -std::sin(result.t[i]);
        CHECK(std::abs(result.x[i](0) - expected_x) < 1e-5);
        CHECK(std::abs(result.x[i](1) - expected_v) < 1e-5);
    }
}


TEST_CASE("Solve harmonic oscillator ode using Euler method with event detection.", "[ode]") {
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

    // Set time span and options for Euler method with event detection
    TimeSpan<double> time_span(0.0, 10.0);
    Options<double, 2> options;
    options.fixed_time_step    = 1e-4;
    options.integration_method = IntegrationMethod::EULER;
    options.event.function     = [](const double t, const Eigen::Matrix<double, 2, 1>& x) {
        Eigen::Matrix<double, 1, 1> events;
        events(0) = x(0);  // Event when position x1 crosses zero
        return events;
    };
    options.event.direction   = 0;
    options.event.is_terminal = false;

    // Solve the ODE with event detection
    Result<double, 2> result = ode<double, 2>(ode_function, time_span, initial_conditions, options);

    // Check the computed event times against the expected event times
    double period      = 2.0 * M_PI;
    double half_period = period / 2.0;
    for (int i = 0; i < result.event_times.size(); ++i) {
        double expected_event_time = half_period * i + M_PI_2;
        CHECK(std::abs(result.event_times[i] - expected_event_time) < 1e-2);
    }
}

TEST_CASE("Solve bouncing ball ode using Euler and event detection.", "[ode]") {
    // Define the ODE function (free fall)
    auto ode_function = [](const double t, const Eigen::Matrix<double, 2, 1>& x) -> Eigen::Matrix<double, 2, 1> {
        Eigen::Matrix<double, 2, 1> dxdt;
        dxdt(0) = x(1);
        dxdt(1) = -9.81;  // Acceleration due to gravity
        return dxdt;
    };

    // Set initial conditions (initial height and zero initial velocity)
    Eigen::Matrix<double, 2, 1> initial_conditions;
    initial_conditions << 10.0, 0.0;

    // Set time span and options for the ODE solver with event detection
    TimeSpan<double> time_span(0.0, 10.0);
    Options<double, 2> options;
    options.fixed_time_step    = 1e-4;
    options.integration_method = IntegrationMethod::EULER;
    options.event.function     = [](const double t, const Eigen::Matrix<double, 2, 1>& x) {
        Eigen::Matrix<double, 1, 1> events;
        events(0) = x(0);  // Event when height crosses zero (ground)
        return events;
    };
    options.event.direction   = -1;  // Detect only when the height is decreasing
    options.event.is_terminal = true;

    // Solve the ODE with event detection
    Result<double, 2> result = ode<double, 2>(ode_function, time_span, initial_conditions, options);

    // Check if the event time is as expected (time when the ball reaches the ground)
    double expected_event_time = std::sqrt(2 * initial_conditions(0) / 9.81);
    CHECK(std::abs(result.event_times[0] - expected_event_time) < 1e-2);

    // Check if the height of the ball is as expected at the end of the simulation
    CHECK(std::abs(result.event_solutions[0](0)) < 1e-2);
}

TEST_CASE("Solve harmonic oscillator ode using RK4 method with event detection.", "[ode]") {
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

    // Set time span and options for RK4 method with event detection
    TimeSpan<double> time_span(0.0, 10.0);
    Options<double, 2> options;
    options.fixed_time_step    = 1e-4;
    options.integration_method = IntegrationMethod::RK4;
    options.event.function     = [](const double t, const Eigen::Matrix<double, 2, 1>& x) {
        Eigen::Matrix<double, 1, 1> events;
        events(0) = x(0);  // Event when position x1 crosses zero
        return events;
    };
    options.event.direction   = 0;
    options.event.is_terminal = false;

    // Solve the ODE with event detection
    Result<double, 2> result = ode<double, 2>(ode_function, time_span, initial_conditions, options);

    // Check the computed event times against the expected event times
    double period      = 2.0 * M_PI;
    double half_period = period / 2.0;
    for (int i = 0; i < result.event_times.size(); ++i) {
        double expected_event_time = half_period * i + M_PI_2;
        CHECK(std::abs(result.event_times[i] - expected_event_time) < 1e-2);
    }
}

TEST_CASE("Solve bouncing ball ode using RK4 and event detection.", "[ode]") {
    // Define the ODE function (free fall)
    auto ode_function = [](const double t, const Eigen::Matrix<double, 2, 1>& x) -> Eigen::Matrix<double, 2, 1> {
        Eigen::Matrix<double, 2, 1> dxdt;
        dxdt(0) = x(1);
        dxdt(1) = -9.81;  // Acceleration due to gravity
        return dxdt;
    };

    // Set initial conditions (initial height and zero initial velocity)
    Eigen::Matrix<double, 2, 1> initial_conditions;
    initial_conditions << 10.0, 0.0;

    // Set time span and options for the ODE solver with event detection
    TimeSpan<double> time_span(0.0, 10.0);
    Options<double, 2> options;
    options.fixed_time_step    = 1e-4;
    options.integration_method = IntegrationMethod::RK4;
    options.event.function     = [](const double t, const Eigen::Matrix<double, 2, 1>& x) {
        Eigen::Matrix<double, 1, 1> events;
        events(0) = x(0);  // Event when height crosses zero (ground)
        return events;
    };
    options.event.direction   = -1;  // Detect only when the height is decreasing
    options.event.is_terminal = true;

    // Solve the ODE with event detection
    Result<double, 2> result = ode<double, 2>(ode_function, time_span, initial_conditions, options);

    // Check if the event time is as expected (time when the ball reaches the ground)
    double expected_event_time = std::sqrt(2 * initial_conditions(0) / 9.81);
    CHECK(std::abs(result.event_times[0] - expected_event_time) < 1e-2);

    // Check if the height of the ball is as expected at the end of the simulation
    CHECK(std::abs(result.event_solutions[0](0)) < 1e-2);
}

TEST_CASE("Solve harmonic oscillator ode using RK5 method with event detection.", "[ode]") {
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

    // Set time span and options for RK5 method with event detection
    TimeSpan<double> time_span(0.0, 10.0);
    Options<double, 2> options;
    options.fixed_time_step    = 1e-4;
    options.integration_method = IntegrationMethod::RK5;
    options.event.function     = [](const double t, const Eigen::Matrix<double, 2, 1>& x) {
        Eigen::Matrix<double, 1, 1> events;
        events(0) = x(0);  // Event when position x1 crosses zero
        return events;
    };
    options.event.direction   = 0;
    options.event.is_terminal = false;

    // Solve the ODE with event detection
    Result<double, 2> result = ode<double, 2>(ode_function, time_span, initial_conditions, options);

    // Check the computed event times against the expected event times
    double period      = 2.0 * M_PI;
    double half_period = period / 2.0;
    for (int i = 0; i < result.event_times.size(); ++i) {
        double expected_event_time = half_period * i + M_PI_2;
        CHECK(std::abs(result.event_times[i] - expected_event_time) < 1e-2);
    }
}

TEST_CASE("Solve bouncing ball ode using RK5 and event detection.", "[ode]") {
    // Define the ODE function (free fall)
    auto ode_function = [](const double t, const Eigen::Matrix<double, 2, 1>& x) -> Eigen::Matrix<double, 2, 1> {
        Eigen::Matrix<double, 2, 1> dxdt;
        dxdt(0) = x(1);
        dxdt(1) = -9.81;  // Acceleration due to gravity
        return dxdt;
    };

    // Set initial conditions (initial height and zero initial velocity)
    Eigen::Matrix<double, 2, 1> initial_conditions;
    initial_conditions << 10.0, 0.0;

    // Set time span and options for the ODE solver with event detection
    TimeSpan<double> time_span(0.0, 10.0);
    Options<double, 2> options;
    options.fixed_time_step    = 1e-4;
    options.integration_method = IntegrationMethod::RK5;
    options.event.function     = [](const double t, const Eigen::Matrix<double, 2, 1>& x) {
        Eigen::Matrix<double, 1, 1> events;
        events(0) = x(0);  // Event when height crosses zero (ground)
        return events;
    };
    options.event.direction   = -1;  // Detect only when the height is decreasing
    options.event.is_terminal = true;

    // Solve the ODE with event detection
    Result<double, 2> result = ode<double, 2>(ode_function, time_span, initial_conditions, options);

    // Check if the event time is as expected (time when the ball reaches the ground)
    double expected_event_time = std::sqrt(2 * initial_conditions(0) / 9.81);
    CHECK(std::abs(result.event_times[0] - expected_event_time) < 1e-2);

    // Check if the height of the ball is as expected at the end of the simulation
    CHECK(std::abs(result.event_solutions[0](0)) < 1e-2);
}
