#ifndef TINYODE_ODEBASE_HPP
#define TINYODE_ODEBASE_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>

/** \file ode.hpp
 * @brief Contains various functions for solving ordinary differential equations (ODEs).
 */
namespace tinyode {

    /// @brief The types of integration methods for ode solver
    enum class IntegrationMethod { EULER, RK4, RK5, RK45 };

    /// @brief Struct for time span
    template <typename Scalar>
    struct TimeSpan {
        /// @brief Start time
        Scalar start;

        /// @brief End time
        Scalar end;

        /// @brief Constructor to initialize start and end times
        TimeSpan(Scalar start_time, Scalar end_time) : start(start_time), end(end_time) {}
    };

    // @brief Event structure for event detection
    template <typename Scalar, int n>
    struct Event {
        /// @brief Event function that defines the event occurs, which is when the function is zero.
        std::function<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>(const Scalar, const Eigen::Matrix<Scalar, n, 1>&)>
            function = nullptr;

        /// @brief True if the integration is to terminate at a zero of event_function.
        bool is_terminal = true;

        /// @brief 0 if all zeros are to be located (the default). A value of +1 locates only zeros where the event
        /// function is increasing, and -1 locates only zeros where the event function is decreasing.
        Scalar direction = 0;
    };

    /// @brief Struct for ODE options
    template <typename Scalar, int n>
    struct Options {
        /// @brief Fixed time step.
        Scalar fixed_time_step = 1e-3;

        /// @brief Simulation time span.
        TimeSpan<Scalar> time_span = TimeSpan<Scalar>(0, 1);

        /// @brief Integration method.
        IntegrationMethod integration_method = IntegrationMethod::RK4;

        /// @brief Relative tolerance for adaptive time step.
        Scalar relative_tolerance = 1e-3;

        /// @brief Absolute tolerance for adaptive time step.
        Scalar absolute_tolerance = 1e-3;

        /// @brief Event structure for event detection.
        Event<Scalar, n> event = {};
    };

    /// @brief Struct for ODE results
    template <typename Scalar, int n>
    struct Result {
        /// @brief Time values
        std::vector<Scalar> t = {};

        /// @brief Solution values
        std::vector<Eigen::Matrix<Scalar, n, 1>> x = {};

        /// @brief Event times
        std::vector<Scalar> event_times = {};

        /// @brief Event solutions
        std::vector<Eigen::Matrix<Scalar, n, 1>> event_solutions = {};

        /// @brief Event indices
        std::vector<int> event_indices = {};
    };

}  // namespace tinyode

#endif  // TINYODE_ODE_HPP