#ifndef TINYODE_ODE_HPP
#define TINYODE_ODE_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>

/** \file ode.hpp
 * @brief Contains various functions for solving ordinary differential equations (ODEs).
 */
namespace tinyode {

    /// @brief The types of integration methods for ode solver
    enum class IntegrationMethod { EULER, RK4 };

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
        std::function<Eigen::Matrix<Scalar, n, 1>(const Scalar, const Eigen::Matrix<Scalar, n, 1>&)> function = nullptr;

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

        /// @brief Event structure for event detection.
        Event<Scalar, n> event = {};
    };

    /// @brief Struct for ODE results
    template <typename Scalar, int n>
    struct Result {
        /// @brief Time values
        Eigen::Matrix<Scalar, 1, Eigen::Dynamic> t = {};

        /// @brief Solution values
        Eigen::Matrix<Scalar, n, Eigen::Dynamic> x = {};

        /// @brief Event times
        Eigen::Matrix<Scalar, 1, Eigen::Dynamic> event_times = {};

        /// @brief Event solutions
        Eigen::Matrix<Scalar, n, Eigen::Dynamic> event_solutions = {};

        /// @brief Event indices
        std::vector<int> event_indices = {};
    };

    /**
     * @brief Solve an ODE given an initial condition, options, and the ODE function using Euler method
     * @param ode_function ODE function that defines the system of equations
     * @param time_span Time span for the ODE solver
     * @param initial_conditions Initial conditions for the ODE
     * @param options Options for the ODE solver, such as time step and method
     * @return A result object containing the time values, solution values, event times, event solutions, and event
     * indices
     */
    template <typename Scalar, int n>
    Result<Scalar, n> ode_euler(
        const std::function<Eigen::Matrix<Scalar, n, 1>(const Scalar, const Eigen::Matrix<Scalar, n, 1>&)>&
            ode_function,
        const TimeSpan<Scalar>& time_span,
        const Eigen::Matrix<Scalar, n, 1>& initial_conditions,
        const Options<Scalar, n>& options) {

        // Calculate the number of time steps
        int num_time_steps = static_cast<int>((time_span.end - time_span.start) / options.fixed_time_step) + 1;

        // Resize the matrices in the Result structure
        Result<Scalar, n> result;
        result.t.resize(1, num_time_steps);
        result.x.resize(n, num_time_steps);

        // Initialize the time and state variables with the initial conditions
        Scalar t                      = time_span.start;
        Eigen::Matrix<Scalar, n, 1> x = initial_conditions;

        // Store the initial conditions in the Result structure
        result.t(0)     = t;
        result.x.col(0) = x;

        // Preallocate the derivative matrix
        Eigen::Matrix<Scalar, n, 1> dxdt;

        // Loop through the time steps and apply the Euler method to update the state variables
        for (int i = 1; i < num_time_steps; ++i) {
            // Compute the derivative
            dxdt = ode_function(t, x);

            // Update the state variable using the Euler method
            x.noalias() += options.fixed_time_step * dxdt;

            // Update the time variable
            t += options.fixed_time_step;

            // Store the computed state variables and corresponding time values in the Result structure
            result.t(i)     = t;
            result.x.col(i) = x;
        }

        return result;
    }

    /**
     * @brief Solve an ODE given an initial condition, options, and the ODE function using RK4 method
     * @param ode_function ODE function that defines the system of equations
     * @param time_span Time span for the ODE solver
     * @param initial_conditions Initial conditions for the ODE
     * @param options Options for the ODE solver, such as time step and method
     * @return A Result object containing the time values, solution values, event times, event solutions, and event
     * indices
     */
    template <typename Scalar, int n>
    Result<Scalar, n> ode_rk4(
        const std::function<Eigen::Matrix<Scalar, n, 1>(const Scalar, const Eigen::Matrix<Scalar, n, 1>&)>&
            ode_function,
        const TimeSpan<Scalar>& time_span,
        const Eigen::Matrix<Scalar, n, 1>& initial_conditions,
        const Options<Scalar, n>& options) {
        // Calculate the number of time steps
        int num_time_steps = static_cast<int>((time_span.end - time_span.start) / options.fixed_time_step) + 1;

        // Resize the matrices in the Result structure
        Result<Scalar, n> result;
        result.t.resize(1, num_time_steps);
        result.x.resize(n, num_time_steps);

        // Initialize the time and state variables with the initial conditions
        Scalar t                      = time_span.start;
        Eigen::Matrix<Scalar, n, 1> x = initial_conditions;

        // Store the initial conditions in the Result structure
        result.t(0)     = t;
        result.x.col(0) = x;

        // Preallocate the derivative matrices
        Eigen::Matrix<Scalar, n, 1> dxdt, k1, k2, k3, k4;

        // Loop through the time steps and apply the RK4 method to update the state variables
        for (int i = 1; i < num_time_steps; ++i) {
            // Compute the derivatives using the current state variable and time value
            dxdt = ode_function(t, x);

            // Compute the four stages of the RK4 method
            k1 = options.fixed_time_step * dxdt;
            k2 = options.fixed_time_step * ode_function(t + 0.5 * options.fixed_time_step, x + 0.5 * k1);
            k3 = options.fixed_time_step * ode_function(t + 0.5 * options.fixed_time_step, x + 0.5 * k2);
            k4 = options.fixed_time_step * ode_function(t + options.fixed_time_step, x + k3);

            // Update the state variable using the RK4 method
            x.noalias() += (1.0 / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);

            // Update the time variable
            t += options.fixed_time_step;

            // Store the computed state variables and corresponding time values in the Result structure
            result.t(i)     = t;
            result.x.col(i) = x;
        }

        return result;
    }

    /**
     * @brief Solve an ODE given an initial condition, options, and the ODE function
     * @param ode_function ODE function that defines the system of equations
     * @param time_span Time span for the ODE solver
     * @param initial_conditions Initial conditions for the ODE
     * @param options Options for the ODE solver, such as time step and method
     * @return A Result object containing the time values, solution values, event times, event solutions, and event
     * indices
     */
    template <typename Scalar, int n>
    Result<Scalar, n> ode(
        const std::function<Eigen::Matrix<Scalar, n, 1>(const Scalar, const Eigen::Matrix<Scalar, n, 1>&)>&
            ode_function,
        const TimeSpan<Scalar>& time_span,
        const Eigen::Matrix<Scalar, n, 1>& initial_conditions,
        const Options<Scalar, n>& options = Options<Scalar, n>()) {

        if (options.event.function) {
            // Call event-based solvers if an event function is provided
            // switch (options.integration_method) {
            //     case IntegrationMethod::EULER:
            //         return ode_euler_events(ode_function, time_span, initial_conditions, options);
            //     case IntegrationMethod::RK4:
            //         return ode_rk4_events(ode_function, time_span, initial_conditions, options);
            //     default: throw std::invalid_argument("Invalid integration method.");
            // }
        }
        else {
            // Call regular solvers if no event function is provided
            switch (options.integration_method) {
                case IntegrationMethod::EULER: return ode_euler(ode_function, time_span, initial_conditions, options);
                case IntegrationMethod::RK4: return ode_rk4(ode_function, time_span, initial_conditions, options);
                default: throw std::invalid_argument("Invalid integration method.");
            }
        }
    }

}  // namespace tinyode

#endif  // TINYODE_ODE_HPP