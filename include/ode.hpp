#ifndef TINYODE_ODE_HPP
#define TINYODE_ODE_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>

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
     * @brief Solve an ODE given an initial condition, options, and the ODE function using RK5 method
     * @param ode_function ODE function that defines the system of equations
     * @param time_span Time span for the ODE solver
     * @param initial_conditions Initial conditions for the ODE
     * @param options Options for the ODE solver, such as time step and method
     * @return A Result object containing the time values, solution values, event times, event solutions, and event
     * indices
     */
    template <typename Scalar, int n>
    Result<Scalar, n> ode_rk5(
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
        Eigen::Matrix<Scalar, n, 1> dxdt, k1, k2, k3, k4, k5, k6;

        // Loop through the time steps and apply the RK5 method to update the state variables
        for (int i = 1; i < num_time_steps; ++i) {
            // Compute the derivatives using the current state variable and time value
            dxdt = ode_function(t, x);

            // Compute the six stages of the RK5 method
            k1 = options.fixed_time_step * dxdt;
            k2 = options.fixed_time_step * ode_function(t + 0.25 * options.fixed_time_step, x + 0.25 * k1);
            k3 =
                options.fixed_time_step * ode_function(t + 0.25 * options.fixed_time_step, x + 0.125 * k1 + 0.125 * k2);
            k4 = options.fixed_time_step * ode_function(t + 0.5 * options.fixed_time_step, x - 0.5 * k2 + k3);
            k5 = options.fixed_time_step
                 * ode_function(t + 0.75 * options.fixed_time_step, x + 0.1875 * k1 + 0.5625 * k4);
            k6 =
                options.fixed_time_step
                * ode_function(t + options.fixed_time_step, x + (1.0 / 5.0) * k1 - (3.0 / 5.0) * k4 + (4.0 / 5.0) * k5);

            // Update the state variable using the RK5 method
            x.noalias() += (1.0 / 90.0) * (7.0 * k1 + 32.0 * k3 + 12.0 * k4 + 32.0 * k5 + 7.0 * k6);

            // Update the time variable
            t += options.fixed_time_step;

            // Store the computed state variables and corresponding time values in the Result structure
            result.t(i)     = t;
            result.x.col(i) = x;
        }

        return result;
    }

    /**
     * @brief Solve an ODE using the Runge-Kutta-Fehlberg 45 method.
     * @param ode_function ODE function that defines the system of equations
     * @param time_span Time span for the ODE solver
     * @param initial_conditions Initial conditions for the ODE
     * @param options Options for the ODE solver, such as time step and method
     * @return A Result object containing the time values, solution values, event times, event solutions, and event
     * indices
     */
    template <typename Scalar, int n>
    Result<Scalar, n> ode_rk45(
        const std::function<Eigen::Matrix<Scalar, n, 1>(const Scalar, const Eigen::Matrix<Scalar, n, 1>&)>&
            ode_function,
        const TimeSpan<Scalar>& time_span,
        const Eigen::Matrix<Scalar, n, 1>& initial_conditions,
        const Options<Scalar, n>& options) {
        // Initialize time and state variables
        Scalar t                      = time_span.start;
        Eigen::Matrix<Scalar, n, 1> x = initial_conditions;

        // Set initial time step
        Scalar h = options.fixed_time_step;

        // Set relative and absolute tolerances
        Scalar rtol = options.relative_tolerance;
        Scalar atol = options.absolute_tolerance;

        // Initialize result structure
        Result<Scalar, n> result;
        result.t.resize(1, 1);
        result.x.resize(n, 1);
        result.t(0)     = t;
        result.x.col(0) = x;

        // Initialize derivative matrices
        Eigen::Matrix<Scalar, n, 1> k1, k2, k3, k4, k5, k6;

        // Loop through time steps
        while (t < time_span.end) {
            // Compute k1-k6
            k1 = ode_function(t, x);
            k2 = ode_function(t + h * (1.0 / 4.0), x + h * (1.0 / 4.0) * k1);
            k3 = ode_function(t + h * (3.0 / 8.0), x + h * (3.0 / 32.0) * k1 + h * (9.0 / 32.0) * k2);
            k4 = ode_function(t + h * (12.0 / 13.0),
                              x + h * (1932.0 / 2197.0) * k1 - h * (7200.0 / 2197.0) * k2 + h * (7296.0 / 2197.0) * k3);
            k5 = ode_function(
                t + h,
                x + h * (439.0 / 216.0) * k1 - h * 8.0 * k2 + h * (3680.0 / 513.0) * k3 - h * (845.0 / 4104.0) * k4);
            k6 = ode_function(t + h * (1.0 / 2.0),
                              x - h * (8.0 / 27.0) * k1 + h * 2.0 * k2 - h * (3544.0 / 2565.0) * k3
                                  + h * (1859.0 / 4104.0) * k4 - h * (11.0 / 40.0) * k5);
            // Compute the 4th and 5th order approximations
            Eigen::Matrix<Scalar, n, 1> x4 = x + h * (25.0 / 216.0) * k1 + h * (1408.0 / 2565.0) * k3
                                             + h * (2197.0 / 4104.0) * k4 - h * (1.0 / 5.0) * k5;
            Eigen::Matrix<Scalar, n, 1> x5 = x + h * (16.0 / 135.0) * k1 + h * (6656.0 / 12825.0) * k3
                                             + h * (28561.0 / 56430.0) * k4 - h * (9.0 / 50.0) * k5
                                             + h * (2.0 / 55.0) * k6;

            // Compute the error estimate
            Eigen::Matrix<Scalar, n, 1> error = x5 - x4;
            Scalar error_norm                 = error.norm();

            // Compute the scale factor for the tolerances
            Eigen::Matrix<Scalar, n, 1> scale =
                atol * Eigen::Matrix<Scalar, n, 1>::Ones() + rtol * (x.cwiseAbs().array().max(1).matrix());

            // Check if error is within tolerance
            if (error_norm < scale.norm()) {
                // Update time and state variables
                t += h;
                x = x5;

                // Store the computed time and state variables in the result structure
                result.t.conservativeResize(result.t.rows(), result.t.cols() + 1);
                result.t(result.t.size() - 1) = t;

                result.x.conservativeResize(result.x.rows(), result.x.cols() + 1);
                result.x.col(result.x.cols() - 1) = x;
            }

            // Adjust time step based on error estimate and tolerance
            Scalar delta = 0.84 * std::pow(scale.norm() / error_norm, 0.25);
            delta        = std::max(delta, 0.1);
            delta        = std::min(delta, 4.0);
            h *= delta;
        }

        return result;
    }

    /**
     * @brief Solve an ODE given an initial condition, options, and the ODE function
     * @param ode_function ODE function that defines the system of equations
     * @param time_span Time span for the ODE solver
     * @param initial_conditions Initial conditions for the ODE
     * @param options Options for the ODE solver, such as time step and method
     * @return A Result object containing the time values, solution values, event times, event solutions, and
     * event indices
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
                case IntegrationMethod::RK5: return ode_rk5(ode_function, time_span, initial_conditions, options);
                case IntegrationMethod::RK45: return ode_rk45(ode_function, time_span, initial_conditions, options);
                default: throw std::invalid_argument("Invalid integration method.");
            }
        }
    }

}  // namespace tinyode

#endif  // TINYODE_ODE_HPP