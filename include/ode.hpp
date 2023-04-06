#ifndef TINYODE_ODE_HPP
#define TINYODE_ODE_HPP

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

    /**
     * @brief Solve an ODE given an initial condition, options, and the ODE function using Euler method
     * @param ode_func ODE function that defines the system of equations
     * @param time_span Time span for the ODE solver
     * @param initial_conditions Initial conditions for the ODE
     * @param options Options for the ODE solver, such as time step and method
     * @return A result object containing the time values, solution values
     */
    template <typename Scalar, int n>
    Result<Scalar, n> ode_euler(
        const std::function<Eigen::Matrix<Scalar, n, 1>(const Scalar, const Eigen::Matrix<Scalar, n, 1>&)>& ode_func,
        const TimeSpan<Scalar>& time_span,
        const Eigen::Matrix<Scalar, n, 1>& initial_conditions,
        const Options<Scalar, n>& options) {

        // Calculate the number of time steps
        int num_time_steps = static_cast<int>((time_span.end - time_span.start) / options.fixed_time_step) + 1;

        // Create a Result structure
        Result<Scalar, n> result;

        // Initialize the time and state variables with the initial conditions
        Scalar t                      = time_span.start;
        Eigen::Matrix<Scalar, n, 1> x = initial_conditions;

        // Store the initial conditions in the Result structure
        result.t.push_back(t);
        result.x.push_back(x);

        // Loop through the time steps and apply the Euler method to update the state variables
        for (int i = 1; i < num_time_steps; ++i) {
            // Update the state variable using the Euler method
            x.noalias() += options.fixed_time_step * ode_func(t, x);

            // Update the time variable
            t += options.fixed_time_step;

            // Store the computed state variables and corresponding time values in the Result structure
            result.t.push_back(t);
            result.x.push_back(x);
        }

        return result;
    }

    /**
     * @brief Solve an ODE given an initial condition, options, and the ODE function using Euler method with event
     * detection
     * @param ode_func ODE function that defines the system of equations
     * @param time_span Time span for the ODE solver
     * @param initial_conditions Initial conditions for the ODE
     * @param options Options for the ODE solver, such as time step and method
     * @return A result object containing the time values, solution values, event times, event solutions, and event
     * indices
     */
    template <typename Scalar, int n>
    Result<Scalar, n> ode_euler_event(
        const std::function<Eigen::Matrix<Scalar, n, 1>(const Scalar, const Eigen::Matrix<Scalar, n, 1>&)>& ode_func,
        const TimeSpan<Scalar>& time_span,
        const Eigen::Matrix<Scalar, n, 1>& initial_conditions,
        const Options<Scalar, n>& options) {

        // Calculate the number of time steps
        int num_time_steps = static_cast<int>((time_span.end - time_span.start) / options.fixed_time_step) + 1;

        // Create a Result structure
        Result<Scalar, n> result;

        // Initialize the time and state variables with the initial conditions
        Scalar t                           = time_span.start;
        Scalar prev_t                      = t;
        Eigen::Matrix<Scalar, n, 1> x      = initial_conditions;
        Eigen::Matrix<Scalar, n, 1> prev_x = x;

        // Store the initial conditions in the Result structure
        result.t.push_back(t);
        result.x.push_back(x);

        // Loop through the time steps and apply the Euler method to update the state variables
        for (int i = 1; i < num_time_steps; ++i) {
            // Store the previous state and time
            prev_x = x;
            prev_t = t;

            // Update the state variable using the Euler method
            x.noalias() += options.fixed_time_step * ode_func(t, x);

            // Update the time variable
            t += options.fixed_time_step;

            // Check for events
            Eigen::Matrix<Scalar, Eigen::Dynamic, 1> prev_event_values = options.event.function(prev_t, prev_x);
            Eigen::Matrix<Scalar, Eigen::Dynamic, 1> curr_event_values = options.event.function(t, x);
            int num_events                                             = prev_event_values.size();

            for (int j = 0; j < num_events; ++j) {
                Scalar prev_event_value = prev_event_values(j);
                Scalar curr_event_value = curr_event_values(j);

                if ((options.event.direction == 0)
                    || (options.event.direction > 0 && prev_event_value < curr_event_value)
                    || (options.event.direction < 0 && prev_event_value > curr_event_value)) {
                    if (prev_event_value * curr_event_value <= 0) {
                        // An event has occurred
                        result.event_times.push_back(t);
                        result.event_solutions.push_back(x);
                        result.event_indices.push_back(j);

                        // If event is terminal, stop the integration
                        if (options.event.is_terminal) {
                            return result;
                        }
                    }
                }
            }

            // Store the computed state variables and corresponding time values in the Result structure
            result.t.push_back(t);
            result.x.push_back(x);
        }

        return result;
    }


    /**
     * @brief Solve an ODE given an initial condition, options, and the ODE function using RK4 method
     * @param ode_func ODE function that defines the system of equations
     * @param time_span Time span for the ODE solver
     * @param initial_conditions Initial conditions for the ODE
     * @param options Options for the ODE solver, such as time step and method
     * @return A Result object containing the time values, solution values
     */
    template <typename Scalar, int n>
    Result<Scalar, n> ode_rk4(
        const std::function<Eigen::Matrix<Scalar, n, 1>(const Scalar, const Eigen::Matrix<Scalar, n, 1>&)>& ode_func,
        const TimeSpan<Scalar>& time_span,
        const Eigen::Matrix<Scalar, n, 1>& initial_conditions,
        const Options<Scalar, n>& options) {
        // Calculate the number of time steps
        int num_time_steps = static_cast<int>((time_span.end - time_span.start) / options.fixed_time_step) + 1;

        // Create a Result structure
        Result<Scalar, n> result;

        // Initialize the time and state variables with the initial conditions
        Scalar t                      = time_span.start;
        Eigen::Matrix<Scalar, n, 1> x = initial_conditions;

        // Store the initial conditions in the Result structure
        result.t.push_back(t);
        result.x.push_back(x);

        // Preallocate the derivative matrices
        Eigen::Matrix<Scalar, n, 1> k1, k2, k3, k4;

        // Loop through the time steps and apply the RK4 method to update the state variables
        for (int i = 1; i < num_time_steps; ++i) {

            // Compute the four stages of the RK4 method
            k1 = options.fixed_time_step * ode_func(t, x);
            k2 = options.fixed_time_step * ode_func(t + 0.5 * options.fixed_time_step, x + 0.5 * k1);
            k3 = options.fixed_time_step * ode_func(t + 0.5 * options.fixed_time_step, x + 0.5 * k2);
            k4 = options.fixed_time_step * ode_func(t + options.fixed_time_step, x + k3);

            // Update the state variable using the RK4 method
            x.noalias() += (1.0 / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);

            // Update the time variable
            t += options.fixed_time_step;

            // Store the computed state variables and corresponding time values in the Result structure
            result.t.push_back(t);
            result.x.push_back(x);
        }

        return result;
    }

    /**
     * @brief Solve an ODE given an initial condition, options, and the ODE function using RK5 method
     * @param ode_func ODE function that defines the system of equations
     * @param time_span Time span for the ODE solver
     * @param initial_conditions Initial conditions for the ODE
     * @param options Options for the ODE solver, such as time step and method
     * @return A Result object containing the time values, solution values
     */
    template <typename Scalar, int n>
    Result<Scalar, n> ode_rk5(
        const std::function<Eigen::Matrix<Scalar, n, 1>(const Scalar, const Eigen::Matrix<Scalar, n, 1>&)>& ode_func,
        const TimeSpan<Scalar>& time_span,
        const Eigen::Matrix<Scalar, n, 1>& initial_conditions,
        const Options<Scalar, n>& options) {
        // Calculate the number of time steps
        int num_time_steps = static_cast<int>((time_span.end - time_span.start) / options.fixed_time_step) + 1;

        // Create a Result structure
        Result<Scalar, n> result;

        // Initialize the time and state variables with the initial conditions
        Scalar t                      = time_span.start;
        Eigen::Matrix<Scalar, n, 1> x = initial_conditions;

        // Store the initial conditions in the Result structure
        result.t.push_back(t);
        result.x.push_back(x);

        // Preallocate the derivative matrices
        Eigen::Matrix<Scalar, n, 1> k1, k2, k3, k4, k5, k6;

        // Loop through the time steps and apply the RK5 method to update the state variables
        for (int i = 1; i < num_time_steps; ++i) {
            // Compute the six stages of the RK5 method
            k1 = options.fixed_time_step * ode_func(t, x);
            k2 = options.fixed_time_step * ode_func(t + 0.25 * options.fixed_time_step, x + 0.25 * k1);
            k3 = options.fixed_time_step * ode_func(t + 0.25 * options.fixed_time_step, x + 0.125 * k1 + 0.125 * k2);
            k4 = options.fixed_time_step * ode_func(t + 0.5 * options.fixed_time_step, x - 0.5 * k2 + k3);
            k5 = options.fixed_time_step * ode_func(t + 0.75 * options.fixed_time_step, x + 0.1875 * k1 + 0.5625 * k4);
            k6 = options.fixed_time_step
                 * ode_func(t + options.fixed_time_step, x + (1.0 / 5.0) * k1 - (3.0 / 5.0) * k4 + (4.0 / 5.0) * k5);

            // Update the state variable using the RK5 method
            x.noalias() += (1.0 / 90.0) * (7.0 * k1 + 32.0 * k3 + 12.0 * k4 + 32.0 * k5 + 7.0 * k6);

            // Update the time variable
            t += options.fixed_time_step;

            // Store the computed state variables and corresponding time values in the Result structure
            result.t.push_back(t);
            result.x.push_back(x);
        }

        return result;
    }

    /**
     * @brief Solve an ODE using the Runge-Kutta-Fehlberg 45 method.
     * @param ode_func ODE function that defines the system of equations
     * @param time_span Time span for the ODE solver
     * @param initial_conditions Initial conditions for the ODE
     * @param options Options for the ODE solver, such as time step and method
     * @return A Result object containing the time values, solution values
     */
    template <typename Scalar, int n>
    Result<Scalar, n> ode_rk45(
        const std::function<Eigen::Matrix<Scalar, n, 1>(const Scalar, const Eigen::Matrix<Scalar, n, 1>&)>& ode_func,
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
        result.t.push_back(t);
        result.x.push_back(x);

        // Initialize derivative matrices
        Eigen::Matrix<Scalar, n, 1> k1, k2, k3, k4, k5, k6;

        // Loop through time steps
        while (t < time_span.end) {
            // Compute k1-k6
            k1 = ode_func(t, x);
            k2 = ode_func(t + h * (1.0 / 4.0), x + h * (1.0 / 4.0) * k1);
            k3 = ode_func(t + h * (3.0 / 8.0), x + h * (3.0 / 32.0) * k1 + h * (9.0 / 32.0) * k2);
            k4 = ode_func(t + h * (12.0 / 13.0),
                          x + h * (1932.0 / 2197.0) * k1 - h * (7200.0 / 2197.0) * k2 + h * (7296.0 / 2197.0) * k3);
            k5 = ode_func(
                t + h,
                x + h * (439.0 / 216.0) * k1 - h * 8.0 * k2 + h * (3680.0 / 513.0) * k3 - h * (845.0 / 4104.0) * k4);
            k6 = ode_func(t + h * (1.0 / 2.0),
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
                result.t.push_back(t);
                result.x.push_back(x);
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
     * @param ode_func ODE function that defines the system of equations
     * @param time_span Time span for the ODE solver
     * @param initial_conditions Initial conditions for the ODE
     * @param options Options for the ODE solver, such as time step and method
     * @return A Result object containing the time values, solution values, event times, event solutions, and
     * event indices
     */
    template <typename Scalar, int n>
    Result<Scalar, n> ode(
        const std::function<Eigen::Matrix<Scalar, n, 1>(const Scalar, const Eigen::Matrix<Scalar, n, 1>&)>& ode_func,
        const TimeSpan<Scalar>& time_span,
        const Eigen::Matrix<Scalar, n, 1>& initial_conditions,
        const Options<Scalar, n>& options = Options<Scalar, n>()) {
        if (options.event.function) {
            // Call event-based solvers if an event function is provided
            switch (options.integration_method) {
                case IntegrationMethod::EULER: return ode_euler_event(ode_func, time_span, initial_conditions, options);
                default: throw std::invalid_argument("Invalid integration method.");
            }
        }
        else {
            // Call regular solvers if no event function is provided
            switch (options.integration_method) {
                case IntegrationMethod::EULER: return ode_euler(ode_func, time_span, initial_conditions, options);
                case IntegrationMethod::RK4: return ode_rk4(ode_func, time_span, initial_conditions, options);
                case IntegrationMethod::RK5: return ode_rk5(ode_func, time_span, initial_conditions, options);
                case IntegrationMethod::RK45: return ode_rk45(ode_func, time_span, initial_conditions, options);
                default: throw std::invalid_argument("Invalid integration method.");
            }
        }
    }

}  // namespace tinyode

#endif  // TINYODE_ODE_HPP