#ifndef TINYODE_EULER_HPP
#define TINYODE_EULER_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>

#include "events.hpp"
#include "ode_base.hpp"

/** \file euler.hpp
 * @brief Contains euler method for solving ordinary differential equations (ODEs).
 */
namespace tinyode {

    /**
     * @brief Perform a single Euler step
     * @param ode_func ODE function that defines the system of equations
     * @param t time variable
     * @param x state variable
     * @param options Options for the ODE solver, such as time step and method
     * @tparam Scalar Scalar type of the ODE
     * @tparam n State dimensions
     * @return Eigen::Matrix<Scalar, n, 1>
     */
    template <typename Scalar, int n>
    Eigen::Matrix<Scalar, n, 1> euler_step(
        const std::function<Eigen::Matrix<Scalar, n, 1>(const Scalar, const Eigen::Matrix<Scalar, n, 1>&)>& ode_func,
        const Scalar t,
        const Eigen::Matrix<Scalar, n, 1>& x,
        const Options<Scalar, n>& options) {
        // Update the state variable using the Euler method
        return x + options.fixed_time_step * ode_func(t, x);
    }

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
            x.noalias() = euler_step(ode_func, t, x, options);

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
            x.noalias() = euler_step(ode_func, t, x, options);

            // Update the time variable
            t += options.fixed_time_step;

            // Check for events
            bool event_detected = detect_event(options, prev_t, prev_x, t, x, result);
            if (event_detected && options.event.is_terminal) {
                return result;
            }

            // Store the computed state variables and corresponding time values in the Result structure
            result.t.push_back(t);
            result.x.push_back(x);
        }

        return result;
    }
}  // namespace tinyode
#endif