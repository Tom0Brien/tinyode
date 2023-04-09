#ifndef TINYODE_EULER_HPP
#define TINYODE_EULER_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>

#include "ode_base.hpp"

/** \file euler.hpp
 * @brief Contains euler method for solving ordinary differential equations (ODEs).
 */
namespace tinyode {

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
}  // namespace tinyode
#endif