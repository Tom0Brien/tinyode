#ifndef TINYODE_ODE_HPP
#define TINYODE_ODE_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>

#include "euler.hpp"
#include "ode_base.hpp"
#include "rk4.hpp"
#include "rk45.hpp"
#include "rk5.hpp"

/** \file ode.hpp
 * @brief Contains various functions for solving ordinary differential equations (ODEs).
 */
namespace tinyode {

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
                case IntegrationMethod::RK4: return ode_rk4_event(ode_func, time_span, initial_conditions, options);
                case IntegrationMethod::RK5: return ode_rk5_event(ode_func, time_span, initial_conditions, options);
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