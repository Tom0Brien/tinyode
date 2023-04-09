#ifndef TINYODE_RK45_HPP
#define TINYODE_RK45_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>

#include "ode_base.hpp"

/** \file rk45.hpp
 * @brief Contains rk45 method for solving ordinary differential equations (ODEs).
 */
namespace tinyode {

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

}  // namespace tinyode
#endif