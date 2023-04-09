#ifndef TINYODE_EVENTS_HPP
#define TINYODE_EVENTS_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>

#include "ode_base.hpp"

/** \file rk45.hpp
 * @brief Contains rk45 method for solving ordinary differential equations (ODEs).
 */
namespace tinyode {

    /**
     * @brief Check if an event has occurred
     * @param options Options for the ODE solver, such as event detection function, direction, and terminal flag
     * @param prev_t Previous time value
     * @param prev_x Previous state value
     * @param curr_t Current time value
     * @param curr_x Current state value
     * @param result Result object that will be updated with event information if an event is detected
     * @return A boolean indicating whether an event has occurred and is terminal
     */
    template <typename Scalar, int n>
    bool detect_event(const Options<Scalar, n>& options,
                      const Scalar& prev_t,
                      const Eigen::Matrix<Scalar, n, 1>& prev_x,
                      const Scalar& curr_t,
                      const Eigen::Matrix<Scalar, n, 1>& curr_x,
                      Result<Scalar, n>& result) {

        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> prev_event_values = options.event.function(prev_t, prev_x);
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> curr_event_values = options.event.function(curr_t, curr_x);
        int num_events                                             = prev_event_values.size();

        bool event_detected = false;
        for (int j = 0; j < num_events; ++j) {
            Scalar prev_event_value = prev_event_values(j);
            Scalar curr_event_value = curr_event_values(j);

            if ((options.event.direction == 0) || (options.event.direction > 0 && prev_event_value < curr_event_value)
                || (options.event.direction < 0 && prev_event_value > curr_event_value)) {
                if (prev_event_value * curr_event_value <= 0) {
                    // An event has occurred
                    result.event_times.push_back(curr_t);
                    result.event_solutions.push_back(curr_x);
                    result.event_indices.push_back(j);

                    event_detected = true;

                    // If the event is terminal, return true to indicate that event detection should stop
                    if (options.event.is_terminal) {
                        return true;
                    }
                }
            }
        }
        return event_detected;
    }

}  // namespace tinyode
#endif