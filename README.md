tinyode
===========

**tinyode** is a lightweight C++ library which provides various ordinary differential equation solver.

The goal of **tinyode** is to be as simple as possible while still being incredibly fast and versatile.

## Features

- Solver Methods:
  - Euler
  - RK4
  - RK5
  - RK45

- Event Detection


## Install

### 1. Install Dependencies
- [Eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page)
- [Catch2](https://github.com/catchorg/Catch2)

### 2. Build with cmake
  ```bash
  git clone https://github.com/Tom0Brien/tinyode.git && cd tinyode
  mkdir build && cd build
  cmake ..
  make
  sudo make install # copies files in the include folder to /usr/local/include*
  ```

## Examples
Numerous examples are provided in the `examples` folder. 

The code below demonstrates how to
```c++
    // Define the ODE function f(t,x)
    auto ode_function = [](const double t, const Eigen::Matrix<double, 1, 1>& x) -> Eigen::Matrix<double, 1, 1> {
        return Eigen::Matrix<double, 1, 1>(-x(0, 0));
    };

    // Specify the initial conditions, time span and solver options
    Eigen::Matrix<double, 1, 1> initial_conditions(1.0);
    TimeSpan<double> time_span(0.0, 1.0);
    Options<double, 1> options;
    options.fixed_time_step    = 0.01;
    options.integration_method = IntegrationMethod::EULER;

    // Solve the ODE!
    auto result = ode(ode_function, time_span, initial_conditions, options);
```
