#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

constexpr double kPi = 3.14159265358979323846;

double deg2rad(double degrees) {
    return degrees * kPi / 180.0;
}

using Complex = std::complex<double>;

std::vector<double> centered_positions(int count, double spacing_lambda) {
    if (count < 1) {
        throw std::invalid_argument("element count must be >= 1");
    }

    std::vector<double> positions(static_cast<std::size_t>(count));
    const double center = static_cast<double>(count - 1) / 2.0;
    for (int i = 0; i < count; ++i) {
        positions[static_cast<std::size_t>(i)] = (static_cast<double>(i) - center) * spacing_lambda;
    }
    return positions;
}

std::vector<double> uniform_window(int count) {
    return std::vector<double>(static_cast<std::size_t>(count), 1.0);
}

std::vector<double> hamming_window(int count) {
    std::vector<double> weights(static_cast<std::size_t>(count), 1.0);
    if (count == 1) {
        return weights;
    }

    for (int n = 0; n < count; ++n) {
        weights[static_cast<std::size_t>(n)] =
            0.54 - 0.46 * std::cos((2.0 * kPi * static_cast<double>(n)) / static_cast<double>(count - 1));
    }
    return weights;
}

std::vector<double> taylor_window(int count, int nbar = 4, double sll_db = 30.0) {
    std::vector<double> weights(static_cast<std::size_t>(count), 1.0);
    if (count == 1) {
        return weights;
    }

    const double a = std::acosh(std::pow(10.0, sll_db / 20.0)) / kPi;
    const double sigma2 = (static_cast<double>(nbar) * static_cast<double>(nbar)) /
                          (a * a + std::pow(static_cast<double>(nbar) - 0.5, 2.0));

    std::vector<double> fm(static_cast<std::size_t>(nbar), 0.0);
    for (int m = 1; m < nbar; ++m) {
        double numerator = 1.0;
        double denominator = 1.0;
        for (int n = 1; n < nbar; ++n) {
            numerator *= 1.0 - (static_cast<double>(m * m) / (sigma2 * (a * a + std::pow(static_cast<double>(n) - 0.5, 2.0))));
            if (n != m) {
                denominator *= 1.0 - (static_cast<double>(m * m) / static_cast<double>(n * n));
            }
        }
        fm[static_cast<std::size_t>(m)] = ((m % 2) == 0 ? 1.0 : -1.0) * numerator / (2.0 * denominator);
    }

    const double midpoint = static_cast<double>(count - 1) / 2.0;
    for (int n = 0; n < count; ++n) {
        const double x = static_cast<double>(n) - midpoint;
        double value = 1.0;
        for (int m = 1; m < nbar; ++m) {
            value += 2.0 * fm[static_cast<std::size_t>(m)] * std::cos((2.0 * kPi * static_cast<double>(m) * x) / static_cast<double>(count));
        }
        weights[static_cast<std::size_t>(n)] = value;
    }
    return weights;
}

std::vector<double> normalize_real(std::vector<double> values) {
    const double max_value = *std::max_element(values.begin(), values.end());
    if (max_value <= 0.0) {
        return values;
    }

    for (double &value : values) {
        value /= max_value;
    }
    return values;
}

std::vector<Complex> normalize_complex_by_max_abs(std::vector<Complex> values) {
    double max_value = 0.0;
    for (const Complex &value : values) {
        max_value = std::max(max_value, std::abs(value));
    }
    if (max_value <= 0.0) {
        return values;
    }

    for (Complex &value : values) {
        value /= max_value;
    }
    return values;
}

std::vector<double> amplitude_taper_impl(int count, const std::string &taper_name) {
    std::vector<double> weights;
    if (taper_name == "uniform") {
        weights = uniform_window(count);
    } else if (taper_name == "hamming") {
        weights = hamming_window(count);
    } else if (taper_name == "taylor") {
        weights = taylor_window(count);
    } else {
        throw std::invalid_argument("Unsupported taper: " + taper_name);
    }

    return normalize_real(std::move(weights));
}

double direction_cosine_x(double theta_deg, double phi_deg) {
    return std::sin(deg2rad(theta_deg)) * std::cos(deg2rad(phi_deg));
}

double direction_cosine_y(double theta_deg, double phi_deg) {
    return std::sin(deg2rad(theta_deg)) * std::sin(deg2rad(phi_deg));
}

py::array_t<double> vector_to_numpy(const std::vector<double> &values) {
    py::array_t<double> array(values.size());
    auto view = array.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < view.shape(0); ++i) {
        view(i) = values[static_cast<std::size_t>(i)];
    }
    return array;
}

py::array_t<Complex> complex_vector_to_numpy(const std::vector<Complex> &values) {
    py::array_t<Complex> array(values.size());
    auto view = array.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < view.shape(0); ++i) {
        view(i) = values[static_cast<std::size_t>(i)];
    }
    return array;
}

py::array_t<double> pair_vectors_to_numpy(const std::vector<double> &x, const std::vector<double> &y) {
    py::array_t<double> array({static_cast<py::ssize_t>(x.size()), static_cast<py::ssize_t>(2)});
    auto view = array.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < view.shape(0); ++i) {
        view(i, 0) = x[static_cast<std::size_t>(i)];
        view(i, 1) = y[static_cast<std::size_t>(i)];
    }
    return array;
}

std::vector<Complex> linear_manifold_vector(
    int count,
    double spacing_lambda,
    double theta_deg,
    double phi_deg
) {
    const auto positions = centered_positions(count, spacing_lambda);
    const double ux = direction_cosine_x(theta_deg, phi_deg);

    std::vector<Complex> manifold(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        const double phase = 2.0 * kPi * positions[static_cast<std::size_t>(i)] * ux;
        manifold[static_cast<std::size_t>(i)] = std::exp(Complex(0.0, phase));
    }
    return manifold;
}

std::pair<std::vector<double>, std::vector<double>> planar_positions_impl(
    int num_x,
    int num_y,
    double spacing_x_lambda,
    double spacing_y_lambda
) {
    const auto x_positions = centered_positions(num_x, spacing_x_lambda);
    const auto y_positions = centered_positions(num_y, spacing_y_lambda);

    std::vector<double> x_flat;
    std::vector<double> y_flat;
    x_flat.reserve(static_cast<std::size_t>(num_x * num_y));
    y_flat.reserve(static_cast<std::size_t>(num_x * num_y));

    for (int iy = 0; iy < num_y; ++iy) {
        for (int ix = 0; ix < num_x; ++ix) {
            x_flat.push_back(x_positions[static_cast<std::size_t>(ix)]);
            y_flat.push_back(y_positions[static_cast<std::size_t>(iy)]);
        }
    }

    return {x_flat, y_flat};
}

std::vector<Complex> solve_linear_system(
    std::vector<std::vector<Complex>> matrix,
    std::vector<Complex> rhs
) {
    const std::size_t n = matrix.size();
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t pivot = i;
        double pivot_abs = std::abs(matrix[i][i]);
        for (std::size_t r = i + 1; r < n; ++r) {
            const double candidate_abs = std::abs(matrix[r][i]);
            if (candidate_abs > pivot_abs) {
                pivot = r;
                pivot_abs = candidate_abs;
            }
        }

        if (pivot_abs < 1e-14) {
            throw std::invalid_argument("Singular constraint matrix in null steering");
        }

        if (pivot != i) {
            std::swap(matrix[i], matrix[pivot]);
            std::swap(rhs[i], rhs[pivot]);
        }

        const Complex diag = matrix[i][i];
        for (std::size_t c = i; c < n; ++c) {
            matrix[i][c] /= diag;
        }
        rhs[i] /= diag;

        for (std::size_t r = 0; r < n; ++r) {
            if (r == i) {
                continue;
            }
            const Complex factor = matrix[r][i];
            for (std::size_t c = i; c < n; ++c) {
                matrix[r][c] -= factor * matrix[i][c];
            }
            rhs[r] -= factor * rhs[i];
        }
    }

    return rhs;
}

py::dict evaluate_linear_response_with_weights(
    const std::vector<Complex> &weights,
    double spacing_lambda,
    py::array_t<double, py::array::c_style | py::array::forcecast> theta_grid_deg,
    py::array_t<double, py::array::c_style | py::array::forcecast> phi_grid_deg
) {
    auto theta = theta_grid_deg.unchecked<2>();
    auto phi = phi_grid_deg.unchecked<2>();
    if (theta.shape(0) != phi.shape(0) || theta.shape(1) != phi.shape(1)) {
        throw std::invalid_argument("theta_grid_deg and phi_grid_deg must have the same shape");
    }

    const py::ssize_t rows = theta.shape(0);
    const py::ssize_t cols = theta.shape(1);
    const auto positions = centered_positions(static_cast<int>(weights.size()), spacing_lambda);

    py::array_t<Complex> response({rows, cols});
    py::array_t<double> magnitude({rows, cols});
    py::array_t<double> magnitude_db({rows, cols});

    auto response_view = response.mutable_unchecked<2>();
    auto magnitude_view = magnitude.mutable_unchecked<2>();
    auto magnitude_db_view = magnitude_db.mutable_unchecked<2>();

    double max_magnitude = 0.0;
    for (py::ssize_t r = 0; r < rows; ++r) {
        for (py::ssize_t c = 0; c < cols; ++c) {
            const double ux = direction_cosine_x(theta(r, c), phi(r, c));
            Complex sum = 0.0;
            for (std::size_t i = 0; i < weights.size(); ++i) {
                const double phase = 2.0 * kPi * positions[i] * ux;
                sum += weights[i] * std::exp(Complex(0.0, phase));
            }
            response_view(r, c) = sum;
            const double mag = std::abs(sum);
            magnitude_view(r, c) = mag;
            max_magnitude = std::max(max_magnitude, mag);
        }
    }

    const double norm = max_magnitude + 1e-12;
    for (py::ssize_t r = 0; r < rows; ++r) {
        for (py::ssize_t c = 0; c < cols; ++c) {
            const double mag = magnitude_view(r, c) / norm;
            magnitude_view(r, c) = mag;
            magnitude_db_view(r, c) = 20.0 * std::log10(std::max(mag, 1e-12));
        }
    }

    py::dict result;
    result["response"] = response;
    result["magnitude"] = magnitude;
    result["magnitude_db"] = magnitude_db;
    result["positions_lambda"] = vector_to_numpy(positions);
    result["weights"] = complex_vector_to_numpy(weights);
    return result;
}

}  // namespace

py::array_t<double> element_positions_linear(int num_elements, double spacing_lambda) {
    return vector_to_numpy(centered_positions(num_elements, spacing_lambda));
}

py::array_t<double> element_positions_planar(
    int num_x,
    int num_y,
    double spacing_x_lambda,
    double spacing_y_lambda
) {
    const auto [x_flat, y_flat] = planar_positions_impl(num_x, num_y, spacing_x_lambda, spacing_y_lambda);
    return pair_vectors_to_numpy(x_flat, y_flat);
}

py::array_t<double> amplitude_taper(int num_elements, const std::string &taper_name) {
    return vector_to_numpy(amplitude_taper_impl(num_elements, taper_name));
}

py::tuple steering_weights_linear(
    int num_elements,
    double spacing_lambda,
    double theta_steer_deg,
    double phi_steer_deg,
    const std::string &taper_name = "uniform"
) {
    const auto positions = centered_positions(num_elements, spacing_lambda);
    const auto amplitudes = amplitude_taper_impl(num_elements, taper_name);
    const double ux0 = direction_cosine_x(theta_steer_deg, phi_steer_deg);

    std::vector<Complex> phase_weights(static_cast<std::size_t>(num_elements));
    std::vector<Complex> weights(static_cast<std::size_t>(num_elements));
    for (int i = 0; i < num_elements; ++i) {
        const double phase = -2.0 * kPi * positions[static_cast<std::size_t>(i)] * ux0;
        phase_weights[static_cast<std::size_t>(i)] = std::exp(Complex(0.0, phase));
        weights[static_cast<std::size_t>(i)] = amplitudes[static_cast<std::size_t>(i)] * phase_weights[static_cast<std::size_t>(i)];
    }

    return py::make_tuple(
        complex_vector_to_numpy(weights),
        vector_to_numpy(amplitudes),
        complex_vector_to_numpy(phase_weights)
    );
}

py::tuple steering_weights_planar(
    int num_x,
    int num_y,
    double spacing_x_lambda,
    double spacing_y_lambda,
    double theta_steer_deg,
    double phi_steer_deg,
    const std::string &taper_x = "uniform",
    const std::string &taper_y = "uniform"
) {
    const auto [x_flat, y_flat] = planar_positions_impl(num_x, num_y, spacing_x_lambda, spacing_y_lambda);
    const auto taper_x_values = amplitude_taper_impl(num_x, taper_x);
    const auto taper_y_values = amplitude_taper_impl(num_y, taper_y);
    const double ux0 = direction_cosine_x(theta_steer_deg, phi_steer_deg);
    const double uy0 = direction_cosine_y(theta_steer_deg, phi_steer_deg);

    std::vector<double> amplitudes;
    std::vector<Complex> phase_weights;
    std::vector<Complex> weights;
    amplitudes.reserve(x_flat.size());
    phase_weights.reserve(x_flat.size());
    weights.reserve(x_flat.size());

    for (int iy = 0; iy < num_y; ++iy) {
        for (int ix = 0; ix < num_x; ++ix) {
            const std::size_t index = static_cast<std::size_t>(iy * num_x + ix);
            const double amplitude = taper_x_values[static_cast<std::size_t>(ix)] * taper_y_values[static_cast<std::size_t>(iy)];
            const double phase = -2.0 * kPi * (x_flat[index] * ux0 + y_flat[index] * uy0);
            const Complex phase_weight = std::exp(Complex(0.0, phase));
            amplitudes.push_back(amplitude);
            phase_weights.push_back(phase_weight);
            weights.push_back(amplitude * phase_weight);
        }
    }

    return py::make_tuple(
        complex_vector_to_numpy(weights),
        vector_to_numpy(amplitudes),
        complex_vector_to_numpy(phase_weights),
        pair_vectors_to_numpy(x_flat, y_flat)
    );
}

py::array_t<Complex> null_steering_weights_linear(
    int num_elements,
    double spacing_lambda,
    double theta_steer_deg,
    double phi_steer_deg,
    py::array_t<double, py::array::c_style | py::array::forcecast> null_thetas_deg,
    py::array_t<double, py::array::c_style | py::array::forcecast> null_phis_deg,
    const std::string &taper_name = "uniform"
) {
    auto null_thetas = null_thetas_deg.unchecked<1>();
    auto null_phis = null_phis_deg.unchecked<1>();
    if (null_thetas.shape(0) != null_phis.shape(0)) {
        throw std::invalid_argument("null_thetas_deg and null_phis_deg must have the same length");
    }

    const auto amplitudes = amplitude_taper_impl(num_elements, taper_name);
    const std::size_t num_nulls = static_cast<std::size_t>(null_thetas.shape(0));
    const std::size_t num_constraints = num_nulls + 1;

    std::vector<std::vector<Complex>> constraints(
        static_cast<std::size_t>(num_elements),
        std::vector<Complex>(num_constraints, 0.0)
    );

    auto desired_manifold = linear_manifold_vector(num_elements, spacing_lambda, theta_steer_deg, phi_steer_deg);
    for (int i = 0; i < num_elements; ++i) {
        constraints[static_cast<std::size_t>(i)][0] =
            amplitudes[static_cast<std::size_t>(i)] * desired_manifold[static_cast<std::size_t>(i)];
    }

    for (std::size_t k = 0; k < num_nulls; ++k) {
        auto null_manifold = linear_manifold_vector(
            num_elements,
            spacing_lambda,
            null_thetas(static_cast<py::ssize_t>(k)),
            null_phis(static_cast<py::ssize_t>(k))
        );
        for (int i = 0; i < num_elements; ++i) {
            constraints[static_cast<std::size_t>(i)][k + 1] =
                amplitudes[static_cast<std::size_t>(i)] * null_manifold[static_cast<std::size_t>(i)];
        }
    }

    std::vector<std::vector<Complex>> gram(num_constraints, std::vector<Complex>(num_constraints, 0.0));
    for (std::size_t i = 0; i < num_constraints; ++i) {
        for (std::size_t j = 0; j < num_constraints; ++j) {
            Complex sum = 0.0;
            for (int n = 0; n < num_elements; ++n) {
                sum += constraints[static_cast<std::size_t>(n)][i] * constraints[static_cast<std::size_t>(n)][j];
            }
            gram[i][j] = sum;
        }
    }

    std::vector<Complex> targets(num_constraints, 0.0);
    targets[0] = 1.0;
    const auto coeffs = solve_linear_system(gram, targets);

    std::vector<Complex> weights(static_cast<std::size_t>(num_elements), 0.0);
    for (int n = 0; n < num_elements; ++n) {
        for (std::size_t k = 0; k < num_constraints; ++k) {
            weights[static_cast<std::size_t>(n)] += constraints[static_cast<std::size_t>(n)][k] * coeffs[k];
        }
    }

    return complex_vector_to_numpy(weights);
}

py::dict array_factor_linear(
    int num_elements,
    double spacing_lambda,
    py::array_t<double, py::array::c_style | py::array::forcecast> theta_grid_deg,
    py::array_t<double, py::array::c_style | py::array::forcecast> phi_grid_deg,
    double theta_steer_deg,
    double phi_steer_deg,
    const std::string &taper_name = "uniform"
) {
    const auto weights_tuple = steering_weights_linear(
        num_elements,
        spacing_lambda,
        theta_steer_deg,
        phi_steer_deg,
        taper_name
    );
    auto weights_array = weights_tuple[0].cast<py::array_t<Complex, py::array::c_style | py::array::forcecast>>();
    auto weights_view = weights_array.unchecked<1>();
    std::vector<Complex> weights(static_cast<std::size_t>(weights_view.shape(0)));
    for (py::ssize_t i = 0; i < weights_view.shape(0); ++i) {
        weights[static_cast<std::size_t>(i)] = weights_view(i);
    }

    py::dict result = evaluate_linear_response_with_weights(weights, spacing_lambda, theta_grid_deg, phi_grid_deg);
    result["amplitudes"] = weights_tuple[1];
    result["phase_weights"] = weights_tuple[2];
    return result;
}

py::dict array_factor_linear_from_weights(
    py::array_t<Complex, py::array::c_style | py::array::forcecast> weights,
    double spacing_lambda,
    py::array_t<double, py::array::c_style | py::array::forcecast> theta_grid_deg,
    py::array_t<double, py::array::c_style | py::array::forcecast> phi_grid_deg
) {
    auto weights_view = weights.unchecked<1>();
    std::vector<Complex> weights_vector(static_cast<std::size_t>(weights_view.shape(0)));
    for (py::ssize_t i = 0; i < weights_view.shape(0); ++i) {
        weights_vector[static_cast<std::size_t>(i)] = weights_view(i);
    }
    return evaluate_linear_response_with_weights(weights_vector, spacing_lambda, theta_grid_deg, phi_grid_deg);
}

py::dict array_factor_planar(
    int num_x,
    int num_y,
    double spacing_x_lambda,
    double spacing_y_lambda,
    py::array_t<double, py::array::c_style | py::array::forcecast> theta_grid_deg,
    py::array_t<double, py::array::c_style | py::array::forcecast> phi_grid_deg,
    double theta_steer_deg,
    double phi_steer_deg,
    const std::string &taper_x = "uniform",
    const std::string &taper_y = "uniform"
) {
    auto theta = theta_grid_deg.unchecked<2>();
    auto phi = phi_grid_deg.unchecked<2>();
    if (theta.shape(0) != phi.shape(0) || theta.shape(1) != phi.shape(1)) {
        throw std::invalid_argument("theta_grid_deg and phi_grid_deg must have the same shape");
    }

    const py::ssize_t rows = theta.shape(0);
    const py::ssize_t cols = theta.shape(1);
    const auto [x_flat, y_flat] = planar_positions_impl(num_x, num_y, spacing_x_lambda, spacing_y_lambda);
    const auto taper_x_values = amplitude_taper_impl(num_x, taper_x);
    const auto taper_y_values = amplitude_taper_impl(num_y, taper_y);
    const double ux0 = direction_cosine_x(theta_steer_deg, phi_steer_deg);
    const double uy0 = direction_cosine_y(theta_steer_deg, phi_steer_deg);

    std::vector<double> amplitudes;
    std::vector<Complex> phase_weights;
    std::vector<Complex> weights;
    amplitudes.reserve(x_flat.size());
    phase_weights.reserve(x_flat.size());
    weights.reserve(x_flat.size());

    for (int iy = 0; iy < num_y; ++iy) {
        for (int ix = 0; ix < num_x; ++ix) {
            const std::size_t index = static_cast<std::size_t>(iy * num_x + ix);
            const double amplitude = taper_x_values[static_cast<std::size_t>(ix)] * taper_y_values[static_cast<std::size_t>(iy)];
            const double phase = -2.0 * kPi * (x_flat[index] * ux0 + y_flat[index] * uy0);
            const Complex phase_weight = std::exp(Complex(0.0, phase));
            amplitudes.push_back(amplitude);
            phase_weights.push_back(phase_weight);
            weights.push_back(amplitude * phase_weight);
        }
    }

    py::array_t<Complex> response({rows, cols});
    py::array_t<double> magnitude({rows, cols});
    py::array_t<double> magnitude_db({rows, cols});
    auto response_view = response.mutable_unchecked<2>();
    auto magnitude_view = magnitude.mutable_unchecked<2>();
    auto magnitude_db_view = magnitude_db.mutable_unchecked<2>();

    double max_magnitude = 0.0;
    for (py::ssize_t r = 0; r < rows; ++r) {
        for (py::ssize_t c = 0; c < cols; ++c) {
            const double ux = direction_cosine_x(theta(r, c), phi(r, c));
            const double uy = direction_cosine_y(theta(r, c), phi(r, c));
            Complex sum = 0.0;
            for (std::size_t i = 0; i < weights.size(); ++i) {
                const double phase = 2.0 * kPi * (x_flat[i] * ux + y_flat[i] * uy);
                sum += weights[i] * std::exp(Complex(0.0, phase));
            }
            response_view(r, c) = sum;
            const double mag = std::abs(sum);
            magnitude_view(r, c) = mag;
            max_magnitude = std::max(max_magnitude, mag);
        }
    }

    const double norm = max_magnitude + 1e-12;
    for (py::ssize_t r = 0; r < rows; ++r) {
        for (py::ssize_t c = 0; c < cols; ++c) {
            const double mag = magnitude_view(r, c) / norm;
            magnitude_view(r, c) = mag;
            magnitude_db_view(r, c) = 20.0 * std::log10(std::max(mag, 1e-12));
        }
    }

    py::dict result;
    result["response"] = response;
    result["magnitude"] = magnitude;
    result["magnitude_db"] = magnitude_db;
    result["positions_lambda"] = pair_vectors_to_numpy(x_flat, y_flat);
    result["amplitudes"] = vector_to_numpy(amplitudes);
    result["phase_weights"] = complex_vector_to_numpy(phase_weights);
    result["weights"] = complex_vector_to_numpy(weights);
    return result;
}

PYBIND11_MODULE(_beamforming_cpp, m) {
    m.doc() = "C++ beamforming core";
    m.def("element_positions_linear", &element_positions_linear, py::arg("num_elements"), py::arg("spacing_lambda"));
    m.def(
        "element_positions_planar",
        &element_positions_planar,
        py::arg("num_x"),
        py::arg("num_y"),
        py::arg("spacing_x_lambda"),
        py::arg("spacing_y_lambda")
    );
    m.def("amplitude_taper", &amplitude_taper, py::arg("num_elements"), py::arg("taper_name"));
    m.def(
        "steering_weights_linear",
        &steering_weights_linear,
        py::arg("num_elements"),
        py::arg("spacing_lambda"),
        py::arg("theta_steer_deg"),
        py::arg("phi_steer_deg"),
        py::arg("taper_name") = "uniform"
    );
    m.def(
        "steering_weights_planar",
        &steering_weights_planar,
        py::arg("num_x"),
        py::arg("num_y"),
        py::arg("spacing_x_lambda"),
        py::arg("spacing_y_lambda"),
        py::arg("theta_steer_deg"),
        py::arg("phi_steer_deg"),
        py::arg("taper_x") = "uniform",
        py::arg("taper_y") = "uniform"
    );
    m.def(
        "null_steering_weights_linear",
        &null_steering_weights_linear,
        py::arg("num_elements"),
        py::arg("spacing_lambda"),
        py::arg("theta_steer_deg"),
        py::arg("phi_steer_deg"),
        py::arg("null_thetas_deg"),
        py::arg("null_phis_deg"),
        py::arg("taper_name") = "uniform"
    );
    m.def(
        "array_factor_linear",
        &array_factor_linear,
        py::arg("num_elements"),
        py::arg("spacing_lambda"),
        py::arg("theta_grid_deg"),
        py::arg("phi_grid_deg"),
        py::arg("theta_steer_deg"),
        py::arg("phi_steer_deg"),
        py::arg("taper_name") = "uniform"
    );
    m.def(
        "array_factor_linear_from_weights",
        &array_factor_linear_from_weights,
        py::arg("weights"),
        py::arg("spacing_lambda"),
        py::arg("theta_grid_deg"),
        py::arg("phi_grid_deg")
    );
    m.def(
        "array_factor_planar",
        &array_factor_planar,
        py::arg("num_x"),
        py::arg("num_y"),
        py::arg("spacing_x_lambda"),
        py::arg("spacing_y_lambda"),
        py::arg("theta_grid_deg"),
        py::arg("phi_grid_deg"),
        py::arg("theta_steer_deg"),
        py::arg("phi_steer_deg"),
        py::arg("taper_x") = "uniform",
        py::arg("taper_y") = "uniform"
    );
}
