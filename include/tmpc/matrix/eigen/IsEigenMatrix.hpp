#pragma once

#include "Eigen.hpp"

#include <type_traits>

namespace tmpc
{
    template <typename T>
    using IsEigenMatrix = std::is_base_of<
        Eigen::MatrixBase<
            std::remove_cv_t<T>
        >,
        std::remove_cv_t<T>
    >;
}