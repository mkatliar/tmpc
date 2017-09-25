#pragma once

#include "ExpressionBase.hpp"

namespace tmpc
{
    class Zero final
    :   public ExpressionBase<Zero>
    {
    public:
        constexpr Zero() 
        {            
        }

        size_t implOperationCount() const
        {
            return 0;
        }
    };
}