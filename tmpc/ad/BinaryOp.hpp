#pragma once

#include "ExpressionBase.hpp"
#include "Eval.hpp"

namespace tmpc
{
    template <typename OP, typename L, typename R>
    class BinaryOp
    :   public ExpressionBase<BinaryOp<OP, L, R>>
    {
    public:
        using Left = L;
        using Right = R;
        using Op = OP;

        constexpr BinaryOp(Left const& left, Right const& right)
        :   left_(left)
        ,   right_(right)
        {        
        }

        auto const& left() const
        {
            return left_;
        }

        auto const& right() const
        {
            return right_;
        }
        
    private:
        Left left_;
        Right right_;
    };


    template <typename OP, typename L, typename R, class... Types>
    decltype(auto) eval(BinaryOp<OP, L, R> const& expr, std::tuple<Types...> const& arg)
    {
        return OP::eval(eval(expr.left(), arg), eval(expr.right(), arg));
    }
}