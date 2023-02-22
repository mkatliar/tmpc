#
# Copyright 2019 Gianluca Frison, Dimitris Kouzoupis, Robin Verschueren,
# Andrea Zanelli, Niels van Duijkeren, Jonathan Frey, Tommaso Sartor,
# Branimir Novoselnik, Rien Quirynen, Rezart Qelibari, Dang Doan,
# Jonas Koenemann, Yutao Chen, Tobias Schöls, Jonas Schlagenhauf, Moritz Diehl
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

from typing import Any, Dict
import casadi as cs

def make_pendulum_ode_model() -> Dict[str, Any]:

    model_name = 'pendulum_ode'

    # constants
    M = 1. # mass of the cart [kg] -> now estimated
    m = 0.1 # mass of the ball [kg]
    g = 9.81 # gravity constant [m/s^2]
    l = 0.8 # length of the rod [m]

    # set up states & controls
    x1      = cs.SX.sym('x1')
    theta   = cs.SX.sym('theta')
    v1      = cs.SX.sym('v1')
    dtheta  = cs.SX.sym('dtheta')

    x = cs.vertcat(x1, theta, v1, dtheta)

    F = cs.SX.sym('F')
    u = cs.vertcat(F)

    # xdot
    x1_dot      = cs.SX.sym('x1_dot')
    theta_dot   = cs.SX.sym('theta_dot')
    v1_dot      = cs.SX.sym('v1_dot')
    dtheta_dot  = cs.SX.sym('dtheta_dot')

    xdot = cs.vertcat(x1_dot, theta_dot, v1_dot, dtheta_dot)

    # dynamics
    cos_theta = cs.cos(theta)
    sin_theta = cs.sin(theta)
    denominator = M + m - m*cos_theta*cos_theta
    f_expl = cs.vertcat(v1,
                     dtheta,
                     (-m*l*sin_theta*dtheta*dtheta + m*g*cos_theta*sin_theta+F)/denominator,
                     (-m*l*cos_theta*sin_theta*dtheta*dtheta + F*cos_theta+(M+m)*g*sin_theta)/(l*denominator)
                     )

    f_impl = xdot - f_expl

    return {
        "f_impl_expr": f_impl,
        "f_expl_expr": f_expl,
        "x": x,
        "xdot": xdot,
        "u": u,
        "name": model_name,
    }


def functions():
    model = make_pendulum_ode_model()

    t = cs.SX.sym("t")
    xdot = model["xdot"]
    x = model["x"]
    z = cs.SX.sym("z", 0)
    u = model["u"]
    f_impl = model["f_impl_expr"]

    nx = x.numel()
    nz = z.numel()
    nu = u.numel()
    Sx = cs.SX.sym("Sx", nx + nz, nx + nu)

    return [
        cs.Function(
            "impl_dae",
            [t, xdot, x, z, u],
            [f_impl, cs.jacobian(f_impl, xdot), cs.jacobian(f_impl, x), cs.jacobian(f_impl, z)]
        ),
        cs.Function(
            "impl_dae_s",
            [t, xdot, x, Sx, z, u],
            [cs.jtimes(f_impl, x, Sx) + cs.horzcat(cs.SX.zeros(nx, nx), cs.jacobian(f_impl, u))]
        )
    ]
