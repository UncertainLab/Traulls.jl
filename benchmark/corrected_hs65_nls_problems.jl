# HS problem 65 in NLS format
#
#   Source:
#   W. Hock and K. Schittkowski,
#   Test examples for nonlinear programming codes,
#   Lecture Notes in Economics and Mathematical Systems 187,
#   Springer Verlag Berlin Heidelberg, 1981
#   10.1007/978-3-642-48320-2
#
# A. S. Siqueira, Curitiba/BR, 04/2018.
#
# NLSProblems.hs65 (version 0.5.3) states the constraint as an equality; Hock and
# Schittkowski, and traulls_models/hs65.jl, state it as an inequality. The equality
# variant has an additional KKT point, x = (-1.658, 4.5, 5), at which Percival and Ipopt
# stop when started from x0.

using JuMP
using NLSProblems.NLPModelsJuMP: MathOptNLSModel

"Hock-Schittkowski problem 65 in NLS format, with the inequality constraint"
function corrected_hs65(args...)
  model = Model()
  lvar, uvar = [-4.5; -4.5; -5.0], [4.5; 4.5; 5.0]
  @variable(model, lvar[i] ≤ x[i = 1:3] ≤ uvar[i])
  set_start_value.(x, [-5.0; 5.0; 0.0])
  @expression(model, F1, x[1] - x[2])
  @expression(model, F2, (x[1] + x[2] - 10) / 3)
  @expression(model, F3, x[3] - 5)
  @NLconstraint(model, 48 - x[1]^2 - x[2]^2 - x[3]^2 >= 0)

  return MathOptNLSModel(model, [F1; F2; F3], name = "hs65")
end
