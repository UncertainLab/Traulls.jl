# Fix for NLPModelsJuMP 0.13.6, the latest release (its main branch has the same code as of
# October 2026): hess_coord!(nls::MathOptNLSModel, x, y, vals) refers to the nonexistent
# field `nls.nnln.ncon`, so that Ipopt with exact second derivatives fails with a FieldError
# on every constrained least-squares model. The method below is copied from
# src/moi_nls_model.jl of NLPModelsJuMP 0.13.6 (MPL 2.0) with `nls.nlcon.nnln` instead, and
# is evaluated in the module of the package.

let mod = Base.require(Base.PkgId(Base.UUID("792afdf1-32c1-5681-94e0-d7bf7a5df49e"),
                                  "NLPModelsJuMP"))
    pkgversion(mod) == v"0.13.6" ||
        error("nlpmodelsjump_fix.jl only applies to NLPModelsJuMP 0.13.6")

    Core.eval(mod, quote
function NLPModels.hess_coord!(
  nls::MathOptNLSModel,
  x::AbstractVector,
  y::AbstractVector,
  vals::AbstractVector;
  obj_weight::Float64 = 1.0,
)
  nls.meta.hess_available ||
    error("The function hess_coord! is not supported by this MathOptNLSModel.")
  increment!(nls, :neval_hess)
  if nls.nls_meta.nlin > 0
    view(vals, 1:(nls.lls.nnzh)) .= obj_weight .* nls.lls.hessian.vals
  end
  if nls.quadcon.nquad > 0
    index = nls.lls.nnzh
    for i = 1:(nls.quadcon.nquad)
      qcon = nls.quadcon.constraints[i]
      view(vals, (index + 1):(index + qcon.nnzh)) .= y[nls.meta.nlin + i] .* qcon.A.vals
      index += qcon.nnzh
    end
  end
  if (nls.nls_meta.nnln > 0) || (nls.nlcon.nnln > 0)
    λ = view(
      y,
      (nls.meta.nlin + nls.quadcon.nquad + 1):(nls.meta.nlin + nls.quadcon.nquad + nls.nlcon.nnln),
    )
    MOI.eval_hessian_lagrangian(
      nls.ceval,
      view(
        vals,
        (nls.lls.nnzh + nls.quadcon.nnzh + 1):(nls.lls.nnzh + nls.quadcon.nnzh + nls.nlcon.nnzh),
      ),
      x,
      obj_weight,
      λ,
    )
  end
  if nls.oracles.ncon > 0
    offset_λ = nls.meta.nlin + nls.quadcon.nquad + nls.nlcon.nnln
    offset_nnzh = nls.lls.nnzh + nls.quadcon.nnzh + nls.nlcon.nnzh
    for (f, s) in nls.oracles.oracles
      for i = 1:s.set.input_dimension
        s.x[i] = x[f.variables[i].value]
      end
      ncon_oracle = s.set.output_dimension
      index_λ = (offset_λ + 1):(offset_λ + ncon_oracle)
      λ_oracles = view(y, index_λ)
      nnzh_oracle = length(s.set.hessian_lagrangian_structure)
      index_nnzh = (offset_nnzh + 1):(offset_nnzh + nnzh_oracle)
      s.set.eval_hessian_lagrangian(view(vals, index_nnzh), s.x, λ_oracles)
      offset_λ += ncon_oracle
      offset_nnzh += nnzh_oracle
    end
  end
  return vals
end
    end)
end
