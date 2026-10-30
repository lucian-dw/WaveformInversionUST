function alpha = linearized_step_length(slope, curvature)
% Minimize the frozen-source quadratic along a descent direction.
assert(isfinite(slope) && isfinite(curvature) && curvature >= 0, ...
    'WUST:NumericalFailure', 'Nonfinite slope/curvature');
assert(curvature > 0 || slope == 0, ...
    'WUST:NumericalFailure', 'Zero curvature with nonzero directional slope');
alpha = 0;
if curvature > 0
    alpha = max(0, -slope/curvature);
end
end
