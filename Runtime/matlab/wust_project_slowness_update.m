function c_new = wust_project_slowness_update(c, direction, alpha, cfg)
    % Project in slowness before reciprocal conversion, including across zero.
    c_lo = max(c - cfg.max_update_mps, cfg.bounds_mps(1));
    c_hi = min(c + cfg.max_update_mps, cfg.bounds_mps(2));
    trial = 1 ./ c + cfg.step_damping * alpha * direction;
    assert(all(isfinite(trial(:))), 'WUST:NumericalFailure', 'Nonfinite slowness trial');
    s_new = max(1 ./ c_hi, min(1 ./ c_lo, trial));
    c_new = 1 ./ s_new;
    c_new(~cfg.update_mask) = c(~cfg.update_mask);
end
