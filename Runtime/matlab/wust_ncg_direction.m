function [direction, beta, gz] = wust_ncg_direction(g, z, prev_z, prev_gz, previous_direction, same_frequency)
    % Preconditioned clipped PR/FR with the true objective slope.
    gz = sum(g(:) .* z(:));
    beta = 0;
    if same_frequency && prev_gz > 0
        beta_fr = gz / prev_gz;
        beta_pr = sum(g(:) .* (z(:) - prev_z(:))) / prev_gz;
        beta = min(max(beta_pr, 0), beta_fr);
    end
    direction = -z + beta * previous_direction;
    if sum(g(:) .* direction(:)) >= 0
        direction = -z;
        beta = 0;
    end
    assert(all(isfinite(direction(:))) && isfinite(gz) && gz >= 0, ...
           'WUST:NumericalFailure', 'Invalid preconditioned gradient');
end
