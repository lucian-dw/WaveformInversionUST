function [curvature, linear_seconds] = directional_curvature(state, direction, rx_index, receiver_mask)
    % Frozen fitted source scales, and the same mass stencil as the oracle.
    curvature = 0;
    offset = 0;
    linearTimer = tic;
    for b = 1:numel(state.virtual)
        v = state.virtual{b};
        a = state.scales{b};
        nb = numel(a);
        ids = offset + (1:nb);
        offset = offset + nb;
        source = state.solver.massApply(v .* direction, false);
        u = state.solver.solve(source, false);
        u = reshape(u, [], nb);
        derivative = -u(rx_index, :) .* a .* receiver_mask(ids, :).';
        val = sum(abs(derivative).^2, 'all');
        if isa(val, 'gpuArray')
            val = gather(val);
        end
        curvature = curvature + double(val);
    end
    if strcmp(state.solver.backend, 'gpu')
        wait(gpuDevice);
    end
    linear_seconds = toc(linearTimer);
end
