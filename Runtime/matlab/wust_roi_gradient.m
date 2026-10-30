function [roi_gradient, z] = wust_roi_gradient(g, mask, x_m, y_m, c0, frequency_hz, cutoff, order)
    % M F M g: restrict before filtering to prevent exterior-gradient leakage.
    roi_gradient = g;
    roi_gradient(~mask) = 0;
    z = roi_gradient;
    if cutoff > 0
        z = ringingRemovalFilt(x_m, y_m, z, c0, frequency_hz, cutoff, order);
    end
    z(~mask) = 0;
end
