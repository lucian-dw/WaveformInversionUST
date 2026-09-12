function geo = wust_geometry(grid, tx, rx)
% Pixel-edge origin [y,x], spacing [dy,dx]; positions are physical [x,y].
assert(isequal(sort(fieldnames(grid)), sort({'shape_yx'; 'origin_yx_m'; ...
    'spacing_yx_m'; 'origin_kind'})), 'WUST:InvalidInput', 'Explicit grid fields required');
n = double(grid.shape_yx(:)); origin = double(grid.origin_yx_m(:));
spacing = double(grid.spacing_yx_m(:));
assert(numel(n)==2 && all(isfinite(n)&n>=3&n==round(n)) && ...
    numel(origin)==2 && all(isfinite(origin)) && numel(spacing)==2 && ...
    all(isfinite(spacing)&spacing>0) && strcmp(grid.origin_kind,'pixel_edge'), ...
    'WUST:InvalidInput', 'Invalid grid shape/origin/spacing');
y = origin(1) + ((0:n(1)-1)+0.5)*spacing(1);
x = origin(2) + ((0:n(2)-1)+0.5)*spacing(2);
[ti, ts] = snap(tx, x, y); [ri, rs] = snap(rx, x, y);
geo = struct('x_m', x, 'y_m', y, 'tx_index', ti, 'rx_index', ri, ...
    'tx_xy_m', tx, 'rx_xy_m', rx, 'tx_snapped_xy_m', ts, 'rx_snapped_xy_m', rs);
end

function [index, snapped] = snap(xy, x, y)
assert(isnumeric(xy) && isreal(xy) && size(xy,2)==2 && ~isempty(xy) && ...
    all(isfinite(xy(:))), 'WUST:InvalidInput', 'Positions must be finite [N,2] physical [x,y]');
assert(all(xy(:,1)>=x(1)&xy(:,1)<=x(end)&xy(:,2)>=y(1)&xy(:,2)<=y(end)), ...
    'WUST:InvalidInput', 'Array outside grid sampling coordinates');
[~, ix] = min(abs(x(:)-xy(:,1).'),[],1);
[~, iy] = min(abs(y(:)-xy(:,2).'),[],1);
index = sub2ind([numel(y),numel(x)],iy,ix).';
assert(numel(unique(index))==numel(index), 'WUST:InvalidInput', ...
    'Distinct elements in one array collapse onto a grid sample');
snapped = [reshape(x(ix),[],1), reshape(y(iy),[],1)];
end
