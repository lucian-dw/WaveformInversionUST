function obs=wust_test_measurements(obs)
% Supply explicit geometry to the unchanged numerical oracle fixture.
dx=mean(diff(obs.x_m));dy=mean(diff(obs.y_m));
obs.grid=struct('shape_yx',[numel(obs.y_m),numel(obs.x_m)], ...
    'origin_yx_m',[obs.y_m(1)-dy/2,obs.x_m(1)-dx/2], ...
    'spacing_yx_m',[dy,dx],'origin_kind','pixel_edge');
[r,c]=ind2sub(obs.grid.shape_yx,obs.tx_index);obs.tx_xy_m=[reshape(obs.x_m(c),[],1),reshape(obs.y_m(r),[],1)];
[r,c]=ind2sub(obs.grid.shape_yx,obs.rx_index);obs.rx_xy_m=[reshape(obs.x_m(c),[],1),reshape(obs.y_m(r),[],1)];
obs.tx_snapped_xy_m=obs.tx_xy_m;obs.rx_snapped_xy_m=obs.rx_xy_m;
obs.schema='wust.measurements';obs.schema_version=1;
obs.data_units='unknown';obs.spectrum_normalization='unknown';
obs.preparation=struct('measurement_provenance','numerical_fixture');
end
