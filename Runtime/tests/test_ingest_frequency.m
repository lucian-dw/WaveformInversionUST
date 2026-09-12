function test_ingest_frequency
root=fileparts(fileparts(mfilename('fullpath')));addpath(fullfile(root,'matlab'));
grid=struct('shape_yx',[7,9],'origin_yx_m',[.01,-.02], ...
    'spacing_yx_m',[.001,.002],'origin_kind','pixel_edge');
x=-.02+((0:8)+.5)*.002;y=.01+((0:6)+.5)*.001;
tx=[x(3),y(3);x(6),y(5)];rx=[tx(1,:);x(7),y(2);x(4),y(6);x(8),y(4)];
p=complex(single(reshape(1:24,3,2,4)),single(reshape(101:124,3,2,4)));
mask=true(size(p));mask(1,2,3)=false;p(1,2,3)=NaN;
input=struct('schema','wust.frequency_input','schema_version',1,'grid',grid, ...
    'tx_xy_m',tx,'rx_xy_m',rx,'frequencies_hz',[3e5,1e5,2e5], ...
    'pressure',p,'pressure_axes','frequency,tx,rx','mask',mask,'mask_axes','frequency,tx,rx', ...
    'fourier_sign',1,'real_pressure',true,'pressure_type','total_pressure', ...
    'data_units','instrument_units','spectrum_normalization','dtft_dt','measurement_provenance','fixture');
obs=wust_ingest_frequency(input);
expected=p;expected(~mask)=0;expected=permute(conj(expected([2,3,1],:,:)),[2,3,1]);
assert(isequal(obs.Y,expected)&&isequal(obs.mask,permute(mask([2,3,1],:,:),[2,3,1])));
assert(isequal(obs.tx_index,[17;40])&&isequal(obs.rx_index,[17;44;27;53]));
assert(isa(obs.Y,'single')&&isequal(obs.frequencies_hz,[1e5;2e5;3e5]));
assert(max(abs(obs.x_m-x))<eps&&max(abs(obs.y_m-y))<eps);
again=input;again.pressure=permute(obs.Y,[3,1,2]);again.mask=permute(obs.mask,[3,1,2]);
again.frequencies_hz=obs.frequencies_hz;again.fourier_sign=-1;
repeated=wust_ingest_frequency(again);assert(isequal(repeated.Y,obs.Y));
bad=input;bad.fourier_sign=0;reject(bad);
bad=input;bad.real_pressure=false;reject(bad);
bad=input;bad.frequencies_hz=[1,1,2];reject(bad);
bad=input;bad.tx_xy_m(2,:)=bad.tx_xy_m(1,:)+1e-6;reject(bad);
bad=input;bad.pressure(2,1,1)=Inf;reject(bad);
bad=input;bad.mask=double(bad.mask);bad.mask(2,1,1)=2;reject(bad);
bad=input;bad.pressure_axes='tx,frequency,rx';reject(bad);
good=input;good.pressure(2,1,1)=0;zero=wust_ingest_frequency(good);assert(zero.mask(1,1,1)&&zero.Y(1,1,1)==0);
good=input;good.pressure(1,2,3)=1;good.mask=true(2,4);good.mask_axes='tx,rx';
broadcast=wust_ingest_frequency(good);assert(all(broadcast.mask(:)));
% Independent dt-weighted DTFT, including nonzero acquisition time origin.
t=(0:511)'*1e-7+7e-7;rf=struct('time_s',t,'pressure',reshape(sin((1:4096)*.17),512,4,2), ...
    'tx_xy_m',tx,'rx_xy_m',rx);
cfg=struct('x_m',x,'y_m',y,'frequencies_hz',[1e5,2e5,3e5], ...
    'c_geom_mps',1500,'window','none','phase_correction','none','mask',true(2,4), ...
    'data_units','instrument_units','measurement_provenance','fixture');
a=wust_prepare(rf,cfg);direct=input;direct.fourier_sign=-1;direct.frequencies_hz=cfg.frequencies_hz;
direct.mask=true(2,4);direct.mask_axes='tx,rx';
for i=1:2,direct.pressure(:,i,:)=single(exp(-2i*pi*cfg.frequencies_hz(:)*t.')*rf.pressure(:,:,i)*mean(diff(t)));end
b=wust_ingest_frequency(direct);
assert(norm(a.Y(:)-b.Y(:))/norm(a.Y(:))<1e-6);
assert(isequal(a.tx_index,b.tx_index)&&isequal(a.rx_index,b.rx_index));
masked_rf=rf;masked_rf.pressure(:,1,1)=NaN;
masked_cfg=cfg;masked_cfg.mask(1,1)=false;
masked=wust_prepare(masked_rf,masked_cfg);
assert(all(masked.Y(1,1,:)==0)&&~any(masked.mask(1,1,:)));
failed=false;try,wust_prepare(masked_rf,cfg);catch,failed=true;end
assert(failed,'Valid RF channels must reject nonfinite samples');
fprintf('Asymmetric ingestion, invalid inputs, Fourier conversion, mask and RF equivalence passed\n');
end

function reject(value)
failed=false;try,wust_ingest_frequency(value);catch err,failed=strcmp(err.identifier,'WUST:InvalidInput');end
assert(failed,'Expected an explicit invalid-input rejection');
end
