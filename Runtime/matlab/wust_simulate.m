function rf=wust_simulate(model,cfg)
% Physical input, no anatomical preprocessing. Matrix convention c(y,x).
% k-Wave calls its first matrix axis x; we explicitly map it to physical y.
% model: c_mps [Ny,Nx], x_m/y_m vectors, tx_index/rx_index MATLAB linear indices,
% source_pressure [1,Nt], dt_s. Input is already on the COMPUTATIONAL grid.
required={'backend','pml_size','density_kg_m3','sound_speed_ref_mps','downsample_factor','time_offset_s','work_dir'};
for j=1:numel(required),assert(isfield(cfg,required{j}),['Missing ' required{j}]);end
assert(exist('kWaveGrid','class')==8,'Add the external k-Wave MATLAB toolbox to path');
c=double(model.c_mps);[ny,nx]=size(c);x=double(model.x_m(:));y=double(model.y_m(:));
assert(numel(x)==nx&&numel(y)==ny && all(isfinite(c(:))&c(:)>0),'Invalid medium');
dx=mean(diff(x));dy=mean(diff(y));
assert(dx>0&&dy>0&&max(abs(diff(x)-dx))<dx*1e-7&&max(abs(diff(y)-dy))<dy*1e-7,'Uniform grid required');
assert(model.dt_s>0 && max(c(:))*model.dt_s/min(dx,dy)<=0.3,'CFL exceeds tested 0.3');
assert(cfg.density_kg_m3>0&&cfg.sound_speed_ref_mps>=max(c(:)),'Invalid density or reference speed');
assert(cfg.downsample_factor>=1&&cfg.downsample_factor==round(cfg.downsample_factor),'Invalid decimation factor');
assert(isscalar(cfg.pml_size)&&isfinite(cfg.pml_size)&&cfg.pml_size>=1&&cfg.pml_size==round(cfg.pml_size),'Positive integer outside PML required');
assert(isvector(model.source_pressure)&&numel(model.source_pressure)>1&&all(isfinite(model.source_pressure(:))),'Finite source waveform required');
assert(isscalar(cfg.time_offset_s)&&isfinite(cfg.time_offset_s),'Finite source time offset required');
tx=double(model.tx_index(:));rx=double(model.rx_index(:));
for ids={tx,rx}
    v=ids{1};assert(all(v==round(v)&v>=1&v<=nx*ny)&&numel(unique(v))==numel(v),'Invalid/duplicate array indices');
end
kgrid=kWaveGrid(ny,dy,nx,dx);kgrid.setTime(numel(model.source_pressure),model.dt_s);
medium=struct('sound_speed',single(c),'sound_speed_ref',cfg.sound_speed_ref_mps,'density',cfg.density_kg_m3);
% Native contract is linear, homogeneous density, zero absorption. No silent disabling.
sensor.mask=zeros(ny,nx);sensor.mask(rx)=1;
[~,order]=sort(rx);nr=numel(rx);nt=numel(tx);
source.p_mode='dirichlet';source.p=single(model.source_pressure(:).');
common={'PMLInside',false,'PMLSize',cfg.pml_size,'PlotSim',false,'DataCast','single'};
if ~exist(cfg.work_dir,'dir'),mkdir(cfg.work_dir);end
timer=tic;
switch cfg.backend
    case 'native128'
        assert(nt==128&&nr==128&&isequal(tx,rx),'native128 requires ordered colocated 128 TX/RX');
        assert(isfield(cfg,'binary_path')&&isfield(cfg,'device_num'),'Native binary_path/device_num required');
        source.p_mask=zeros(ny,nx);source.p_mask(tx(1))=1;
        [row,col]=ind2sub([ny,nx],tx);p=cfg.pml_size;
        h5index=(row+p)+(col+p-1)*(ny+2*p);
        old=getenv('KWAVE_NATIVE_TX_INDICES_H5');cleanup=onCleanup(@()setenv('KWAVE_NATIVE_TX_INDICES_H5',old));
        setenv('KWAVE_NATIVE_TX_INDICES_H5',strjoin(string(h5index.'),','));
        raw=kspaceFirstOrder2DG(kgrid,medium,source,sensor,common{:}, ...
            'BinaryPath',cfg.binary_path,'DeviceNum',cfg.device_num, ...
            'DataPath',cfg.work_dir,'DataName',['wust_' char(java.util.UUID.randomUUID())], ...
            'DeleteData',false);
        assert(size(raw,1)==nr*nt,'Unexpected native receiver output size');
        raw=reshape(single(raw),nr,nt,[]);
    case 'matlab-serial'
        raw=zeros(nr,nt,numel(model.source_pressure),'single');
        for j=1:nt
            source.p_mask=zeros(ny,nx);source.p_mask(tx(j))=1;
            p=kspaceFirstOrder2D(kgrid,medium,source,sensor,common{:});
            raw(:,j,:)=reshape(p,nr,1,[]);
        end
    otherwise
        error('backend must be native128 or matlab-serial');
end
generationSeconds=toc(timer);timer=tic;pressure=[];
% Filter ONE TX at a time, avoiding a double-precision copy of the full RF cube.
for j=1:nt
    ordered=zeros(nr,size(raw,3),'single');ordered(order,:)=reshape(raw(:,j,:),nr,[]);
    if cfg.downsample_factor==1
        channels=ordered.';time=kgrid.t_array;meta=struct('method','none','downsample_factor',1);
    else
        [channels,time,meta]=downsampleKWaveChannelData(ordered,kgrid.t_array,cfg.downsample_factor,'polyphase');
    end
    if isempty(pressure),pressure=zeros(size(channels,1),nr,nt,'single');end
    pressure(:,:,j)=channels;
end
[tr,tc]=ind2sub([ny,nx],tx);[rr,rc]=ind2sub([ny,nx],rx);
rf=struct('schema','wfi.rf.v1','pressure',pressure,'time_s',time(:)+cfg.time_offset_s, ...
    'tx_xy_m',[x(tc),y(tr)],'rx_xy_m',[x(rc),y(rr)], ...
    'simulation',struct('dt_s',model.dt_s,'sound_speed_ref_mps',cfg.sound_speed_ref_mps, ...
    'generation_seconds',generationSeconds,'filter_seconds',toc(timer), ...
    'resampling',meta,'config',cfg,'axis_order','time,receiver,transmitter', ...
    'physics','2D linear acoustics; scalar density; zero attenuation'));
end
