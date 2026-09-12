function wust_write_artifact(path, result)
% This is private staging output; Python hashes, validates and atomically publishes.
file=fullfile(fileparts(path),'arrays.h5');assert(~isfile(file)&&~isfile(path),'Output exists');
meta=result;specs=struct;
switch result.schema
    case 'wust.frequency_input'
        names={'pressure','mask','frequencies_hz','tx_xy_m','rx_xy_m'};
        axes={result.pressure_axes,result.mask_axes,'frequency','tx,xy','rx,xy'};
        units={result.data_units,'1','Hz','m','m'};
    case 'wust.measurements'
        names={'Y','mask','frequencies_hz','tx_xy_m','rx_xy_m','tx_index','rx_index', ...
            'x_m','y_m','tx_snapped_xy_m','rx_snapped_xy_m'};
        axes={'tx,rx,frequency','tx,rx,frequency','frequency','tx,xy','rx,xy','tx','rx', ...
            'x','y','tx,xy','rx,xy'};
        units={result.data_units,'1','Hz','m','m','1','1','m','m','m','m'};
    case 'wust.reconstruction'
        names={'c_mps','history_mps'};axes={'y,x','y,x,model_step'};units={'m/s','m/s'};
        meta.config=rmfield(meta.config,'update_mask');
        meta.executed_schedule=num2cell(meta.executed_schedule(:).');
        meta.config.schedule=num2cell(meta.config.schedule(:).');
        meta.records=arrayfun(@(r)r,meta.records,'UniformOutput',false);
    case 'wust.rf'
        names={'pressure','time_s','tx_xy_m','rx_xy_m'};
        axes={'time,rx,tx','time','tx,xy','rx,xy'};units={'Pa','s','m','m'};
    otherwise,error('WUST:InvalidInput','Unsupported output artifact');
end
for i=1:numel(names)
    key=names{i};a=result.(key);meta=rmfield(meta,key);
    rank=numel(strsplit(axes{i},','));shape=zeros(1,rank);
    for k=1:rank,shape(k)=size(a,k);end
    if rank==1,shape=numel(a);end
    dtype=class(a);
    switch dtype
        case 'double',dtype='float64';
        case 'single',dtype='float32';
        case 'logical',dtype='bool';
    end
    spec=struct('axes',axes{i},'shape',shape,'dtype',dtype,'units',units{i});
    % Always emit shape as a JSON array, including rank-one arrays.
    spec.shape=num2cell(shape);
    if strcmp(key,'Y') || (strcmp(result.schema,'wust.frequency_input')&&strcmp(key,'pressure'))
        if isa(a,'single'),spec.dtype='complex64';else,spec.dtype='complex128';end
        spec.real_dataset=['/' key '_real'];spec.imag_dataset=['/' key '_imag'];
        write(file,spec.real_dataset,real(a),shape);write(file,spec.imag_dataset,imag(a),shape);
    else
        spec.dataset=['/' key];write(file,spec.dataset,a,shape);
    end
    specs.(key)=spec;
end
meta=rmfield(meta,{'schema','schema_version'});
doc=struct('schema',result.schema,'schema_version',1,'metadata',meta, ...
    'arrays_file','arrays.h5','arrays_sha256','','arrays',specs);
fid=fopen(path,'w');assert(fid>=0,'Cannot create output');close=onCleanup(@()fclose(fid));
fprintf(fid,'%s',jsonencode(doc));
end

function write(file,name,a,shape)
if islogical(a),a=uint8(a);end
if numel(shape)>1,a=permute(a,numel(shape):-1:1);else,a=a(:);end
h5create(file,name,fliplr(shape),'Datatype',class(a));h5write(file,name,a);
end
