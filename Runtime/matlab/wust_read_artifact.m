function value = wust_read_artifact(path, expected)
% HDF5 dimensions are explicitly reversed at the MATLAB transport boundary.
doc=jsondecode(fileread(path));wust_assert_schema(doc,expected);
value=doc.metadata;value.schema=doc.schema;value.schema_version=doc.schema_version;
[~,name,ext]=fileparts(doc.arrays_file);
assert(strcmp([name ext],doc.arrays_file),'WUST:InvalidInput','Array file must be a sibling filename');
file=fullfile(fileparts(path),doc.arrays_file);
names=fieldnames(doc.arrays);
for i=1:numel(names)
    key=names{i};spec=doc.arrays.(key);
    if startsWith(spec.dtype,'complex')
        a=read(file,spec.real_dataset,spec.shape)+1i*read(file,spec.imag_dataset,spec.shape);
    else
        a=read(file,spec.dataset,spec.shape);
        if strcmp(spec.dtype,'bool'),a=logical(a);end
    end
    value.(key)=a;
    if ismember(key,{'pressure','mask'}),value.([key '_axes'])=spec.axes;end
end
end

function a=read(file,dataset,shape)
shape=double(shape(:).');
a=h5read(file,dataset);
if numel(shape)==1
    a=reshape(a,shape(1),1);
else
    a=permute(reshape(a,fliplr(shape)),numel(shape):-1:1);
end
end
