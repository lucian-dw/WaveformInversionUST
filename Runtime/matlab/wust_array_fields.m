function fields = wust_array_fields(schema)
% Generated from Runtime/python/contracts.py; one field table for both runtimes.
persistent definitions
if isempty(definitions),definitions=struct;end
key=matlab.lang.makeValidName(schema);
if ~isfield(definitions,key)
    root=fileparts(fileparts(mfilename('fullpath')));
    doc=jsondecode(fileread(fullfile(root,'schemas',[schema '.json'])));
    definitions.(key)=doc.array_fields;
end
fields=definitions.(key);
end
