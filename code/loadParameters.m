function theta = loadParameters(paramsFile, model)

params = load(paramsFile);

if model == 6 
    theta = [params.W(:); params.Wdists(:); params.b(:); params.bdists(:); params.w(:)];
else 
    disp(['Error: unknown model ' num2str(model) ' in loadParameters()']); % TODO change
end

if isfield(params, 'ext')
    theta = [theta(:); params.ext(:)];
end



end