function theta = initializeParameters(params, model)

inputSize = double(params.inputSize+params.extDim); % add extended dimensions
scaleParam = params.scaleParam;

if model == 6
    theta = initializeParametersHeadDist(inputSize, params.numDistances, scaleParam);
else 
    disp(['Error: unknown model ' num2str(model)]); % TODO change
end

if params.updateExt
    ext = ones(params.extDim, 1);
    theta = [theta(:); ext];
end


end


function theta = initializeParametersHeadDist(inputSize, varargin)

% Define different Ws for composing with heads at different distances,
% and another W for other compositions


%%%% default values %%%%
numvarargs = length(varargin);
if numvarargs > 2
    error('initializeParametersHeadDist:TooManyInputs', ...
        'requires at most 2 optional input');
end

% set defaults for optional inputs 
optargs = {5 1};

% now put these defaults into the valuesToUse cell array, 
% and overwrite the ones specified in varargin.
optargs(1:numvarargs) = varargin;

% Place optional args in memorable variable names
[numDistances scaleParam] = optargs{:};
%%%%%%%%%%%%%%%%%%%%%%%%%


% r = sqrt(6/(2*inputSize + inputSize));
r = 1/sqrt(inputSize*100);

% global W, b
W = 0.5*[eye(inputSize) eye(inputSize)] + (-r + 2*r.*rand(inputSize, 2*inputSize));
% W = 0.5*[eye(inputSize) eye(inputSize)] + (0.001 + 0.002.*rand(inputSize, 2*inputSize));
% b = 0.001 + 0.002.*rand(inputSize,1);
% b = 0.001*ones(inputSize, 1);
b = zeros(inputSize, 1);

Wdists = 0.5 * repmat(eye(inputSize), numDistances, 2) + (-r + 2*r.*rand(numDistances*inputSize, 2*inputSize));
bdists = zeros(numDistances*inputSize, 1);

% w
w = 0.5*(-1/sqrt(inputSize) + 2/sqrt(inputSize).*rand(inputSize,1)); %%% for now   
% w = ones(inputSize, 1) + 0.001 + 0.002 .* rand(inputSize,1);
% w = 1e-3*rand(inputSize,1);


theta = [W(:); Wdists(:); b(:); bdists(:); w(:)];
theta = theta*scaleParam;

end
