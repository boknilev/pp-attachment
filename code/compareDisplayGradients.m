function compareDisplayGradients(numgrad, grad, model, params)

inputSize = params.inputSize+params.extDim;
extDim = params.extDim;

diff = norm(numgrad-grad)/norm(numgrad+grad); % 
disp(['all ' num2str(diff)]); % Should be small. param: all
if model == 6
    diff = norm(numgrad(1:2*inputSize*inputSize)-grad(1:2*inputSize*inputSize))/norm(numgrad(1:2*inputSize*inputSize)+grad(1:2*inputSize*inputSize));
    disp(['W ' num2str(diff)]); % param: W
    diff = norm(numgrad(2*inputSize*inputSize+1:(params.numDistances+1)*2*inputSize*inputSize)-grad(2*inputSize*inputSize+1:(params.numDistances+1)*2*inputSize*inputSize))...
        /norm(numgrad(2*inputSize*inputSize+1:(params.numDistances+1)*2*inputSize*inputSize)+grad(2*inputSize*inputSize+1:(params.numDistances+1)*2*inputSize*inputSize));
    disp(['Wdists ' num2str(diff)]); % param: Wdists
    diff = norm(numgrad((params.numDistances+1)*2*inputSize*inputSize+1:(params.numDistances+1)*2*inputSize*inputSize+inputSize)-grad((params.numDistances+1)*2*inputSize*inputSize+1:(params.numDistances+1)*2*inputSize*inputSize+inputSize))...
        /norm(numgrad((params.numDistances+1)*2*inputSize*inputSize+1:(params.numDistances+1)*2*inputSize*inputSize+inputSize)+grad((params.numDistances+1)*2*inputSize*inputSize+1:(params.numDistances+1)*2*inputSize*inputSize+inputSize));
    disp(['b ' num2str(diff)]); % param: b
    diff = norm(numgrad((params.numDistances+1)*2*inputSize*inputSize+inputSize+1:(params.numDistances+1)*2*inputSize*inputSize+(params.numDistances+1)*inputSize)...
            -grad((params.numDistances+1)*2*inputSize*inputSize+inputSize+1:(params.numDistances+1)*2*inputSize*inputSize+(params.numDistances+1)*inputSize))...
            /norm(numgrad((params.numDistances+1)*2*inputSize*inputSize+inputSize+1:(params.numDistances+1)*2*inputSize*inputSize+(params.numDistances+1)*inputSize)...
            +grad((params.numDistances+1)*2*inputSize*inputSize+inputSize+1:(params.numDistances+1)*2*inputSize*inputSize+(params.numDistances+1)*inputSize));
    disp(['bdists ' num2str(diff)]); % param: bdists        
else
    disp(['Warning: Unkown model ' num2str(model) ' in compareDisplayGradients()']);
end

if params.updateExt
    diff = norm(numgrad(end-extDim-inputSize+1:end-extDim)-grad(end-extDim-inputSize+1:end-extDim))/norm(numgrad(end-extDim-inputSize+1:end-extDim)+grad(end-extDim-inputSize+1:end-extDim));
    disp(['w ' num2str(diff)]); % param: w
    diff = norm(numgrad(end-extDim+1:end)-grad(end-extDim+1:end))/norm(numgrad(end-extDim+1:end)+grad(end-extDim+1:end));
    disp(['ext ' num2str(diff)]); % param: w    
else
    diff = norm(numgrad(end-inputSize+1:end)-grad(end-inputSize+1:end))/norm(numgrad(end-inputSize+1:end)+grad(end-inputSize+1:end));
    disp(['w ' num2str(diff)]); % param: w
end

