function saveParameters(model, theta, params, saveFile)

if model == 6
    saveParametersHeadDist(theta, params, saveFile);
else
    error('Error', ['Unknown model ' num2str(model) ' in saveParameters()']);
end

end



function saveParametersHeadDist(theta, params, saveFile)

numDistances = params.numDistances;

inputSize = params.inputSize + params.extDim; % extDim = 0 if not used
W = reshape(theta(1:2*inputSize*inputSize), inputSize, 2*inputSize);
Wdists = reshape(theta(2*inputSize*inputSize+1:(numDistances+1)*2*inputSize*inputSize), numDistances*inputSize, 2*inputSize);
b = theta((numDistances+1)*2*inputSize*inputSize+1:(numDistances+1)*2*inputSize*inputSize+inputSize);
bdists = theta((numDistances+1)*2*inputSize*inputSize+inputSize+1:(numDistances+1)*2*inputSize*inputSize+(numDistances+1)*inputSize);
w = theta((numDistances+1)*2*inputSize*inputSize+(numDistances+1)*inputSize+1:(numDistances+1)*2*inputSize*inputSize+(numDistances+1)*inputSize+inputSize);

if params.updateExt
    ext = theta(end-params.extDim+1:end);
    save(saveFile, 'W', 'Wdists', 'b', 'bdists', 'w', 'ext');    
elseif params.updateWordVectors
    wordVectorsVocabSize = params.vocabSize;
    disp('Warning: saving word vectors as params is not up-to-date!');
    wordVectors = reshape(theta(end-params.inputSize*wordVectorsVocabSize+1:end), params.inputSize, wordVectorsVocabSize);
    save(saveFile, 'W', 'Wdists', 'b', 'bdists', 'w', 'wordVectors');
else
    save(saveFile, 'W', 'Wdists', 'b', 'bdists', 'w');
end


end