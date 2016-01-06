function testModel(opttheta, model, params, wordVectors, trainData, filenames, varargin)

%%%% default values %%%%
numvarargs = length(varargin);
if numvarargs > 2
    error('testModel:TooManyInputs', ...
        'requires at most 2 optional input');
end
if params.useExt && numvarargs ~= 2
    error('testModel:TooFewInputs', ...
        'if useExt=true, must have exactly 2 more inputs (Verbnet, Wordnet)');
end

% set defaults for optional inputs 
optargs = {'' ''};

% now put these defaults into the valuesToUse cell array, 
% and overwrite the ones specified in varargin.
optargs(1:numvarargs) = varargin;

% Place optional args in memorable variable names
[vn wn] = optargs{:};
%%%%%%%%%%%%%%%%%%%%%%%%%  


if params.updateWordVectors
    % replace the wordVectors map with an updated map
    disp('replacing wordVectors with updated ones');
    vocabSize = wordVectors.Count;
    updatedWordVectorsMat = reshape(opttheta(end-(params.inputSize*vocabSize)+1:end), params.inputSize, vocabSize);
    updatedWordVectorsValues = mat2cell(updatedWordVectorsMat, params.inputSize, ones(1,vocabSize));
    updatedWordVectorsValues = cellfun(@transpose, updatedWordVectorsValues, 'UniformOutput', false);
    updatedWordVectors = containers.Map(wordVectors.keys, updatedWordVectorsValues);
    wordVectors = updatedWordVectors;
    % remove the updated word vectors from opttheta
    disp('removing updated word vectors from opttheta');
    opttheta = opttheta(1:end-(params.inputSize*vocabSize));
end

% load test data
if params.useExt
    testData = loadData(model, params, filenames.testFilePref, wordVectors, vn, wn);
else
    testData = loadData(model, params, filenames.testFilePref, wordVectors);
end

[pred] = functionPredict(opttheta, model, params, testData);
acc = mean(testData.labels(:) == pred(:));
fprintf('Test Accuracy: %0.3f%%\n', acc * 100);

% save predictions
labels = testData.labels;
origIndices = testData.includeInd;
save(filenames.predictionsFile, 'labels', 'pred', 'origIndices');
disp(['predictions saved to: ' filenames.predictionsFile]);

% now predict on the training data
if params.updateWordVectors
    if params.useExt
        trainData = loadData(model, params, filenames.trainFilePref, wordVectors, vn, wn);
    else
        trainData = loadData(model, params, filenames.trainFilePref, wordVectors);
    end
end    

[pred] = functionPredict(opttheta, model, params, trainData);
acc = mean(trainData.labels(:) == pred(:));
fprintf('Train Accuracy: %0.3f%%\n', acc * 100);


end
