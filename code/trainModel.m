function opttheta = trainModel(theta, data, wordVectors, params, model, trainParams, filenames)


opttheta = theta;    
datasize = size(data.heads, 3);
batchsize = trainParams.batchsize;
numBatches = floor(datasize/batchsize)+1;        
iters = trainParams.iters; % iterations per batch    
sumSquares = ones(size(theta));
if trainParams.usePretrainedSumSquares
    disp('using pretrained sumSquares');
    s = load(filenames.sumSquaresFile);
    sumSquares = s.sumSquares;
end
% for updating with different learning rates
paramsSize = length(opttheta);
wordVectorsRange = (paramsSize-params.inputSize*wordVectors.Count+1:paramsSize);
otherParamsRange = (1:paramsSize-params.inputSize*wordVectors.Count);

bestTheta = opttheta;
[bestCost, bestGrad] = functionCostGrad(opttheta, model, params, data); 
disp(['initial cost: ' num2str(bestCost)]);
tic;
for t = 1:trainParams.epochs        
    if ~trainParams.quiet
        disp(['epoch ' num2str(t)]);
    end
    rp = randperm(datasize);
    data = getDataSubind(data, model, rp, params);
    for i = 1:numBatches
        % startIdx = mod((i-1) * batchsize, datasize) + 1;
        startIdx = (i-1)*batchsize + 1;
        if startIdx > datasize
            continue;
        end
        endIdx = min(startIdx + batchsize - 1, datasize);
        % disp(['minibatch start: ' num2str(startIdx) ' end: ' num2str(endIdx)]);
        curData = getDataSubind(data, model, startIdx:endIdx, params);
        if strcmpi(trainParams.trainingMethod, 'adagrad')                 
            for j = 1:iters
                learningRate = trainParams.initLearningRate;
                [cost, grad] = functionCostGrad(opttheta, model, params, curData);
                if params.updateWordVectors && params.updateWordVectorsRate > 1  
                    % if word vectors should be updated based on a different rate
                    % update other params
                    sumSquares(otherParamsRange) = sumSquares(otherParamsRange) + ...
                        grad(otherParamsRange).*grad(otherParamsRange);
                    opttheta(otherParamsRange) = opttheta(otherParamsRange) - ...
                        (learningRate * 1./sqrt(sumSquares(otherParamsRange)) .* grad(otherParamsRange));
                    % update word vectors
                    sumSquares(wordVectorsRange) = sumSquares(wordVectorsRange) ...
                        + grad(wordVectorsRange).*grad(wordVectorsRange);
                    opttheta(wordVectorsRange) = opttheta(wordVectorsRange) - ...
                        (learningRate/params.updateWordVectorsRate * 1./sqrt(sumSquares(wordVectorsRange)) .* grad(wordVectorsRange));
                else % word vectors updated the same as other parameters 
                    sumSquares = sumSquares + grad.*grad;
                    opttheta = opttheta - (learningRate * 1./sqrt(sumSquares) .* grad);
                end
            end
        else
            disp(['Error: unknown optimization method ' trainParams.trainingMethod]);
            disp('This version only supports adagrad optimization');
            return;
        end
        % disp(['cost ' num2str(cost)]);
    end % end batch loop        
    [totalCost, totalGrad] = functionCostGrad(opttheta, model, params, data);
    if ~trainParams.quiet
        disp(['total cost ' num2str(totalCost)]);
    end
    if totalCost < bestCost
        bestTheta = opttheta;
        bestCost = totalCost;
        bestGrad = totalGrad;
    end
end % end epoch loop
disp('training elapsed time:');
toc;

if trainParams.backpocket
    opttheta = bestTheta;
    cost = bestCost; 
    grad = bestGrad;
end
disp(['final cost ' num2str(cost)]);

% save parameters
saveParamsFile = filenames.saveParamsFile; 
saveParameters(model, opttheta, params, saveParamsFile); % add extended dimensions
disp(['opt params saved to ' saveParamsFile]);
if params.updateWordVectors
    vocabSize = wordVectors.Count;
    updatedWordVectorsMat = reshape(opttheta(end-(params.inputSize*vocabSize)+1:end), params.inputSize, vocabSize);
    updatedWordVectorsWords = wordVectors.keys;
    saveUpdatedWordVectorsFile = filenames.saveUpdatedWordVectorsFile; 
    save(saveUpdatedWordVectorsFile, 'updatedWordVectorsWords', 'updatedWordVectorsMat');
    disp(['word vectors saved to ' saveUpdatedWordVectorsFile]);
end
if strcmpi(trainParams.trainingMethod, 'adagrad')
    saveSumSquaresFile = filenames.saveSumSquaresFile; 
    save(saveSumSquaresFile, 'sumSquares');
    disp(['last sum of squares (adagrad) saved to ' saveSumSquaresFile]);
end


end


function indData = getDataSubind(data, model, ind, params)

% ind - index for subset of the data (can be all data with permuted indices)

indData.heads = data.heads(:,:,ind); indData.labels = data.labels(ind); indData.nheads = data.nheads(ind); 
indData.includeInd = data.includeInd(ind);
if model == 6 
    indData.preps = data.preps(:,ind); indData.ppChildren = data.ppChildren(:,ind);
    if params.updateWordVectors
        indData.indPrepsToWordVectors = data.indPrepsToWordVectors(:,ind); indData.indHeadsToWordVectors = data.indHeadsToWordVectors(:,:,ind); indData.indChildrenToWordVectors = data.indChildrenToWordVectors(:,ind);
        indData.wordVectorsMat = data.wordVectorsMat;
    end
else  % TODO change?
    error('Error', ['unknown model ' num2str(model) ' in trainModel.m']);
end

    
end



