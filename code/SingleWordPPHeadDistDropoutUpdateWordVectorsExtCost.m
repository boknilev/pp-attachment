function [cost,grad] = SingleWordPPHeadDistDropoutUpdateWordVectorsExtCost(theta, inputSize, extDim, beta, p, ...
                                          maxNumHeads, labels, nheads, ... 
                                          scaleParents, wordVectorsVocabSize, ...
                                          indPrepsToWordVectors, indHeadsToWordVectors, indChildrenToWordVectors, ...
                                          prepsExt, headsExt, ppChildrenExt)

% inputSize - size N of input vectors
% extDim - size E of extended dimensions
% beta - loss parameter
% p - dropout parameter
% maxNumHeads - maximum number of candidate heads H
% scaleParents - scale the composite parents by this to match the scaled
%                word vectors
% updateWordVectors - boolean indicating whether word vectors will also be
%                     updated. If yes, following arguments are given:
% wordVectorsVocabSize - number of in-vocab word vectors V
% indPrepToWordVector - a N x M matrix, where column (:,i) is the index of
%                       the preposition of the i-th example in the gradWordVectors matrix
% indChildToWordVector - a N x M matrix, where column (:,i) is the index of
%                       the child of the i-th example in the gradWordVectors matrix
% indHeadToWordVector - a N x H x M matrix, where column (:,h,i) is the
%                       index of the h-th candidate head of the i-th example in the
%                       gradWordVectors matrix
% labels - an M x 1 matrix containing the labels for the input data
%          a label is an index in (1:H) corresponding to the gold head
% nheads - an M x 1 matrix containing the number of possible heads for 
%          each training instance
% prepsExt - an E x M matrix, where column (:,i) is the extended vector for
%            the preposition in the i-th training example
% ppChildrenExt - an E x M matrix, where column (:,i) is the extended
%                 vector for the direct child of the preposition in the
%                 i-th training example
% headsExt - an E x H x M matrix, where column (:,h,i) is the extended
%            vector for the h-th candidate head in the i-th training
%            example
% The following will be built from the parameters:
% heads - an (N + E) x H x M matrix, where column (:, h, i) is the h-th 
%         candidate head in the i-th training example
% preps - an (N + E) x M matrix, where column (:, i) is the vector for the 
%         preposition in the i-th training example
% ppChildren - an (N + E) x M matrix, where column (:,i) is the vector for 
%              the direct child of the preposition in the i-th 
%              training example

% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% First convert theta to the (W, Wdists, b, bdists, w, wordVectors) matrix/vector format
% W is a (N+H) x 2(N+H) matrix, Wdists is a numDistances(N+H) x 2(N+H) matrix, 
% b is a (N+H) x 1 vectors, bdists is a numDistances(N+H) x 1 vector, w is
% a 1 x (N+H) vector, wordVectors is a N x V matrix


% This objective function assumes preposition has only one word child and
% therefore no need to score the PP or the NP 
% Heads at different distances are composed with the PP using Wdists
% matrix
% dropout version
% update original word vectors and include extended dimensions (w/o
% updating them)

numDistances = 5; % hard coded for now

% unroll parameters
inputPlusExtSize = inputSize+extDim;
W = reshape(theta(1:2*inputPlusExtSize*inputPlusExtSize), inputPlusExtSize, 2*inputPlusExtSize);
Wdists = reshape(theta(2*inputPlusExtSize*inputPlusExtSize+1:(numDistances+1)*2*inputPlusExtSize*inputPlusExtSize), ...
                 numDistances*inputPlusExtSize, 2*inputPlusExtSize);
b = theta((numDistances+1)*2*inputPlusExtSize*inputPlusExtSize+1:(numDistances+1)*2*inputPlusExtSize*inputPlusExtSize+inputPlusExtSize);
bdists = theta((numDistances+1)*2*inputPlusExtSize*inputPlusExtSize+inputPlusExtSize+1:(numDistances+1)*2*inputPlusExtSize*inputPlusExtSize+(numDistances+1)*inputPlusExtSize);
w = theta((numDistances+1)*2*inputPlusExtSize*inputPlusExtSize+(numDistances+1)*inputPlusExtSize+1:(numDistances+1)*2*inputPlusExtSize*inputPlusExtSize+(numDistances+1)*inputPlusExtSize+inputPlusExtSize);
wordVectors = reshape(theta(end-inputSize*wordVectorsVocabSize+1:end), inputSize, wordVectorsVocabSize);
preps = [wordVectors(indPrepsToWordVectors); prepsExt];
ppChildren = [wordVectors(indChildrenToWordVectors); ppChildrenExt];
datasize = size(preps, 2);
heads = zeros(inputSize, maxNumHeads, datasize);
indNonzeroHeadIndices = find(indHeadsToWordVectors);
heads(indNonzeroHeadIndices) = wordVectors(indHeadsToWordVectors(indNonzeroHeadIndices));
heads = [heads; headsExt];

groundTruth = full(sparse(labels, 1:datasize, 1));
if size(groundTruth, 1) < maxNumHeads  % complete zero rows if needed (not all possible head values seen in labels)
    groundTruth = [groundTruth; zeros(maxNumHeads-size(groundTruth,1), datasize)];
end

cost = 0;
Wgrad = zeros(size(W));
WdistsGrad = zeros(size(Wdists));
bgrad = zeros(size(b));
bdistsGrad = zeros(size(bdists));
wgrad = zeros(size(w));
wordVectorsGrad = zeros(size(wordVectors));

% dropout from initial vectors
preps = preps .* binornd(1, p, size(preps)); 
ppChildren = ppChildren .* binornd(1, p, size(ppChildren)); 
heads = heads .* binornd(1, p, size(heads));


ppParentsBeforeNonlin = CompositionFunction(W, b, inputPlusExtSize, preps, ppChildren); % compose preposition with its child
ppParentsBeforeNonlin = ppParentsBeforeNonlin .* binornd(1, p, size(ppParentsBeforeNonlin)); % dropout
ppParents = applyNonLinearity(ppParentsBeforeNonlin);
ppParents = scaleParents * ppParents;
headPPParents = zeros(size(heads)); % compose candidate head with PP
headPPParentsBeforeNonlin = zeros(size(heads)); 
scores = zeros(maxNumHeads, datasize);
losses = 1-groundTruth; % here loss is simply 0/1
for h = 1:maxNumHeads
    dist = min(h, numDistances); % allowd distances are 1:numDistances
    curHeadPPParentsBeforeNonlin = ...
        CompositionFunction(Wdists(1+(dist-1)*inputPlusExtSize:dist*inputPlusExtSize,:), ...
                            bdists(1+(dist-1)*inputPlusExtSize:dist*inputPlusExtSize), ...
                            inputPlusExtSize, squeeze(heads(:,h,:)), ppParents);
    headPPParents(:,h,:) = applyNonLinearity(curHeadPPParentsBeforeNonlin);
    headPPParentsBeforeNonlin(:,h,:) = curHeadPPParentsBeforeNonlin;
    scores(h,:) = w'*squeeze(headPPParents(:,h,:));   
end

% for now, loop, later figure out a better way
maxScoresLosses = zeros(datasize, 1);
maxHeadPPParentsIndices = zeros(datasize, 1);
for i = 1:datasize
    curnheads = nheads(i);
    [maxScoresLosses(i), maxHeadPPParentsIndices(i)] = max(scores(1:curnheads,i) + beta*losses(1:curnheads,i));
    % make sure score of fake heads (indices more than nheads) are zero
    for j = curnheads+1:maxNumHeads
        scores(j,i) = 0;
    end
end
% [maxScoresLosses, maxHeadPPParentsIndices] = max(scores + beta*losses);
% maxHeadPPParentsIndices = maxHeadPPParentsIndices';

% score correct heads
cost = cost + sum(sum(groundTruth .* scores));
% substract maximal scores
cost = cost - sum(maxScoresLosses);
% normalize by data set size
cost = 1/datasize*cost;


startInd = 1 + inputPlusExtSize*maxNumHeads*[0:datasize-1]' + inputPlusExtSize*(labels-1);
linIndGold = repmat(startInd', inputPlusExtSize, 1) + repmat([0:inputPlusExtSize-1]', 1, datasize);
goldHeadPPParents = headPPParents(linIndGold);
goldHeads = heads(linIndGold);
goldHeadPPParentsBeforeNonlin = headPPParentsBeforeNonlin(linIndGold);
% pad with zeros
startInd = ((min(labels,numDistances)'-1)*inputPlusExtSize+1)+numDistances*inputPlusExtSize*[0:datasize-1];
linIndGoldPadded = repmat(startInd, inputPlusExtSize, 1) + repmat([0:inputPlusExtSize-1]', 1, datasize);
goldHeadPPParentsBeforeNonlinPadded = zeros(numDistances*inputPlusExtSize, datasize);
goldHeadPPParentsBeforeNonlinPadded(linIndGoldPadded) = goldHeadPPParentsBeforeNonlin;

startInd = 1 + inputPlusExtSize*maxNumHeads*[0:datasize-1]' + inputPlusExtSize*(maxHeadPPParentsIndices-1);
linIndMax = repmat(startInd', inputPlusExtSize, 1) + repmat([0:inputPlusExtSize-1]', 1, datasize);
maxHeadPPParents = headPPParents(linIndMax);
maxHeads = heads(linIndMax);
maxHeadPPParentsBeforeNonlin = headPPParentsBeforeNonlin(linIndMax);
% pad with zeros
startInd = ((min(maxHeadPPParentsIndices,numDistances)'-1)*inputPlusExtSize+1)+numDistances*inputPlusExtSize*[0:datasize-1];
linIndMaxPadded = repmat(startInd, inputPlusExtSize, 1) + repmat([0:inputPlusExtSize-1]', 1, datasize);
maxHeadPPParentsBeforeNonlinPadded = zeros(numDistances*inputPlusExtSize, datasize);
maxHeadPPParentsBeforeNonlinPadded(linIndMaxPadded) = maxHeadPPParentsBeforeNonlin;

wgrad = sum(goldHeadPPParents - maxHeadPPParents, 2);

% backprop

% gold
deltaHeadPPsGold = applyNonLinearityDerivative(goldHeadPPParentsBeforeNonlin) .* repmat(w, 1, datasize);
deltaHeadPPsGoldPadded = zeros(numDistances*inputPlusExtSize, datasize);
deltaHeadPPsGoldPadded(linIndGoldPadded) = deltaHeadPPsGold;
WdeltaHeadPPsGold = Wdists' * deltaHeadPPsGoldPadded ;
deltaPPsGold = WdeltaHeadPPsGold(inputPlusExtSize+1:end, :) .* applyNonLinearityDerivative(ppParentsBeforeNonlin);
% we'll only update the original vectors w/o extended dimensions
deltaHeadsGold = WdeltaHeadPPsGold(1:inputSize, :); 
WdeltaPPsGold = W' * deltaPPsGold;
deltaChildrenGold = WdeltaPPsGold(inputPlusExtSize+1:end-extDim, :); 
deltaPrepsGold = WdeltaPPsGold(1:inputSize, :);

WgradGold = deltaPPsGold * [preps; ppChildren]';
bgradGold = sum(deltaPPsGold, 2);
WdistsGradGold = deltaHeadPPsGoldPadded * [goldHeads; ppParents]';
bdistsGradGold = sum(deltaHeadPPsGoldPadded, 2);

% max
deltaHeadPPsMax = applyNonLinearityDerivative(maxHeadPPParentsBeforeNonlin) .* repmat(w, 1, datasize); %%% is this the correct top delta?
deltaHeadPPsMaxPadded = zeros(numDistances*inputPlusExtSize, datasize);
deltaHeadPPsMaxPadded(linIndMaxPadded) = deltaHeadPPsMax;
WdeltaHeadPPsMax = Wdists' * deltaHeadPPsMaxPadded ;
deltaPPsMax = WdeltaHeadPPsMax(inputPlusExtSize+1:end, :) .* applyNonLinearityDerivative(ppParentsBeforeNonlin);
% we'll only update the original vectors w/o extended dimensions
deltaHeadsMax = WdeltaHeadPPsMax(1:inputSize, :);
WdeltaPPsMax = W' * deltaPPsMax;
deltaChildrenMax = WdeltaPPsMax(inputPlusExtSize+1:end-extDim, :);
deltaPrepsMax = WdeltaPPsMax(1:inputSize, :);

WgradMax = deltaPPsMax * [preps; ppChildren]';
bgradMax = sum(deltaPPsMax, 2);
WdistsGradMax = deltaHeadPPsMaxPadded * [maxHeads; ppParents]';
bdistsGradMax = sum(deltaHeadPPsMaxPadded, 2);


% grad = gold - max
Wgrad = WgradGold - WgradMax;
bgrad = bgradGold - bgradMax;
WdistsGrad = WdistsGradGold - WdistsGradMax;
bdistsGrad = bdistsGradGold - bdistsGradMax;

% word vectors grad
% find gold/max indices for original word vectors (w/o extended dimensions)
startInd = 1 + inputSize*maxNumHeads*[0:datasize-1]' + inputSize*(labels-1);
linIndGold = repmat(startInd', inputSize, 1) + repmat([0:inputSize-1]', 1, datasize);
startInd = 1 + inputSize*maxNumHeads*[0:datasize-1]' + inputSize*(maxHeadPPParentsIndices-1);
linIndMax = repmat(startInd', inputSize, 1) + repmat([0:inputSize-1]', 1, datasize);
% calculate gradients
wordVectorsGrad(indHeadsToWordVectors(linIndGold)) = wordVectorsGrad(indHeadsToWordVectors(linIndGold)) + deltaHeadsGold;
wordVectorsGrad(indHeadsToWordVectors(linIndMax)) = wordVectorsGrad(indHeadsToWordVectors(linIndMax)) - deltaHeadsMax;
wordVectorsGrad(indChildrenToWordVectors) = wordVectorsGrad(indChildrenToWordVectors) + deltaChildrenGold - deltaChildrenMax;    
wordVectorsGrad(indPrepsToWordVectors) = wordVectorsGrad(indPrepsToWordVectors) + deltaPrepsGold - deltaPrepsMax;

% roll grads to one vector grad
grad = [Wgrad(:) ; WdistsGrad(:) ; bgrad(:) ; bdistsGrad(:) ; wgrad(:); wordVectorsGrad(:)];
% normalize by data set size
grad = 1/datasize*grad;

% % flip cost and grad since we're minimizing instead of maximizing
cost = -cost;
grad = -grad;

end




