function [cost,grad] = SingleWordPPHeadDistDropoutUpdateExtCost(theta, inputSize, extDim, beta, p, ...
                                              maxNumHeads, heads, preps, ...
                                              ppChildren, labels, nheads, scaleParents)

% inputSize - size N of input vectors (including extended dimensions)
% extDim - size E of extended dimensions for the input vectors
% beta - loss parameter
% p - dropout parameter
% maxNumHeads - maximum number of candidate heads H
% heads - an N x H x M matrix, where column (:, h, i) is the h-th 
%         candidate head in the i-th training example
% preps - an N x M matrix, where column (:, i) is the vector for the 
%         preposition in the i-th training example
% ppChildren - an N x M matrix, where column (:,i) is the vector for 
%              the direct child of the preposition in the i-th 
%              training example
% labels - an M x 1 matrix containing the labels corresponding for the input data
%          a label is an index in (1:H) corresponding to the gold head
% nheads - an M x 1 matrix containing the number of possible heads for 
%          each training instance
% scaleParents - scale the composite parents by this to match the scaled
%                word vectors

% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% First convert theta to the (W, Wdists, b, bdists, w, ext) matrix/vector format
% W is a N x 2N matrix, Wdists is a numDistances*N x 2N matrix, 
% b is a N x 1 vectors, bdists is a numDistances*N x 1 vector, w is a 1 x N vector
% ext is a E x 1 vector


% This objective function assumes preposition has only one word child and
% therefore no need to score the PP or the NP 
% Heads at different distances are composed with the PP using Wdists
% matrix
% this version updates the extended dimensions of the input vectors

numDistances = 5; % hard coded for now

W = reshape(theta(1:2*inputSize*inputSize), inputSize, 2*inputSize);
Wdists = reshape(theta(2*inputSize*inputSize+1:(numDistances+1)*2*inputSize*inputSize), numDistances*inputSize, 2*inputSize);
b = theta((numDistances+1)*2*inputSize*inputSize+1:(numDistances+1)*2*inputSize*inputSize+inputSize);
bdists = theta((numDistances+1)*2*inputSize*inputSize+inputSize+1:(numDistances+1)*2*inputSize*inputSize+(numDistances+1)*inputSize);
w = theta(end-extDim-inputSize+1:end-extDim);
ext = theta(end-extDim+1:end);
datasize = size(preps, 2);

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
extgrad = zeros(size(ext));


% dropout from initial vectors
preps = preps .* binornd(1, p, size(preps)); 
ppChildren = ppChildren .* binornd(1, p, size(ppChildren)); 
heads = heads .* binornd(1, p, size(heads));
% update with current extended dimensions
prepsExt = preps; ppChildrenExt = ppChildren; headsExt = heads;
prepsExt(end-extDim+1:end, :) = prepsExt(end-extDim+1:end, :) .* repmat(ext, 1, datasize);
ppChildrenExt(end-extDim+1:end, :) = ppChildrenExt(end-extDim+1:end, :) .* repmat(ext, 1, datasize);
headsExt(end-extDim+1:end, :, :) = headsExt(end-extDim+1:end, :, :) .* repmat(ext, [1 maxNumHeads datasize]);


ppParentsBeforeNonlin = CompositionFunction(W, b, inputSize, prepsExt, ppChildrenExt); % compose preposition with its child
ppParentsBeforeNonlin = ppParentsBeforeNonlin .* binornd(1, p, size(ppParentsBeforeNonlin)); % dropout
ppParents = applyNonLinearity(ppParentsBeforeNonlin);
ppParents = scaleParents * ppParents;
headPPParents = zeros(size(headsExt)); % compose candidate head with PP
headPPParentsBeforeNonlin = zeros(size(headsExt)); 
scores = zeros(maxNumHeads, datasize);
losses = 1-groundTruth; % here loss is simply 0/1
for h = 1:maxNumHeads
    dist = min(h, numDistances); % allowd distances are 1:numDistances
    curHeadPPParentsBeforeNonlin = ...
        CompositionFunction(Wdists(1+(dist-1)*inputSize:dist*inputSize,:), ...
                            bdists(1+(dist-1)*inputSize:dist*inputSize), ...
                            inputSize, squeeze(headsExt(:,h,:)), ppParents);
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


startInd = 1 + inputSize*maxNumHeads*[0:datasize-1]' + inputSize*(labels-1);
linIndGold = repmat(startInd', inputSize, 1) + repmat([0:inputSize-1]', 1, datasize);
goldHeadPPParents = headPPParents(linIndGold);
goldHeads = headsExt(linIndGold);
goldHeadPPParentsBeforeNonlin = headPPParentsBeforeNonlin(linIndGold);
linIndGoldUnpadded = linIndGold;
% pad with zeros
startInd = ((min(labels,numDistances)'-1)*inputSize+1)+numDistances*inputSize*[0:datasize-1];
linIndGold = repmat(startInd, inputSize, 1) + repmat([0:inputSize-1]', 1, datasize);
goldHeadPPParentsBeforeNonlinPadded = zeros(numDistances*inputSize, datasize);
goldHeadPPParentsBeforeNonlinPadded(linIndGold) = goldHeadPPParentsBeforeNonlin;

startInd = 1 + inputSize*maxNumHeads*[0:datasize-1]' + inputSize*(maxHeadPPParentsIndices-1);
linIndMax = repmat(startInd', inputSize, 1) + repmat([0:inputSize-1]', 1, datasize);
maxHeadPPParents = headPPParents(linIndMax);
maxHeads = headsExt(linIndMax);
maxHeadPPParentsBeforeNonlin = headPPParentsBeforeNonlin(linIndMax);
linIndMaxUnpadded = linIndMax;
% pad with zeros
startInd = ((min(maxHeadPPParentsIndices,numDistances)'-1)*inputSize+1)+numDistances*inputSize*[0:datasize-1];
linIndMax = repmat(startInd, inputSize, 1) + repmat([0:inputSize-1]', 1, datasize);
maxHeadPPParentsBeforeNonlinPadded = zeros(numDistances*inputSize, datasize);
maxHeadPPParentsBeforeNonlinPadded(linIndMax) = maxHeadPPParentsBeforeNonlin;

wgrad = sum(goldHeadPPParents - maxHeadPPParents, 2);

% backprop

% gold
deltaHeadPPsGold = applyNonLinearityDerivative(goldHeadPPParentsBeforeNonlin) .* repmat(w, 1, datasize);
deltaHeadPPsGoldPadded = zeros(numDistances*inputSize, datasize);
deltaHeadPPsGoldPadded(linIndGold) = deltaHeadPPsGold;
%deltaHeadPPsGoldDown = Wdists' * deltaHeadPPsGoldPadded .* applyNonLinearityDerivative([goldHeads; ppParentsBeforeNonlin]);
%deltaPPsGold = deltaHeadPPsGoldDown(inputSize+1:end, :); % + applyNonLinearityDerivative(ppParentsBeforeNonlin) .* repmat(w,1,datasize); %% consider this addition
WdeltaHeadPPsGold = Wdists' * deltaHeadPPsGoldPadded ;
deltaPPsGold = WdeltaHeadPPsGold(inputSize+1:end, :) .* applyNonLinearityDerivative(ppParentsBeforeNonlin);
deltaHeadsGold = WdeltaHeadPPsGold(1:inputSize, :);
WdeltaPPsGold = W' * deltaPPsGold;
deltaChildrenGold = WdeltaPPsGold(inputSize+1:end, :);
deltaPrepsGold = WdeltaPPsGold(1:inputSize, :);

WgradGold = deltaPPsGold * [prepsExt; ppChildrenExt]';
bgradGold = sum(deltaPPsGold, 2);
WdistsGradGold = deltaHeadPPsGoldPadded * [goldHeads; ppParents]';
bdistsGradGold = sum(deltaHeadPPsGoldPadded, 2);
% ext grad
headsDeltaUpdateGold = heads(linIndGoldUnpadded) .* deltaHeadsGold;
prepsDeltaUpdateGold = preps .* deltaPrepsGold;
childrenDeltaUpdateGold = ppChildren .* deltaChildrenGold;
extgradGold = sum(headsDeltaUpdateGold(end-extDim+1:end,:), 2) + ...
              sum(prepsDeltaUpdateGold(end-extDim+1:end,:), 2) + ...
              sum(childrenDeltaUpdateGold(end-extDim+1:end,:), 2);

% max
deltaHeadPPsMax = applyNonLinearityDerivative(maxHeadPPParentsBeforeNonlin) .* repmat(w, 1, datasize); %%% is this the correct top delta?
deltaHeadPPsMaxPadded = zeros(numDistances*inputSize, datasize);
deltaHeadPPsMaxPadded(linIndMax) = deltaHeadPPsMax;
% deltaHeadPPsMaxDown = Wdists' * deltaHeadPPsMaxPadded .* applyNonLinearityDerivative([maxHeads; ppParentsBeforeNonlin]);
% deltaPPsMax = deltaHeadPPsMaxDown(inputSize+1:end, :); % + applyNonLinearityDerivative(ppParentsBeforeNonlin) .* repmat(w,1,datasize); %% consider this addition;
WdeltaHeadPPsMax = Wdists' * deltaHeadPPsMaxPadded ;
deltaPPsMax = WdeltaHeadPPsMax(inputSize+1:end, :) .* applyNonLinearityDerivative(ppParentsBeforeNonlin);
deltaHeadsMax = WdeltaHeadPPsMax(1:inputSize, :);
WdeltaPPsMax = W' * deltaPPsMax;
deltaChildrenMax = WdeltaPPsMax(inputSize+1:end, :);
deltaPrepsMax = WdeltaPPsMax(1:inputSize, :);

WgradMax = deltaPPsMax * [prepsExt; ppChildrenExt]';
bgradMax = sum(deltaPPsMax, 2);
WdistsGradMax = deltaHeadPPsMaxPadded * [maxHeads; ppParents]';
bdistsGradMax = sum(deltaHeadPPsMaxPadded, 2);
% ext grad
headsDeltaUpdateMax = heads(linIndMaxUnpadded) .* deltaHeadsMax;
prepsDeltaUpdateMax = preps .* deltaPrepsMax;
childrenDeltaUpdateMax = ppChildren .* deltaChildrenMax;
extgradMax = sum(headsDeltaUpdateMax(end-extDim+1:end,:), 2) + ...
              sum(prepsDeltaUpdateMax(end-extDim+1:end,:), 2) + ...
              sum(childrenDeltaUpdateMax(end-extDim+1:end,:), 2);


% grad = gold - max
Wgrad = WgradGold - WgradMax;
bgrad = bgradGold - bgradMax;
%%%%%% is this correct???
WdistsGrad = WdistsGradGold - WdistsGradMax;
bdistsGrad = bdistsGradGold - bdistsGradMax;
extgrad = extgradGold - extgradMax;

% roll grads to one vector grad
grad = [Wgrad(:) ; WdistsGrad(:) ; bgrad(:) ; bdistsGrad(:) ; wgrad(:) ; extgrad(:)];
% normalize by data set size
grad = 1/datasize*grad;


% % flip cost and grad since we're minimizing instead of maximizing
cost = -cost;
grad = -grad;

end




