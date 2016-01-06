function [pred] = singleWordPPHeadDistDropoutUpdateExtPredict(theta, p, inputSize, extDim, ...
                                              maxNumHeads, heads, preps, ...
                                              ppChildren, nheads, scaleParents)

% inputSize - size N of input vectors (including extended dimensions)
% extDim - size E of extended dimensions for the input vectors
% p - dropout parameter
% maxNumHeads - maximum number of candidate heads H
% heads - an N x H x M matrix, where column (:, h, i) is the h-th 
%         candidate head in the i-th training example
% preps - an N x M matrix, where column (:, i) is the vector for the 
%         preposition in the i-th training example
% ppChildren - an N x M matrix, where column (:,i) is the vector for 
%              the direct child of the preposition in the i-th 
%              training example
% nheads - an M x 1 matrix containing the number of possible heads for 
%          each training instance

% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% First convert theta to the (W, Wdists, b, bdists, w, ext) matrix/vector format
% W is a N x 2N matrix, Wdists is a numDistances*N x 2N matrix, 
% b is a N x 1 vectors, bdists is a numDistances*N x 1 vector, w is a 1 x N vector
% ext is a E x 1 vector


numDistances = 5; % hard coded for now

W = reshape(theta(1:2*inputSize*inputSize), inputSize, 2*inputSize);
Wdists = reshape(theta(2*inputSize*inputSize+1:(numDistances+1)*2*inputSize*inputSize), numDistances*inputSize, 2*inputSize);
b = theta((numDistances+1)*2*inputSize*inputSize+1:(numDistances+1)*2*inputSize*inputSize+inputSize);
bdists = theta((numDistances+1)*2*inputSize*inputSize+inputSize+1:(numDistances+1)*2*inputSize*inputSize+(numDistances+1)*inputSize);
w = theta(end-extDim-inputSize+1:end-extDim);
ext = theta(end-extDim+1:end);
datasize = size(preps, 2);
% scale for dropout
W = p*W;
Wdists = p*Wdists;

% update with current extended dimensions
preps(end-extDim+1:end, :) = preps(end-extDim+1:end, :) .* repmat(ext, 1, datasize);
ppChildren(end-extDim+1:end, :) = ppChildren(end-extDim+1:end, :) .* repmat(ext, 1, datasize);
heads(end-extDim+1:end, :, :) = heads(end-extDim+1:end, :, :) .* repmat(ext, [1 maxNumHeads datasize]);


ppParentsBeforeNonlin = CompositionFunction(W, b, inputSize, preps, ppChildren); % compose preposition with its child
ppParents = applyNonLinearity(ppParentsBeforeNonlin);
ppParents = scaleParents * ppParents;
headPPParents = zeros(size(heads)); % compose candidate head with PP
headPPParentsBeforeNonlin = zeros(size(heads)); 
scores = zeros(maxNumHeads, datasize);
for h = 1:maxNumHeads
    dist = min(h, numDistances); % allowd distances are 1:numDistances
    curHeadPPParentsBeforeNonlin = ...
        CompositionFunction(Wdists(1+(dist-1)*inputSize:dist*inputSize,:), ...
                            bdists(1+(dist-1)*inputSize:dist*inputSize), ...
                            inputSize, squeeze(heads(:,h,:)), ppParents);
    curHeadPPParents = applyNonLinearity(curHeadPPParentsBeforeNonlin);
    headPPParents(:,h,:) = curHeadPPParents;
    headPPParentsBeforeNonlin(:,h,:) = curHeadPPParentsBeforeNonlin;
    scores(h,:) = w'*squeeze(headPPParents(:,h,:));   
end

% for now, loop, later figure out a better way
maxScoresLosses = zeros(datasize, 1);
maxHeadPPParentsIndices = zeros(datasize, 1);
for i = 1:datasize
    curnheads = nheads(i);
    [maxScoresLosses(i), maxHeadPPParentsIndices(i)] = max(scores(1:curnheads,i));
end
% [maxScoresLosses, maxHeadPPParentsIndices] = max(scores + beta*losses);
% maxHeadPPParentsIndices = maxHeadPPParentsIndices';

pred = maxHeadPPParentsIndices;


end
