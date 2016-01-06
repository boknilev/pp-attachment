function parents = CompositionFunction(W, b, inputSize, firstChildData, secondChildData)

% inputSize - size N of input vectors
% W - an N x 2N weight matrix
% b - a N x 1 bias vector
% firstChildData - an N x M matrix, where the (:,i) column contains the 
%                  i-th child
% secondChildData - an N x M matrix, where the (:,i) column contains the 
%                  i-th child

% returns an N x M matrix of the composite parents, NOT applying
% non-linearity

datasize = size(firstChildData,2);
concatChildren = [firstChildData; secondChildData];
parents = W*concatChildren + repmat(b,1,datasize);

end


