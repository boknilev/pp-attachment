function [wordVectorsMat, indHeadsToWordVectors, indPrepsToWordVectors, indChildrenToWordVectors, labels, nheads] = ...
        loadDataSingleWordPPIndices(wordVectors, ...
                                    inputSize, maxNumHeads, ...
                                    headWordsFilename, prepWordsFilename, ...
                                    ppChildWordsFilename, labelsFilename, ...
                                    nheadsFilename, scaleVectors)

% wordVectors - Map of word string to vector                        
                        
% load data for the model assuming preposition has only a single child                        
% this version will create indices to be used when
% updating word vectors during training
                        
                        
headLines = loadLinesFromFile(headWordsFilename);
prepWords = loadLinesFromFile(prepWordsFilename);
ppChildWords = loadLinesFromFile(ppChildWordsFilename);

datasize = size(headLines, 1);
if size(prepWords, 1) ~= datasize || size(ppChildWords, 1) ~= datasize
    disp('Error: incompatible sizes')
end

% first build the word vectors matrix and word index
vocabSize = wordVectors.Count;
wordVectorsMat = cell2mat(wordVectors.values')';
% disp(['wordVectorsMat size ' num2str(size(wordVectorsMat))]);
mapWordToVecInd = containers.Map(wordVectors.keys, 1:vocabSize);


preps = zeros(inputSize, datasize);
heads = zeros(inputSize, maxNumHeads, datasize);
ppChildren = zeros(inputSize, datasize);
includeInd = [];
indHeadsToWordVectors = [];
indPrepsToWordVectors = [];
indChildrenToWordVectors = [];
countIndcluded = 0;
for i = 1:datasize
    prepWord = prepWords{i};
    childWord = ppChildWords{i};
    if isKey(wordVectors, prepWord) && isKey(wordVectors, childWord) 
        % only consider examples which have word vectors
        headLine = headLines(i);
        curHeadWords = regexp(headLine, '\s+', 'split');
        curHeadWords = curHeadWords{1};
        curHeads = zeros(inputSize, maxNumHeads);
        numExistingHeads = size(curHeadWords,2);
        missingHeadVector = false;
        for j = 1:numExistingHeads
            curHeadWord = curHeadWords{j};
            if isKey(wordVectors, curHeadWord)
                curHead = wordVectors(curHeadWord);
                curHeads(:,j) = curHead;
            else
                missingHeadVector = true;
            end
        end
        
        if ~missingHeadVector
            countIndcluded = countIndcluded + 1;
            % all words (heads, prep, child) have vectors, so can add
            includeInd = [includeInd; i];
            preps(:, i) = wordVectors(prepWord);
            ppChildren(:, i) = wordVectors(childWord);
            heads(:,:,i) = curHeads;            
            prepVecInd = mapWordToVecInd(prepWord);
            indPrepsToWordVectors(:,countIndcluded) = ((prepVecInd-1)*inputSize+1:prepVecInd*inputSize)';
            childVecInd = mapWordToVecInd(childWord);
            indChildrenToWordVectors(:,countIndcluded) = ((childVecInd-1)*inputSize+1:childVecInd*inputSize)';
            curHeadsVecInd = zeros(inputSize, maxNumHeads);
            for j = 1:numExistingHeads
                curHeadWord = curHeadWords{j};
                curHeadVecInd = mapWordToVecInd(curHeadWord);
                curHeadsVecInd(:,j) = ((curHeadVecInd-1)*inputSize+1:curHeadVecInd*inputSize)';
            end
            indHeadsToWordVectors(:,:,countIndcluded) = curHeadsVecInd;
        end
    end    
end

labels = load(labelsFilename);
nheads = load(nheadsFilename);
% take only included instances
labels = labels(includeInd);
nheads = nheads(includeInd);
preps = preps(:,includeInd);
ppChildren = ppChildren(:,includeInd);
heads = heads(:,:,includeInd);

% scale
preps = scaleVectors*preps;
heads = scaleVectors*heads;
ppChildren = scaleVectors*ppChildren;

end

