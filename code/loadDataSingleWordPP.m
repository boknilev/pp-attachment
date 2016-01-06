function [heads, preps, ppChildren, labels, nheads, includeInd] = loadDataSingleWordPP(wordVectors, ...
                            inputSize, maxNumHeads, ...
                            headWordsFilename, prepWordsFilename, ...
                            ppChildWordsFilename, labelsFilename, ...
                            nheadsFilename, scaleVectors, ...
                            useExt, varargin)
                        
%%%% default values %%%%
numvarargs = length(varargin);
if numvarargs > 6
    error('loadDataSingleWordPP:TooManyInputs', ...
        'requires at most 6 optional input');
end
if useExt && numvarargs ~= 6
    error('loadDataSingleWordPP:TooFewInputs', ...
        'if useExt=true, must have exactly 6 more inputs');
end

% set defaults for optional inputs 
optargs = {'' '' '' '' '' ''};

% now put these defaults into the valuesToUse cell array, 
% and overwrite the ones specified in varargin.
optargs(1:numvarargs) = varargin;

% Place optional args in memorable variable names
[headsPosFilename headsNextPosFilename pos2idx vn wn, language] = optargs{:};
%%%%%%%%%%%%%%%%%%%%%%%%%                        


% wordVectors - Map of word string to vector                    
% useExt - if true, load extended vector dimensions
% pos2idx - map from pos to index (to be used in one-hot vector)
                        
% load data for the model assuming preposition has only a single child
                        
headLines = loadLinesFromFile(headWordsFilename);
prepWords = loadLinesFromFile(prepWordsFilename);
ppChildWords = loadLinesFromFile(ppChildWordsFilename);
extDim = 0;
if useExt
    headPosLines = loadLinesFromFile(headsPosFilename);
    headNextPosLines = loadLinesFromFile(headsNextPosFilename);    
%     extDim = 1 + pos2idx.Count + 1 + 1 + avn.verbclass2idx.Count + awn.hypernym2idx.Count; % head pos, next pos, verb-prep, verbalnoun-prep, verbclass, word hypernym %%% TODO add dims
%     hypOffset = 1 + pos2idx.Count + 1 + 1 + avn.verbclass2idx.Count;
    if strcmpi(language, 'arabic')
        extDim = 2 + pos2idx.Count + 1 + 1 + wn.hypernym2idx.Count; % head pos, next pos, verb-prep, verbalnoun-prep, word hypernym 
        hypOffset = 2 + pos2idx.Count + 1 + 1;
    elseif strcmpi(language, 'english')
        extDim = 2 + pos2idx.Count + 1 + wn.nounhypernym2idx.Count; % head pos, next pos, verb-prep, noun hypernym 
        hypOffset = 2 + pos2idx.Count + 1;
    end        
end

datasize = size(headLines, 1);
if size(prepWords, 1) ~= datasize || size(ppChildWords, 1) ~= datasize
    disp('Error: incompatible sizes')
end

preps = zeros(inputSize+extDim, datasize);
heads = zeros(inputSize+extDim, maxNumHeads, datasize);
ppChildren = zeros(inputSize+extDim, datasize);
includeInd = [];
for i = 1:datasize
    prepWord = prepWords{i};
    childWord = ppChildWords{i};
    if isKey(wordVectors, prepWord) && isKey(wordVectors, childWord) 
        % only consider examples which have word vectors
        headLine = headLines(i);
        curHeadWords = regexp(headLine, '\s+', 'split');
        curHeadWords = curHeadWords{1};
        curHeads = zeros(inputSize+extDim, maxNumHeads);
        numExistingHeads = size(curHeadWords,2);
        if useExt
            headPosLine = headPosLines(i);
            curHeadPos = regexp(headPosLine, '\s+', 'split');
            curHeadPos = str2double(curHeadPos{1});
            headNextPosLine = headNextPosLines(i);
            curHeadNextPos = regexp(headNextPosLine, '\s+', 'split');
            curHeadNextPos = curHeadNextPos{1};
        end
        missingHeadVector = false;
        for j = 1:numExistingHeads
            curHeadWord = curHeadWords{j};
            if isKey(wordVectors, curHeadWord)
                curHead = wordVectors(curHeadWord);
                if useExt
                    headExt = getHeadExt(j, curHeadPos, curHeadNextPos, ...
                                            pos2idx, vn, wn, curHeadWord, prepWord, language);
                    if size(headExt, 2) < extDim
                        headExt = [headExt zeros(1, extDim-size(headExt, 2))];
                    end
                    curHead = [curHead headExt];
                end
                curHeads(:,j) = curHead;
            else
                missingHeadVector = true;
            end
        end
        for j = numExistingHeads+1:maxNumHeads
            curHeads(:,j) = zeros(inputSize+extDim,1);
        end
        
        if ~missingHeadVector
            % all words (heads, preps, children) have vectors, so can add
            includeInd = [includeInd; i];
            prep = wordVectors(prepWord);
            child = wordVectors(childWord);
            if useExt
                prepExt = getPrepExt(extDim); 
                prep = [prep prepExt]; 
                childExt = getChildExt(extDim, wn, childWord, hypOffset, language);
                child = [child childExt];
            end
            preps(:, i) = prep;
            ppChildren(:, i) = child;
            heads(:,:,i) = curHeads;
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



function headExt = getHeadExt(j, curHeadPos, curHeadNextPos, pos2idx, vn, wn, curHeadWord, prepWord, language)
    % get extended dimensions for head
    headExt = [];
    % pos
    pos = curHeadPos(j);
    posVec = [0 0];
    if pos == 1; posVec(1) = 1; else posVec(2) = 1; end                    
    headExt = [headExt posVec];
    % next pos
    nextPosVec = zeros(1, pos2idx.Count); % one hot vector
    if j <= size(curHeadNextPos, 2)
        nextPos = curHeadNextPos{j};
        if isKey(pos2idx, nextPos)
            nextPosVec(pos2idx(nextPos)) = 1;                         
        end
    end
    headExt = [headExt nextPosVec]; 
    % verb-prep
    verbPrep = 0;
    if pos == 1 && isKey(vn.verb2prep, curHeadWord)
        if ismember(prepWord, vn.verb2prep(curHeadWord))
            verbPrep = 1;
    %                         else
    %                             verbPrep = -1;
        end
    end
    headExt = [headExt verbPrep];
    if strcmpi(language, 'arabic')
        % verbalnoun-prep
        verbalnounPrep = 0;
        if pos == -1 && isKey(vn.verbalnoun2prep, curHeadWord)
            if ismember(prepWord, vn.verbalnoun2prep(curHeadWord))
                verbalnounPrep = 1;
        %                         else
        %                             verbalnounPrep = -1;
            end
        end
        headExt = [headExt verbalnounPrep];
    end
    %                     % verb class
    %                     verbClassVec = zeros(1, vn.verbclass2idx.Count); % one-hot vector
    %                     if pos == 1 && isKey(vn.verb2class, curHeadWord)
    %                         curVerbClasses = vn.verb2class(curHeadWord);
    %                         for c = 1:size(curVerbClasses, 2)
    %                             verbClassVec(vn.verbclass2idx(curVerbClasses{c})) = 1;
    %                         end
    %                     end
    %                     curHead = [curHead verbClassVec];
    % head top hypernym
    if strcmpi(language, 'arabic')
        hypernymVec = zeros(1, wn.hypernym2idx.Count);
        if isKey(wn.word2tophypernym, curHeadWord)
            curWordHypernyms = wn.word2tophypernym(curHeadWord);
            for h = 1:size(curWordHypernyms, 2)
                hypernymVec(wn.hypernym2idx(curWordHypernyms{h})) = 1;
            end
        end
    elseif strcmpi(language, 'english')  
        hypernymVec = zeros(1, wn.nounhypernym2idx.Count);
        if pos == -1 && isKey(wn.noun2tophypernym, curHeadWord) % for now use only noun hypernyms
            curWordHypernyms = wn.noun2tophypernym(curHeadWord);
            for h = 1:size(curWordHypernyms, 2)
                hypernymVec(wn.nounhypernym2idx(curWordHypernyms{h})) = 1;
            end
        end
    end
    headExt = [headExt hypernymVec];


end

function childExt = getChildExt(extDim, wn, childWord, hypOffset, language)

    childExt = zeros(1, extDim);
    if strcmpi(language, 'arabic')
        if isKey(wn.word2tophypernym, childWord) % child top hypernym
            curWordHypernyms = wn.word2tophypernym(childWord);
            for h = 1:size(curWordHypernyms, 2)
                    childExt(hypOffset + wn.hypernym2idx(curWordHypernyms{h})) = 1;
            end
        end
    elseif strcmpi(language, 'english')
        if isKey(wn.noun2tophypernym, childWord) % child top hypernym
            curWordHypernyms = wn.noun2tophypernym(childWord);
            for h = 1:size(curWordHypernyms, 2)
                    childExt(hypOffset + wn.nounhypernym2idx(curWordHypernyms{h})) = 1;
            end
        end        
    end
end

function prepExt = getPrepExt(extDim)
% for now prepExt is just a zero vector
    prepExt = zeros(1, extDim);
end