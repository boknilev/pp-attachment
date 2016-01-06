function data = loadData(model, params, pref, wordVectors, varargin)

%%%% default values %%%%
numvarargs = length(varargin);
if numvarargs > 2
    error('loadData:TooManyInputs', ...
        'requires at most 2 optional input');
end
if params.useExt && numvarargs ~= 2
    error('loadData:TooFewInputs', ...
        'if useExt=true, must have exactly 2 more inputs');
end

% set defaults for optional inputs 
optargs = {'' ''};

% now put these defaults into the valuesToUse cell array, 
% and overwrite the ones specified in varargin.
optargs(1:numvarargs) = varargin;

% Place optional args in memorable variable names
[vn wn] = optargs{:};
%%%%%%%%%%%%%%%%%%%%%%%%%  

    
if model == 6 
    prepsFilename = [pref '.' 'preps'];
    ppChildrenFilename = [pref '.' 'children'];
    headsFilePref = [pref '.' 'heads'];
    labelsFilename = [pref '.' 'labels'];
    nheadsFilename = [pref '.' 'nheads'];
    if params.useExt 
        headsPosfilename = [headsFilePref '.' 'pos'];
        headsNextPosFilename = [headsFilePref '.' 'next.pos'];
        if ~isfield(params, 'pos2idx') % create map pos2idx if not created already
            data.pos2idx = loadPosMapFromFile(headsNextPosFilename);
        else
            data.pos2idx = params.pos2idx;
        end
        [heads, preps, ppChildren, labels, nheads, includeInd] = loadDataSingleWordPP(wordVectors, params.inputSize, params.maxNumHeads, ...
                        [headsFilePref '.words'], [prepsFilename '.words'], [ppChildrenFilename '.words'], ...
                        labelsFilename, nheadsFilename, params.scaleVectors, ...
                        params.useExt, headsPosfilename, headsNextPosFilename, data.pos2idx, vn, wn, params.language);
    else      
        [heads, preps, ppChildren, labels, nheads, includeInd] = loadDataSingleWordPP(wordVectors, params.inputSize, params.maxNumHeads, ...
                    [headsFilePref '.words'], [prepsFilename '.words'], [ppChildrenFilename '.words'], ...
                    labelsFilename, nheadsFilename, params.scaleVectors, params.useExt);
    end
    data.heads = heads; data.preps = preps; data.ppChildren = ppChildren; ...
        data.labels = labels; data.nheads = nheads; data.includeInd = includeInd;
    if params.updateWordVectors
        [wordVectorsMat, indHeadsToWordVectors, indPrepsToWordVectors, indChildrenToWordVectors, labels, nheads] = ...
            loadDataSingleWordPPIndices(wordVectors, params.inputSize, params.maxNumHeads, ...
                [headsFilePref '.words'], [prepsFilename '.words'], [ppChildrenFilename '.words'], ...
                labelsFilename, nheadsFilename, params.scaleVectors);
        data.indHeadsToWordVectors = indHeadsToWordVectors; data.indPrepsToWordVectors = indPrepsToWordVectors;
        data.indChildrenToWordVectors = indChildrenToWordVectors; data.labels = labels; data.nheads = nheads;
        data.wordVectorsMat = wordVectorsMat;
    end
else   
    error('Error', ['unknown model ' num2str(model) ' in loadData()']);
end
    
    
end



function pos2idx = loadPosMapFromFile(headsNextPosFilename)

% load a map from pos to idx from a file with the pos for words following
% candidate heads

headNextPosLines = loadLinesFromFile(headsNextPosFilename);
pos2idx = containers.Map(); % map pos to idx
for i = 1:size(headNextPosLines, 1)
    poses = regexp(headNextPosLines(i), '\s+', 'split');
    poses = poses{1};
    for j = 1:size(poses, 2)
        pos = poses{j};
        if ~isKey(pos2idx, pos)
            pos2idx(pos) = pos2idx.Count+1;
        end
    end
end


end

