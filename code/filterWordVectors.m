
function filteredWordVectors = filterWordVectors(wordVectors, model, params, filenames)

disp('filtering word vectors based on train/test data');
disp(['wordVectors size before filtering: ' num2str(wordVectors.Count)]);
intrainWordVectors = filterWordVectorsFromData(wordVectors, model, params, filenames.trainFilePref);
intestWordVectors = filterWordVectorsFromData(wordVectors, model, params, filenames.testFilePref);
filteredWordVectors = mergeMaps(intrainWordVectors, intestWordVectors);
disp(['wordVectors size after filtering: ' num2str(filteredWordVectors.Count)]);

end




function filteredWordVectors = filterWordVectorsFromData(wordVectors, model, params, pref)

% store UNK vector before filtering
if params.useUnk 
    if ~isKey(wordVectors, 'UNK')
        disp('WARNING: cannot useUnk when no UNK vector exists, resorting to zeros');
        unkVec = zeros(params.inputSize);
    else
        unkVec = wordVectors('UNK');
    end
end


if model == 1 || model == 2 || model == 5 || model == 6 || model == 7 || model == 10 || model == 12
    prepsFilename = [pref '.' 'preps'];
    ppChildrenFilename = [pref '.' 'children'];
    headsFilePref = [pref '.' 'heads'];
    filteredWordVectors = filterWordVectorsFromFile(wordVectors, [prepsFilename '.words']);
    filteredWordVectors = mergeMaps(filteredWordVectors, filterWordVectorsFromFile(wordVectors, [ppChildrenFilename '.words']));
    filteredWordVectors = mergeMaps(filteredWordVectors, filterWordVectorsFromHeadFile(wordVectors, [headsFilePref '.words']));        
    if model == 10 || model == 12
        headsNextWordsFilename = [pref '.' 'heads.next.words'];
        filteredWordVectors = mergeMaps(filteredWordVectors, filterWordVectorsFromHeadFile(wordVectors, headsNextWordsFilename));
    end
else    
    disp(['Error: filtering word vectors not implemented for model ' num2str(model)]);
    return;
end

% add UNK vector if necessary
if params.useUnk
    filteredWordVectors('UNK') = unkVec;
end    

end

function filteredWordVectors = filterWordVectorsFromFile(wordVectors, file)
    words = loadLinesFromFile(file);
    filteredWordVectors = containers.Map();
    for i = 1:size(words, 1)
        word = words{i};
        if isKey(wordVectors, word) && ~isKey(filteredWordVectors, word)
            filteredWordVectors(word) = wordVectors(word);
        end
    end
end

function filteredWordVectors = filterWordVectorsFromHeadFile(wordVectors, file)
    headLines = loadLinesFromFile(file);
    filteredWordVectors = containers.Map();
    for i = 1:size(headLines, 1)
        headLine = headLines(i);
        curHeadWords = regexp(headLine, '\s+', 'split');
        curHeadWords = curHeadWords{1};
        numExistingHeads = size(curHeadWords,2);
        for j = 1:numExistingHeads
            curHeadWord = curHeadWords{j};
            if isKey(wordVectors, curHeadWord) && ~isKey(filteredWordVectors, curHeadWord)
                filteredWordVectors(curHeadWord) = wordVectors(curHeadWord);
            end
        end
    end
end
  

function m = mergeMaps(m1, m2)

m = m1;
m2keys = m2.keys();
m2vals = m2.values();
for i = 1:m2.Count
    k = m2keys{i};
    v = m2vals{i};
    if ~isKey(m, k)
        m(k) = v;
    end
end

end
    
