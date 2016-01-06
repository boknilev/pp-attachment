function [vn, wn] = loadVerbnetWordnet(vnDir, wnDir, language)

% load Verbnet and Wordnet from disk (prepared by Python scripts)

if strcmpi(language, 'english')
    [vn, wn] = loadEnglishVerbnetWordnet([vnDir '/' 'verb2prep.txt'], [vnDir '/' 'verbalnoun2prep.txt'], ...
                              [vnDir '/' 'verb2class.txt'], [wnDir '/' 'word2tophypernym.txt']);
elseif strcmpi(language, 'arabic')
    [vn, wn] = loadArabicVerbnetWordnet([vnDir '/' 'verb2prep.txt'], [vnDir '/' 'verb2class.txt'], ...
                                [wnDir '/' 'verb2tophypernym.txt'], [wnDir '/' 'noun2tophypernym.txt']);
else
    disp(['Error: unknown language ' language]);    
end





end


function [evn, ewn] = loadEnglishVerbnetWordnet(verbPrepFilename, verbClassFilename, ...
                                                verbTopHypernymFilename, nounTopHypernymFilename)
% load English Verbnet and Wordnet 

% verbnet
verbPrepLines = loadLinesFromFile(verbPrepFilename);
evn.verb2prep = getMapFromLines(verbPrepLines);
verbClassLines = loadLinesFromFile(verbClassFilename);
evn.verb2class = getMapFromLines(verbClassLines);
verbClasses = {}; values = evn.verb2class.values;
for i = 1:size(values, 2)
    verbClasses = union(verbClasses, values{i});
end
evn.verbclass2idx = containers.Map(verbClasses, [1:size(verbClasses, 2)]);

% wordnet
% noun
nounTopHypernymLines = loadLinesFromFile(nounTopHypernymFilename);
ewn.noun2tophypernym = getMapFromLines(nounTopHypernymLines);
hypernyms = {}; values = ewn.noun2tophypernym.values;
for i = 1:length(values)
    hypernyms = union(hypernyms, values{i});
end
ewn.nounhypernym2idx = containers.Map(hypernyms, [1:length(hypernyms)]);

% verb (not used for now)
verbTopHypernymLines = loadLinesFromFile(verbTopHypernymFilename);
ewn.verb2tophypernym = getMapFromLines(verbTopHypernymLines);
hypernyms = {}; values = ewn.verb2tophypernym.values;
for i = 1:size(values, 2)
    hypernyms = union(hypernyms, values{i});
end
ewn.verbhypernym2idx = containers.Map(hypernyms, [1:size(hypernyms, 2)]);

end




function [avn, awn] = loadArabicVerbnetWordnet(verbPrepFilename, verbalnounPrepFilename, verbClassFilename, wordTopHypernymFilename)
% load Arabic Verbnet and Wordnet 

% verbnet
verbPrepLines = loadLinesFromFile(verbPrepFilename);
avn.verb2prep = getMapFromLines(verbPrepLines);
verbalnounPrepLines = loadLinesFromFile(verbalnounPrepFilename);
avn.verbalnoun2prep = getMapFromLines(verbalnounPrepLines);
% verb classes (not used for now)
verbClassLines = loadLinesFromFile(verbClassFilename);
avn.verb2class = getMapFromLines(verbClassLines);
verbClasses = {}; values = avn.verb2class.values;
for i = 1:size(values, 2)
    verbClasses = union(verbClasses, values{i});
end
avn.verbclass2idx = containers.Map(verbClasses, [1:size(verbClasses, 2)]);

% wordnet
wordTopHypernymLines = loadLinesFromFile(wordTopHypernymFilename);
awn.word2tophypernym = getMapFromLines(wordTopHypernymLines);
hypernyms = {}; values = awn.word2tophypernym.values;
for i = 1:size(values, 2)
    hypernyms = union(hypernyms, values{i});
end
awn.hypernym2idx = containers.Map(hypernyms, [1:size(hypernyms, 2)]);

end


function m = getMapFromLines(lines)
    m = containers.Map();
    for i = 1:size(lines)
        line = regexp(lines(i), '\s+', 'split');
        line = line{1};
        k = line{1};
        if size(line, 2) > 1
            vals = line(2:end);
        else
            vals = {};
        end
        m(k) = vals;    
    end
end