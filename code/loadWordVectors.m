function mapWordVectors = loadWordVectors(filename, inputSize)

% filename - file containing word vectors, each line starts with
%            a word, then followed by space-separated vector
% inputSize - size of word vectors

% Returns a Map object from words to vector arrays

fid = fopen(filename);
C = textscan(fid, ['%s', repmat('%f', 1, inputSize)]);
fclose(fid);
words = C{1};
vectors = [C{2:end}];
vectorsCell = mat2cell(vectors, ones(1, size(words, 1)));
mapWordVectors = containers.Map(words', vectorsCell');

end
