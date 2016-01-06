function lines = loadLinesFromFile(filename)

% filename - file containing single word per line

% Return a cell array holding the words

fid = fopen(filename);
C = textscan(fid, '%s', 'delimiter', '\n');
fclose(fid);
lines = C{1};

end