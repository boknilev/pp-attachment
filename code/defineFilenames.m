function filenames = defineFilenames(language)

%%% Filenames %%%%%%
% Set the following file names as per your paths
% They come pre-set with examples for Arabic/English, but you only need to
% set one set of paths
%%%%%%%%%%%%%%%%%%%%

% data files 
if strcmpi(language, 'arabic')
    filenames.trainFilePref = '../data/pp-data-arabic/spmrl.train.lab.all.pp';
    filenames.testFilePref = '../data/pp-data-arabic/spmrl.test.lab.1.pp';
    filenames.wordVectorsFilename = '../data/vectors.arabic.100.txt';
elseif strcmpi(language, 'english')
    filenames.trainFilePref = '../data/pp-data-english/wsj.2-21.txt.dep.pp';
    filenames.testFilePref = '../data/pp-data-english/wsj.23.txt.dep.pp';
    filenames.wordVectorsFilename = '../data/vectors.english.100.txt';
else 
    error('Error', ['unknown language ' language ' in defineFilenames()']);
end

% directories for certain verbnet and wordnet files
% (not included with this distribution)
filenames.vnDir = '';
filenames.wnDir = '';


% files for saving and loading parameters
filenames.paramsFile = 'paramOpt.mat';
filenames.saveParamsFile = 'paramOpt.mat';
filenames.sumSquaresFile = 'sumSquares.mat';
filenames.saveSumSquaresFile = 'sumSquares.mat';
filenames.updatedWordVectorsFile = 'updatedWordVectors.mat';
filenames.saveUpdatedWordVectorsFile = 'updatedWordVectors.mat';
filenames.predictionsFile = 'predictions.mat';