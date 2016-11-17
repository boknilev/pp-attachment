# PP Attachment with Vector Representations 

This repository contains sample code for our TACL paper on PP attachment:

"Exploring Compositional Architectures and Word Vector Representations for Prepositional Phrase Attachment", Yonatan Belinkov, Tao Lei, Regina Barzilay, and Amir Globerson, TACL 2014.


## Instructions
1. Download code and data into your working folder.

2. Unzip word vector files in the data directory:
    ```
    $ cd data
    $ gunzip vectors.english.100.txt.gz
    $ gunzip vectors.arabic.100.txt.gz
    ```

3. Run Matlab from the code directory:
    ```
    $ cd code
    $ matlab
    ```

4. In Matlab, run the runner file:
    ```matlab
    >> run('english');
    ```
or 
    ```matlab
    >> run('arabic');
    ```



## Vector representations
Controlling the word vector representation can be done in three ways.

1. Using a different initial word vector file:
    
    Change the wordVectorsFilename in defineFilenames.m to point to the word vector file of your choice.

    You can experiment with our Arabic syntactic vectors, available at:
    http://groups.csail.mit.edu/rbg/code/pp/arabic-syntactic-vecs

2. Relearning word vectors during training: 
    
    Edit run.m and set:
    ```matlab
    params.updateWordVectors = true;
    ```

3. Enriching word vectors with other resources:
    
    This option requires using Verbnet and Wordnet, which are not provided with the current distribution. Get in touch if you would like to explore this option.


## Compositional architectures
This sample code only implements the HPCD model described in the paper.
Contact us if you would like to explore other model architectures.


## Training parameters
In run.m, you can change some parameters such as the dropout rate, learning rate and the batch size. In particular, to get good performance try increasing the epochs:
```matlab
trainParams.epochs = 3200
```

You can save and load intermediate models by defining the proper filenames in defineFilenames.m, and setting the following flags in run.m to "true":
```matlab
usePretrainedParams = true;
usePretrainedSumSquares = true;
usePretrainedWordVectors = true;
```


## Experimenting with your own PP attachment dataset
This distribution comes with our PP attachment datasets for Arabic (pp-data-arabic) and English (pp-data-english). If you would like to experiment with your own dataset, you will need to prepare it in a certain format. The train/test files must have the following names and formats:

\<train/test suffix\>.children.words - each line has one word which is the preposition's child.  
\<train/test suffix\>.heads.words - each line has a space-delimited list of candidate heads.  
\<train/test suffix\>.pp.labels - each line has an integer indicating the gold head index (1-based).  
\<train/test suffix\>.pp.nheads - each line has an integer indicating the number of candidate.  
\<train/test suffix\>.pp.preps.words - each line has one word which is the preposition.  

where \<train/test suffix\> is a prefix for the train/test file names. You will need to edit defineFilenames.m to point to these prefixes.
Have a look at the files under pp-data-english/arabic to see what the format looks like.

If you are experimenting with a different language, you will also need to provide a new word vectors file and edit defineFilenames.m to point to it. 

We would be happy to hear about your experiments with different datasets.  


## Citing
If you use this code your work, please cite the following paper:

Yonatan Belinkov, Tao Lei, Regina Barzilay, and Amir Globerson. 2014. Exploring Compositional Architectures and Word Vector Representations for Prepositional Phrase Attachment. Transactions of the Association for Computational Linguistics (TACL).
https://tacl2013.cs.columbia.edu/ojs/index.php/tacl/article/download/488/100

```bib
@article{belinkovTACL:2014,
	author = {Yonatan Belinkov and Tao Lei and Regina Barzilay and Amir Globerson},
	title = {Exploring Compositional Architectures and Word Vector Representations for Prepositional Phrase Attachment},
	journal = {Transactions of the Association for Computational Linguistics},
	volume = {2},
	year = {2014},
	issn = {2307-387X},
	url = {https://tacl2013.cs.columbia.edu/ojs/index.php/tacl/article/view/488},
	pages = {561--572}
}
```



## Questions?
For any questions or suggestions, email belinkov@mit.edu.

## TODO
* Provide other models besides HPCD.

