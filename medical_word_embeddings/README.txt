This dataset contains English word embeddings pre-trained on biomedical texts from MEDLINE®/PubMed® using gensim's Word2Vec implementation. The embeddings of this dataset are an improved version of the Word2Vec embeddings we released in 2014 (http://bioasq.lip6.fr/info/BioASQword2vec/) in the context of the BioASQ challenge (http://www.bioasq.org/).


Two versions of word embeddings are provided, both in Word2Vec's C binary format:
        200-dimensional embeddings: file pubmed2018_w2v_200D.bin
        400-dimensional embeddings: file pubmed2018_w2v_400D.bin
        
In both versions, the vocabulary size is 2,665,547 types (distinct words).


Additional technical information:


- Papers and code of Mikolov et al.'s original Word2Vec:
        https://code.google.com/archive/p/word2vec/
        https://arxiv.org/pdf/1301.3781.pdf
        https://arxiv.org/pdf/1310.4546.pdf
        https://www.aclweb.org/anthology/N13-1090


- Word2Vec implementation used: gensim's Word2Vec (version 3.3.0). 
        https://radimrehurek.com/gensim/models/word2vec.html


- Corpus used: MEDLINE/PubMed Baseline Repository 2018 (January 2018).
        https://www.nlm.nih.gov/databases/download/pubmed_medline.html


- Preprocessing:
        Step 1: From the XML files of the MEDLINE/PubMed Baseline Repository, we extracted and used only the title and abstract of each article.
        
        Step 2: All the text fields of the abstracts were split into sentences using the sentence splitter (sent_tokenize) of NLTK (version 3.2.3).
                http://www.nltk.org/api/nltk.tokenize.html
        
        Step 3: All the titles and all the sentences of the abstracts were preprocessed and tokenized using the "bioclean" function, which is included in the toolkit.py script that accompanies the word embeddings of the BioASQ challenge.
                
        Step 4: gensim's Word2Vec implementation (skip-gram model) was then applied to the preprocessed and tokenized titles and sentences of the abstracts.


- Data statistics:
        Number of articles: 27,836,723
        Number of articles with title and abstract: 17,730,230
        Number of articles with title only (no abstract): 10,106,493
        Number of titles and sentences: 173,755,513 
        Number of tokens: 3,580,134,037
        Average sentence length (treating titles as sentences): 20.6 tokens
        
- Word2Vec settings used:
        min_count=5 (minimum corpus frequency)
        sg=1 (use skip-gram)
        hs=0 (use negative sampling)
        size=200, 400 (embedding dimensions)
        window=5 (maximum distance between current and predicted word)
        workers=20
        All other parameters were set to the defaults they have in gensim version 3.3.0.


Terms and conditions:


This dataset (word embeddings) was produced from a dataset (the corpus described above) provided by the National Library of Medicine (NLM). The following Terms and Conditions apply to NLM data: 
        https://www.nlm.nih.gov/databases/download/terms_and_conditions.html
This dataset does not reflect the most current/accurate data available from NLM.


This dataset was produced and is provided by the Natural Language Processing Group of the Department of Informatics, Athens University of Economics and Business, Greece (http://nlp.cs.aueb.gr/) with a Creative Commons Attribution-NonCommercial-ShareAlike (CC BY-NC-SA) license.
        https://creativecommons.org/licenses/by-nc-sa/4.0/
        https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

If you use this dataset or part of it, please cite the following paper: 

R. McDonald, G. Brokos and I. Androutsopoulos, "Deep Relevance Ranking Using Enhanced Document-Query Interactions". Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP 2018), Brussels, Belgium, 2018.

George Brokos and Ion Androutsopoulos
August 20, 2018