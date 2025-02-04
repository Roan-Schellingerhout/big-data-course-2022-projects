from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import multiprocessing as mp
from tqdm import tqdm
import pandas as pd
import math

def d2v_embed(df_col, filename, max_epochs = 100, vec_size = 128, alpha = 0.025):
    
    
    try:
        print(f"Looking for pre-made d2v embedding of {df_col.name} column...")
        return pd.read_csv(f"{filename}_doc2vec_embeddings_{df_col.name}.csv", index_col = 'id')
    except:
        print("Not found, creating new one..")
        
   
    df_col = df_col.fillna(" ")
    df_col = df_col.str.lower()\
                   .str.normalize('NFKD')\
                   .str.encode('ascii', errors='ignore')\
                   .str.decode('utf-8')\
                   .str.replace("\W", " ", regex=True)
    
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(df_col)]

    model = Doc2Vec(vector_size=vec_size,
                    alpha=alpha, 
                    min_alpha=0.00025,
                    min_count=1,
                    dm =1,
                    workers = mp.cpu_count())
  
    model.build_vocab(tagged_data)

    for epoch in tqdm(range(max_epochs), desc=f"Training on {df_col.name}"):
    #     print('iteration {0}'.format(epoch))
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.epochs)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha
    
    # save model
    model.save(f"{filename}_doc2vec_model_{df_col.name}.model")
    
    #return df with doc embeddings
    
    embs = pd.DataFrame([model.docvecs[i] for i in range(len(df_col))], 
                        index = df_col.index,
                        columns = [f"{df_col.name}_{i}" for i in range(vec_size)])
    
    embs.to_csv(f"{filename}_doc2vec_embeddings_{df_col.name}.csv")
    
    return embs