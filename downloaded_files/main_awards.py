import spacy
from functools import partial
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import string
import re
import json
import sys
from extraction import best_script, wins_script, goes_script, nominated_script, nominee_script, receives_script, winner_script


files=["best","wins","goes","nominated","nominee","receives","winner"]

method_map = {
    "best": best_script,
    "wins": wins_script,
    "goes": goes_script,
    "nominated": nominated_script,
    "nominee": nominee_script,
    "receives": receives_script,
    "winner": winner_script
    }

# Load the spaCy English language model
nlp = spacy.load("en_core_web_sm")

def custom_tokenize(text):
    # Tokenize using spaCy
    doc = nlp(text)
    
    # Filter out punctuation and spaces
    tokens = [token.text.lower() for token in doc if token.text not in string.punctuation + ' ']

    # print(tokens)
    return ''.join(tokens)

def join_awards(tweet_text):
    all_texts={}
    for file in files:
        data=method_map[file](tweet_text)
        for key, val in data.items():
            if key not in all_texts:
                all_texts[key]=0
            all_texts[key]+=val
    return all_texts

def get_awards(filename):
    award_list=[]
    f = open(filename,encoding="utf-8", errors="ignore")
    json_text=json.load(f)
    all_awards= join_awards(json_text)
    f.close()

    texts=all_awards.keys()
    tokenized_texts_str = [custom_tokenize(text) for text in texts]

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Vectorize the texts and convert to a dense matrix
    X = vectorizer.fit_transform(tokenized_texts_str).toarray()


    # Calculate Jaccard similarity
    jaccard_similarities = pairwise_distances(X, metric="jaccard")

    # hierarchical clustering based on Jaccard similarity
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.2, linkage='average', affinity='precomputed').fit(jaccard_similarities)

    # Get the cluster labels
    cluster_labels = clustering.labels_

    # Group the texts based on the cluster labels
    grouped_texts = {}
    for label, text in zip(cluster_labels, texts):
        if label in grouped_texts:
            grouped_texts[label].append(text)
        else:
            grouped_texts[label] = [text]

    d=open(f"awards_names.txt", "w")
    with open(f"inspect_files/clusters.txt", "w") as temp_file:
        for label, group in grouped_texts.items():
            temp_file.write(f"Cluster {label}:\n")
            cluster = []
            for text in group:
                cluster.append([text, all_awards[text]])
                try:
                    temp_file.write(f"    {text} (Count: {all_awards[text]})\n")
                except Exception as e:
                    print(f"Unable to write text to the file: {e}")
                    
            temp_file.write("\n")
            max_count_item = max(cluster, key=lambda x: x[1])

            # if something had many iterations or was mentioned multiple times, it is more likely that it's a W
            if (len(cluster) > 1 and max_count_item[1] > 7) or max_count_item[1]>70:
                max_count_item.append('1')
            else:
                max_count_item.append('0')

            try:
                # proposed award, instances, confidence
                prep=[re.sub("\n","" ,str(i).strip()) for i in max_count_item if len(str(i).strip())>0]
                if len(prep)==3 and prep[2]=='1':
                    entry=",".join(prep)
                    # print(entry, prep)
                    d.write(f'{prep[0]}\n')
                    award_list.append(prep[0])
            except Exception as e:
                print(f"Unable to write text to the file: {e}")

    return award_list
                
if __name__ == '__main__':
    filename=sys.argv[1]
    get_awards(filename)
