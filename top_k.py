import numpy as np
import sys
import json

vecs = np.loadtxt(sys.argv[1]).tolist()
vocab = np.loadtxt(sys.argv[2], dtype=str).tolist()

data = { word: vec for (vec,word) in zip(vecs, vocab)}

def most_similar(input_word, k):
    distances = {} 
    input_word_vec = data[input_word]
    for (word, vec) in data.items():
        if input_word != word:
           distances[word] = distance(input_word_vec, vec)
    sorted_distances = {word: dist for (word, dist) in sorted(distances.items(), key=lambda item: item[1], reverse=True)} 
    return {word: sorted_distances[word] for i, word in enumerate(sorted_distances) if i < k }


def distance(a,b):
    return np.dot(a,b)/(np.sqrt(np.dot(a,a))*np.sqrt(np.dot(b,b))) 


if __name__ == "__main__":

    words = ['dog', 'england', 'john', 'explode', 'office']
    part2 = open("part2.txt","w")
 
    for word in words:
        output = most_similar(word, 5)
        part2.writelines(word+'\n'+json.dumps(output)+'\n')
    part2.close()
