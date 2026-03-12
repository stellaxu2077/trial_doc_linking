import pickle
import numpy as np

data = pickle.load(open('data/bm25_candidates_test.pkl', 'rb'))
candidates_map = data['candidates_map']
all_ncts = list(candidates_map.keys())
print('Total trials:', len(all_ncts))

np.random.seed(42)
sampled = np.random.choice(all_ncts, size=int(len(all_ncts)*0.1), replace=False)
print('Sampled:', len(sampled))

sampled_map = {k: candidates_map[k] for k in sampled}
data['candidates_map'] = sampled_map
pickle.dump(data, open('data/bm25_candidates_test_sampled10pct.pkl', 'wb'))
print('Saved!')
