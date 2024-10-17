from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.neighbors import KNeighborsClassifier
from Classifiers.classifier import Classifier
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

class KNNTrainer(Classifier):
    def findBestParams(self):
        name = "KNNClassifier"
        self.model = KNeighborsClassifier()
        param_grid = {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'kd_tree', 'brute'],
            'metric': ['euclidean', 'chebyshev'],
        }
        
        best_params = self.loadBestParams(name)
        if best_params:
            print(f'Using saved best parameters for {name}:', best_params)
            self.params = best_params
            return None

        scorer = make_scorer(accuracy_score)
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(self.model, param_grid, scoring=scorer, cv=cv, verbose=1)
        grid_search.fit(self.X, self.Y)

        best_params = grid_search.best_params_
        print(f'Best parameters for {name}:', best_params)
        
        self.saveBestParams(best_params, name)
        self.params = best_params
        return None
