# custom_iforest_analyzer.py
from sklearn.ensemble import IsolationForest
from sklearn.ensemble._iforest import _average_path_length
from sklearn.preprocessing import StandardScaler
from base_analyzer import BaseAnalyzer
import numpy as np
from collections import defaultdict
from scipy import stats

class LoggingIsolationTree:
    """Single isolation tree with logging capabilities"""
    def __init__(self, max_samples, random_state=None):
        self.max_samples = max_samples
        self.random_state = random_state
        self.feature_splits = defaultdict(int)
        self.split_values = []
        self.path_lengths = defaultdict(list)
    
    def fit(self, X):
        """Fit the tree while logging decisions"""
        self.n_samples = X.shape[0]
        self.height_limit = int(np.ceil(np.log2(self.max_samples)))
        self.root = self._grow_tree(X, 0)
        return self

    def _grow_tree(self, X, current_height):
        """Grow tree and log split decisions"""
        n_samples = X.shape[0]

        if current_height >= self.height_limit or n_samples <= 1:
            return {'type': 'exNode', 'size': n_samples}

        # Randomly select a feature
        split_feature = np.random.randint(X.shape[1])
        self.feature_splits[split_feature] += 1

        # Find min and max for selected feature
        x_max = X[:, split_feature].max()
        x_min = X[:, split_feature].min()
        
        if x_max == x_min:
            return {'type': 'exNode', 'size': n_samples}

        # Random split value
        split_value = np.random.uniform(x_min, x_max)
        self.split_values.append((split_feature, split_value))

        # Split the data
        left_indices = X[:, split_feature] < split_value
        right_indices = ~left_indices
        
        # Log path lengths for samples
        for idx in range(n_samples):
            self.path_lengths[idx].append(current_height)

        return {
            'type': 'inNode',
            'split_feature': split_feature,
            'split_value': split_value,
            'left': self._grow_tree(X[left_indices], current_height + 1),
            'right': self._grow_tree(X[right_indices], current_height + 1)
        }

class LoggingIsolationForest(IsolationForest):
    """Custom Isolation Forest that logs decision process"""
    
    def __init__(self, **kwargs):
        # Handle max_samples parameter
        if 'max_samples' in kwargs:
            if isinstance(kwargs['max_samples'], str):
                if kwargs['max_samples'] == 'auto':
                    kwargs['max_samples'] = 'auto'
                else:
                    try:
                        kwargs['max_samples'] = int(kwargs['max_samples'])
                    except ValueError:
                        kwargs['max_samples'] = 256  # default value
            elif isinstance(kwargs['max_samples'], float):
                kwargs['max_samples'] = int(kwargs['max_samples'])
        else:
            kwargs['max_samples'] = 256  # default value

        super().__init__(**kwargs)
        self.feature_importances_ = None
        self.trees = []
        self.feature_names = None
    
    def fit(self, X, y=None, feature_names=None):
        """Fit with additional logging of tree decisions"""
        if feature_names is not None:
            self.feature_names = feature_names
        
        # Initialize logging structures
        n_samples, n_features = X.shape
        self.trees = []
        all_feature_splits = defaultdict(int)
        
        # Determine actual sample size
        if isinstance(self.max_samples, str):
            if self.max_samples == 'auto':
                max_samples = min(256, n_samples)
            else:
                max_samples = 256
        elif isinstance(self.max_samples, float):
            max_samples = int(self.max_samples * n_samples)
        else:
            max_samples = min(self.max_samples, n_samples)
        
        # Train trees with logging
        for i in range(self.n_estimators):
            # Sample data for this tree
            sample_size = max_samples
            sample_idx = np.random.choice(n_samples, size=sample_size, replace=False)
            X_sample = X[sample_idx]
            
            # Create and fit logging tree
            tree = LoggingIsolationTree(sample_size, random_state=self.random_state)
            tree.fit(X_sample)
            self.trees.append(tree)
            
            # Aggregate feature splits
            for feature, count in tree.feature_splits.items():
                all_feature_splits[feature] += count
        
        # Calculate feature importances
        total_splits = sum(all_feature_splits.values())
        self.feature_importances_ = np.zeros(n_features)
        for feature, count in all_feature_splits.items():
            self.feature_importances_[feature] = count / total_splits
        
        # Fit the original isolation forest
        super().fit(X)
        return self
    
    def decision_path(self, X):
        """Get detailed decision path for each sample"""
        paths = []
        for tree in self.trees:
            current_paths = []
            for idx in range(X.shape[0]):
                if idx in tree.path_lengths:
                    current_paths.append(tree.path_lengths[idx])
            paths.append(current_paths)
        return paths

    def get_feature_importance_details(self):
        """Get detailed feature importance information"""
        if self.feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(self.feature_importances_))]
        else:
            feature_names = self.feature_names
            
        details = []
        for idx, importance in enumerate(self.feature_importances_):
            details.append({
                'feature': feature_names[idx],
                'importance': importance,
                'percentage': importance * 100,
                'total_splits': sum(tree.feature_splits[idx] for tree in self.trees)
            })
        return sorted(details, key=lambda x: x['importance'], reverse=True)

class CustomIsolationForestAnalyzer(BaseAnalyzer):
    """Analyzer using custom Isolation Forest with detailed logging"""
    
    def _get_statistical_context(self, value, column):
        """Get statistical context for a value in a column"""
        mean = self.df[column].mean()
        std = self.df[column].std()
        percentile = stats.percentileofscore(self.df[column], value)
        z_score = (value - mean) / std if std != 0 else 0
        
        context = []
        if abs(z_score) > 2:
            context.append(f"{abs(z_score):.2f} standard deviations from mean")
        if percentile < 5:
            context.append(f"In bottom {percentile:.1f}% of values")
        elif percentile > 95:
            context.append(f"In top {(100-percentile):.1f}% of values")
            
        return context
    
    def analyze(self, columns=None, contamination='auto'):
        print("\nRunning Custom Isolation Forest analysis with decision logging...")
        columns = columns or self.numerical_columns
        
        # Prepare data
        X_multi = self.df[columns].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_multi)
        
        # Fit custom Isolation Forest with explicit max_samples
        iso_forest = LoggingIsolationForest(
            contamination=contamination, 
            random_state=42,
            n_estimators=100,
            max_samples=min(256, len(self.df))
        )
        iso_forest.fit(X_scaled, feature_names=columns)
        
        # Get predictions and scores
        predictions = iso_forest.predict(X_scaled)
        all_scores = iso_forest.score_samples(X_scaled)
        
        # Get detailed feature importance information
        feature_importance_details = iso_forest.get_feature_importance_details()
        
        # Prepare results DataFrame
        results = self.df.copy()
        results['anomaly_score'] = all_scores
        results['is_anomaly'] = predictions == -1
        results['analysis_type'] = 'Custom-IF'
        
        # Add actual feature importances
        for detail in feature_importance_details:
            feature = detail['feature']
            results[f'{feature}_importance'] = detail['percentage']
        
        # Calculate average path length more safely
        try:
            decision_paths = iso_forest.decision_path(X_scaled)
            if decision_paths:
                # Initialize list to store average path lengths
                avg_path_lengths = []
                
                # Calculate average path length for each sample
                for i in range(len(X_scaled)):
                    sample_paths = []
                    for tree_paths in decision_paths:
                        if i < len(tree_paths) and tree_paths[i]:
                            sample_paths.append(len(tree_paths[i]))
                    
                    # Calculate average for this sample
                    if sample_paths:
                        avg_path_lengths.append(np.mean(sample_paths))
                    else:
                        avg_path_lengths.append(0)
                
                results['avg_path_length'] = avg_path_lengths
            else:
                # If no paths available, set default values
                results['avg_path_length'] = [0] * len(X_scaled)
        except Exception as e:
            print(f"Warning: Could not calculate path lengths: {str(e)}")
            results['avg_path_length'] = [0] * len(X_scaled)
        
        # Generate explanations
        results['explanation'] = results.apply(
            lambda row: self._generate_explanation(
                row,
                feature_importance_details,
                columns
            ),
            axis=1
        )
        
        # Print feature importance summary
        print("\nFeature Importance Summary:")
        for detail in feature_importance_details:
            print(f"{detail['feature']}: {detail['percentage']:.2f}% "
                  f"({detail['total_splits']} splits)")
        
        # Print anomaly detection summary
        n_anomalies = results['is_anomaly'].sum()
        print(f"\nDetected {n_anomalies} anomalies ({(n_anomalies/len(results)*100):.1f}% of data)")
        
        return results
    
    def _generate_explanation(self, row, feature_importance_details, columns):
        """Generate explanation using actual IF decisions"""
        if not row['is_anomaly']:
            return "No significant anomalies detected"
            
        explanation = []
        
        # Add feature importance information
        importance_text = ", ".join([
            f"{detail['feature']} ({detail['percentage']:.1f}%)"
            for detail in feature_importance_details[:3]
        ])
        explanation.append(f"Most important features: {importance_text}")
        
        # Add path length context
        avg_path = row['avg_path_length']
        if avg_path < np.log2(len(self.df)) * 0.5:
            explanation.append(f"Isolated quickly (avg path length: {avg_path:.1f})")
        
        # Add statistical context for important features
        top_feature = feature_importance_details[0]['feature']
        contexts = self._get_statistical_context(row[top_feature], top_feature)
        if contexts:
            explanation.append(f"Main feature ({top_feature}): {'; '.join(contexts)}")
        
        # Add anomaly score interpretation
        score = row['anomaly_score']
        if score < -0.5:
            explanation.append(f"Highly anomalous (score: {score:.3f})")
        elif score < -0.3:
            explanation.append(f"Moderately anomalous (score: {score:.3f})")
        else:
            explanation.append(f"Normal range (score: {score:.3f})")
        
        return "; ".join(explanation)
