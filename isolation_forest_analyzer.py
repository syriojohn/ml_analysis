from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from base_analyzer import BaseAnalyzer
import pandas as pd
import numpy as np
from scipy import stats

class IsolationForestAnalyzer(BaseAnalyzer):
    def analyze(self, single_column=None, multi_columns=None, contamination='auto'):
        """Enhanced Isolation Forest analysis"""
        single_column = single_column or self.target_column
        multi_columns = multi_columns or self.numerical_columns
        
        print("\nPerforming Enhanced Isolation Forest analysis...")
        print(f"Dataset size: {len(self.df)} samples, {len(multi_columns)} features")
        
        # Check for sufficient data
        if len(self.df) < 10:
            print("Warning: Very small dataset. Results may not be reliable.")
        
        # Check for constant columns
        constant_columns = [col for col in multi_columns 
                          if self.df[col].nunique() == 1]
        if constant_columns:
            print(f"Warning: The following columns have constant values: {constant_columns}")
            multi_columns = [col for col in multi_columns if col not in constant_columns]
        
        return {
            'single_feature': self._analyze_single_feature(single_column, contamination),
            'multi_feature': self._analyze_multi_feature(multi_columns, contamination)
        }
    
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

    def _analyze_single_feature(self, column, contamination):
        print(f"\nRunning single-feature analysis on {column}...")
        
        values = self.df[[column]].values
        
        # Determine contamination if auto
        if contamination == 'auto':
            z_scores = np.abs((values - np.mean(values)) / np.std(values))
            suggested_contamination = np.mean(z_scores > 2.5)
            contamination = min(max(suggested_contamination, 0.01), 0.5)
            print(f"Auto-detected contamination level: {contamination:.3f}")
        
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        iso_forest.fit(values)
        
        all_scores = iso_forest.score_samples(values)
        predictions = iso_forest.predict(values)
        
        results = self.df.copy()
        results['anomaly_score'] = all_scores
        results['is_anomaly'] = predictions == -1
        results['analysis_type'] = 'Single-feature'
        results['analyzed_column'] = column
        
        results['explanation'] = results.apply(
            lambda row: self._generate_single_feature_explanation(
                row[column], 
                row['anomaly_score'],
                column
            ),
            axis=1
        )
        
        return results
    
    def _analyze_multi_feature(self, columns, contamination):
        print(f"Running multi-feature analysis...")
        
        # Prepare data
        X_multi = self.df[columns]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_multi)
        
        # Determine contamination if auto
        if contamination == 'auto':
            try:
                # Try Mahalanobis distance method
                cov = np.cov(X_scaled.T)
                try:
                    inv_cov = np.linalg.inv(cov)
                    mean = np.mean(X_scaled, axis=0)
                    dist = stats.chi2.ppf(0.975, df=len(columns))
                    
                    mahal_dist = np.array([
                        np.sqrt((x - mean).T.dot(inv_cov).dot(x - mean))
                        for x in X_scaled
                    ])
                    
                    suggested_contamination = np.mean(mahal_dist > dist)
                    
                except np.linalg.LinAlgError:
                    # Fall back to simpler method if matrix is singular
                    print("Note: Covariance matrix is singular, using Z-score based contamination estimation")
                    z_scores = np.max(np.abs(X_scaled), axis=1)
                    suggested_contamination = np.mean(z_scores > 2.5)
            
            except Exception as e:
                print(f"Warning: Error in contamination estimation ({str(e)}), using default value")
                suggested_contamination = 0.1
                
            contamination = min(max(suggested_contamination, 0.01), 0.5)
            print(f"Auto-detected contamination level: {contamination:.3f}")
        
        # Fit Isolation Forest
        iso_forest_multi = IsolationForest(contamination=contamination, random_state=42)
        iso_forest_multi.fit(X_scaled)
        
        # Get scores for all points
        all_scores = iso_forest_multi.score_samples(X_scaled)
        predictions = iso_forest_multi.predict(X_scaled)
        
        # Prepare results DataFrame
        results = self.df.copy()
        results['anomaly_score'] = all_scores
        results['is_anomaly'] = predictions == -1
        results['analysis_type'] = 'Multi-feature'
        
        # Calculate feature contributions for all points
        for feature in columns:
            feature_zscore = abs((results[feature] - self.df[feature].mean()) / self.df[feature].std())
            results[f'{feature}_contribution'] = feature_zscore
            
        # Calculate contribution percentages
        contribution_cols = [f'{f}_contribution' for f in columns]
        total_contributions = results[contribution_cols].sum(axis=1)
        
        # Handle zero total contributions
        total_contributions = np.where(total_contributions == 0, 1, total_contributions)
        
        for feature in columns:
            results[f'{feature}_percent'] = (
                results[f'{feature}_contribution'] / total_contributions * 100
            ).round(2)
        
        # Add explanations
        results['explanation'] = results.apply(
            lambda row: self._generate_multi_feature_explanation(
                row,
                columns,
                contribution_cols
            ),
            axis=1
        )
        
        # Print feature importance summary
        unique_anomalies = len(results[results['is_anomaly']]) 
        print(f"\nDetected {unique_anomalies} anomalies ({(unique_anomalies/len(results)*100):.1f}% of data)")
        
        return results
    
    def _generate_single_feature_explanation(self, value, score, column):
        """Generate explanation for single-feature analysis"""
        contexts = self._get_statistical_context(value, column)
        
        explanation = []
        if contexts:
            explanation.extend(contexts)
        
        if score < -0.5:
            explanation.append(f"Highly anomalous (score: {score:.3f})")
        elif score < -0.3:
            explanation.append(f"Moderately anomalous (score: {score:.3f})")
        else:
            explanation.append(f"Normal range (score: {score:.3f})")
                
        return "; ".join(explanation) if explanation else "No significant anomalies detected"
    
    def _generate_multi_feature_explanation(self, row, columns, contribution_cols):
        """Generate explanation for multi-feature analysis"""
        if not row['is_anomaly']:
            return "No significant anomalies detected"
            
        explanation = []
        
        # Add contribution information
        contributions = []
        for col in columns:
            contributions.append((col, row[f'{col}_percent']))
        
        # Sort by contribution percentage
        contributions.sort(key=lambda x: x[1], reverse=True)
        
        # Add top contributors to explanation
        top_contributors = contributions[:3]
        contribution_text = ", ".join([
            f"{feat} ({pct:.1f}%)"
            for feat, pct in top_contributors
        ])
        explanation.append(f"Top contributing features: {contribution_text}")
        
        # Add statistical context for top contributor
        top_feature = top_contributors[0][0]
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