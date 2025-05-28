#!/usr/bin/env python3
"""
Comprehensive Model Search Script
Trains multiple ML models with GridSearchCV to find the best performer for task duration prediction.
"""

import pandas as pd
import numpy as np
import pickle
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline

# Additional ML libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

try:
    from sklearn.experimental import enable_hist_gradient_boosting
    from sklearn.ensemble import HistGradientBoostingRegressor
    HIST_GB_AVAILABLE = True
except ImportError:
    HIST_GB_AVAILABLE = False

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

class ComprehensiveModelSearch:
    def __init__(self, cv_folds=5, test_size=0.2, random_state=42, n_jobs=-1):
        self.cv_folds = cv_folds
        self.test_size = test_size
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.results = {}
        self.best_model = None
        self.best_score = -np.inf
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        """Load embeddings data"""
        try:
            with open('../data/embeddings_data.pkl', 'rb') as f:
                data = pickle.load(f)
            
            embeddings = np.array(data['embeddings'])
            durations = np.array(data['duration_minutes'])
            
            print(f"Loaded {len(embeddings)} samples with {embeddings.shape[1]} embedding dimensions")
            print(f"Duration range: {durations.min():.1f} - {durations.max():.1f} minutes")
            
            return embeddings, durations
            
        except FileNotFoundError:
            print("embeddings_data.pkl not found. Run generate_embeddings.py first.")
            return None, None
    
    def prepare_data(self, X, y):
        """Split data and prepare for training"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        print(f"Training samples: {self.X_train.shape[0]}")
        print(f"Test samples: {self.X_test.shape[0]}")
        
    def get_model_configs(self):
        """Define only the most promising models for sentence embeddings"""
        
        models = {}
        
        # Simple baselines (unscaled to preserve embedding semantics)
        models['Linear Regression'] = {
            'model': LinearRegression(),
            'params': {}
        }
        
        models['Ridge Regression'] = {
            'model': Ridge(),
            'params': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            }
        }
        
        # Tree-based models (excellent for embeddings, no scaling needed)
        models['Random Forest'] = {
            'model': RandomForestRegressor(random_state=self.random_state, n_jobs=self.n_jobs),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [10, 15, None],
                'min_samples_split': [2, 5],
                'max_features': ['sqrt', None]
            }
        }
        
        models['Gradient Boosting'] = {
            'model': GradientBoostingRegressor(random_state=self.random_state),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            }
        }
        
        # XGBoost (if available)
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = {
                'model': xgb.XGBRegressor(
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                    verbosity=0
                ),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0]
                }
            }
        
        # Neural Network (with scaling since it benefits from normalization)
        models['Neural Network'] = {
            'model': Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', MLPRegressor(
                    random_state=self.random_state,
                    max_iter=1000,
                    early_stopping=True,
                    validation_fraction=0.2
                ))
            ]),
            'params': {
                'regressor__hidden_layer_sizes': [
                    (100,), (200,), (100, 50)
                ],
                'regressor__alpha': [0.001, 0.01],
                'regressor__learning_rate': ['adaptive']
            }
        }
        
        return models
    
    def train_model(self, name, model_config):
        """Train a single model with GridSearchCV"""
        print(f"\n{'='*20} Training {name} {'='*20}")
        start_time = time.time()
        
        # Calculate total parameter combinations
        total_combinations = 1
        for param_values in model_config['params'].values():
            total_combinations *= len(param_values)
        
        print(f" Model: {name}")
        print(f" Parameter combinations to test: {total_combinations}")
        print(f" Cross-validation folds: {self.cv_folds}")
        print(f" Total fits required: {total_combinations * self.cv_folds}")
        
        # Estimate time based on model complexity
        estimated_time = self._estimate_training_time(name, total_combinations)
        print(f" Estimated time: {estimated_time:.1f} seconds")
        
        # Setup cross-validation
        cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        # Grid search with more verbose output
        print(f" Starting GridSearchCV for {name}...")
        grid_search = GridSearchCV(
            model_config['model'],
            model_config['params'],
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=self.n_jobs,
            verbose=2  # Increased verbosity
        )
        
        # Fit the model with timing
        fit_start = time.time()
        print(f" Fitting {name} with {total_combinations} parameter combinations...")
        grid_search.fit(self.X_train, self.y_train)
        fit_time = time.time() - fit_start
        
        print(f" GridSearchCV completed for {name} in {fit_time:.1f} seconds")
        print(f" Best CV score: {grid_search.best_score_:.3f}")
        
        # Make predictions
        print(f" Generating predictions for {name}...")
        y_train_pred = grid_search.predict(self.X_train)
        y_test_pred = grid_search.predict(self.X_test)
        
        # Calculate metrics
        print(f" Calculating performance metrics for {name}...")
        train_mse = mean_squared_error(self.y_train, y_train_pred)
        test_mse = mean_squared_error(self.y_test, y_test_pred)
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        
        # Cross-validation score
        print(f" Running additional cross-validation for {name}...")
        cv_scores = cross_val_score(
            grid_search.best_estimator_, 
            self.X_train, 
            self.y_train, 
            cv=cv, 
            scoring='neg_mean_squared_error'
        )
        cv_mse = -cv_scores.mean()
        cv_std = cv_scores.std()
        
        training_time = time.time() - start_time
        
        # Store results
        results = {
            'model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'cv_mse': cv_mse,
            'cv_std': cv_std,
            'training_time': training_time,
            'n_params_tested': len(grid_search.cv_results_['params'])
        }
        
        self.results[name] = results
        
        # Check if this is the best model
        if test_r2 > self.best_score:
            self.best_score = test_r2
            self.best_model = name
            print(f" NEW BEST MODEL: {name} with Test R² = {test_r2:.3f}")
        
        # Print detailed results
        print(f"\n RESULTS for {name}:")
        print(f"   Best parameters: {grid_search.best_params_}")
        print(f"   CV MSE: {cv_mse:.3f} (±{cv_std:.3f})")
        print(f"   Test R²: {test_r2:.3f}")
        print(f"   Test MAE: {test_mae:.2f} minutes")
        print(f"   Training time: {training_time:.1f}s")
        print(f"   Parameters tested: {results['n_params_tested']}")
        print(f"   Actual vs Estimated time: {training_time:.1f}s vs {estimated_time:.1f}s")
        
        return results

    def _estimate_training_time(self, model_name, total_combinations):
        """Estimate training time based on model type and complexity"""
        base_times = {
            'Linear Regression': 0.1,
            'Ridge Regression': 0.2,
            'Random Forest': 3.0,
            'Gradient Boosting': 5.0,
            'XGBoost': 2.0,
            'SVM': 8.0,
            'MLP Neural Network': 10.0,
            'K-Nearest Neighbors': 1.0,
            'Decision Tree': 0.5,
            'Extra Trees': 2.0,
            'AdaBoost': 4.0,
            'Hist Gradient Boosting': 2.5
        }
        
        base_time = base_times.get(model_name, 3.0)  # Default fallback
        sample_factor = len(self.X_train) / 1000  # Scale with dataset size
        cv_factor = self.cv_folds
        
        return base_time * total_combinations * sample_factor * cv_factor

    def train_all_models(self):
        """Train all models and find the best one"""
        print(" STARTING COMPREHENSIVE MODEL SEARCH")
        print("="*60)
        print(f" Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f" Cross-validation folds: {self.cv_folds}")
        print(f" Training samples: {len(self.X_train)}")
        print(f" Test samples: {len(self.X_test)}")
        print(f" Features: {self.X_train.shape[1]}")
        
        models = self.get_model_configs()
        total_models = len(models)
        
        print(f"\n MODELS TO TRAIN: {total_models}")
        for i, name in enumerate(models.keys(), 1):
            print(f"   {i}. {name}")
        
        # Calculate total estimated time
        total_estimated_time = 0
        for name, config in models.items():
            total_combinations = 1
            for param_values in config['params'].values():
                total_combinations *= len(param_values)
            estimated_time = self._estimate_training_time(name, total_combinations)
            total_estimated_time += estimated_time
        
        print(f"\n TOTAL ESTIMATED TIME: {total_estimated_time/60:.1f} minutes")
        print("="*60)
        
        # Train each model
        overall_start_time = time.time()
        for i, (name, config) in enumerate(models.items(), 1):
            elapsed_time = time.time() - overall_start_time
            remaining_models = total_models - i + 1
            
            print(f"\n\n PROGRESS: Model {i}/{total_models} ({(i-1)/total_models*100:.1f}% complete)")
            print(f" Elapsed time: {elapsed_time/60:.1f} minutes")
            
            if i > 1:  # Only estimate after first model
                avg_time_per_model = elapsed_time / (i - 1)
                estimated_remaining = avg_time_per_model * remaining_models
                print(f" Estimated remaining time: {estimated_remaining/60:.1f} minutes")
            
            try:
                model_start = time.time()
                self.train_model(name, config)
                model_time = time.time() - model_start
                
                print(f" Model {i}/{total_models} completed in {model_time/60:.1f} minutes")
                
                # Show current best
                if self.best_model:
                    best_score = self.results[self.best_model]['test_r2']
                    print(f" Current best: {self.best_model} (R² = {best_score:.3f})")
                    
            except Exception as e:
                print(f" ERROR training {name}: {e}")
                continue
        
        total_time = time.time() - overall_start_time
        print(f"\n ALL MODELS COMPLETED!")
        print(f" Total actual time: {total_time/60:.1f} minutes")
        print(f" Models successfully trained: {len(self.results)}")
        
        if self.best_model:
            best_score = self.results[self.best_model]['test_r2']
            print(f" FINAL BEST MODEL: {self.best_model}")
            print(f" Best Test R²: {best_score:.3f}")
        else:
            print(" No models were successfully trained!")
    
    def create_results_summary(self):
        """Create a comprehensive results summary"""
        if not self.results:
            print("No results to summarize!")
            return
        
        # Create DataFrame with results
        summary_data = []
        for name, results in self.results.items():
            summary_data.append({
                'Model': name,
                'Test R²': results['test_r2'],
                'Test MSE': results['test_mse'],
                'Test MAE': results['test_mae'],
                'CV MSE': results['cv_mse'],
                'CV Std': results['cv_std'],
                'Training Time (s)': results['training_time'],
                'Params Tested': results['n_params_tested']
            })
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values('Test R²', ascending=False)
        
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL COMPARISON RESULTS")
        print("="*80)
        print(df.to_string(index=False, float_format='%.3f'))
        
        # Save results
        df.to_csv('../results/model_comparison_results.csv', index=False)
        print(f"\nResults saved to '../results/model_comparison_results.csv'")
        
        # Best model info
        print(f"\n BEST MODEL: {self.best_model}")
        best_results = self.results[self.best_model]
        print(f"Test R²: {best_results['test_r2']:.3f}")
        print(f"Test MAE: {best_results['test_mae']:.2f} minutes")
        print(f"Best parameters: {best_results['best_params']}")
        
        return df
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        if not self.results:
            return
        
        # Prepare data for plotting
        models = list(self.results.keys())
        test_r2_scores = [self.results[m]['test_r2'] for m in models]
        test_mae_scores = [self.results[m]['test_mae'] for m in models]
        training_times = [self.results[m]['training_time'] for m in models]
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # R² scores
        axes[0,0].barh(models, test_r2_scores, color='skyblue')
        axes[0,0].set_xlabel('Test R² Score')
        axes[0,0].set_title('Model Performance Comparison (R²)')
        axes[0,0].axvline(x=max(test_r2_scores), color='red', linestyle='--', alpha=0.7)
        
        # MAE scores
        axes[0,1].barh(models, test_mae_scores, color='lightcoral')
        axes[0,1].set_xlabel('Test MAE (minutes)')
        axes[0,1].set_title('Model Performance Comparison (MAE)')
        axes[0,1].axvline(x=min(test_mae_scores), color='red', linestyle='--', alpha=0.7)
        
        # Training times
        axes[1,0].barh(models, training_times, color='lightgreen')
        axes[1,0].set_xlabel('Training Time (seconds)')
        axes[1,0].set_title('Training Time Comparison')
        axes[1,0].set_xscale('log')
        
        # Performance vs Time scatter
        axes[1,1].scatter(training_times, test_r2_scores, c='purple', alpha=0.7, s=100)
        for i, model in enumerate(models):
            axes[1,1].annotate(model, (training_times[i], test_r2_scores[i]), 
                             rotation=45, fontsize=8, alpha=0.8)
        axes[1,1].set_xlabel('Training Time (seconds)')
        axes[1,1].set_ylabel('Test R² Score')
        axes[1,1].set_title('Performance vs Training Time')
        axes[1,1].set_xscale('log')
        
        plt.tight_layout()
        plt.savefig('../results/comprehensive_model_comparison.png', dpi=300, bbox_inches='tight')
        print("Visualizations saved as '../results/comprehensive_model_comparison.png'")
        
        # Best model predictions plot
        if self.best_model in self.results:
            best_model_obj = self.results[self.best_model]['model']
            y_test_pred = best_model_obj.predict(self.X_test)
            
            plt.figure(figsize=(10, 6))
            plt.subplot(1, 2, 1)
            plt.scatter(self.y_test, y_test_pred, alpha=0.6, color='blue')
            plt.plot([self.y_test.min(), self.y_test.max()], 
                    [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
            plt.xlabel('Actual Duration (minutes)')
            plt.ylabel('Predicted Duration (minutes)')
            plt.title(f'Best Model: {self.best_model}')
            
            # Residuals
            plt.subplot(1, 2, 2)
            residuals = self.y_test - y_test_pred
            plt.scatter(y_test_pred, residuals, alpha=0.6, color='orange')
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted Duration (minutes)')
            plt.ylabel('Residuals')
            plt.title('Residuals Plot')
            
            plt.tight_layout()
            plt.savefig('../results/best_model_predictions.png', dpi=300, bbox_inches='tight')
            print("Best model predictions saved as '../results/best_model_predictions.png'")
    
    def save_best_model(self):
        """Save the best model and all results"""
        if not self.best_model:
            print("No best model to save!")
            return
        
        # Save best model
        best_model_data = {
            'model': self.results[self.best_model]['model'],
            'model_name': self.best_model,
            'best_params': self.results[self.best_model]['best_params'],
            'test_r2': self.results[self.best_model]['test_r2'],
            'test_mae': self.results[self.best_model]['test_mae'],
            'training_timestamp': datetime.now().isoformat()
        }
        
        with open('../models/best_model.pkl', 'wb') as f:
            pickle.dump(best_model_data, f)
        
        # Save all results
        with open('../models/all_model_results.pkl', 'wb') as f:
            pickle.dump(self.results, f)
        
        print(f"Best model ({self.best_model}) saved as '../models/best_model.pkl'")
        print("All results saved as '../models/all_model_results.pkl'")

def main():
    print(" COMPREHENSIVE MODEL SEARCH AND COMPARISON")
    print("="*60)
    
    # Initialize the search
    search = ComprehensiveModelSearch(cv_folds=5, n_jobs=-1)
    
    # Load data
    X, y = search.load_data()
    if X is None:
        return
    
    # Prepare data
    search.prepare_data(X, y)
    
    # Train all models
    start_time = time.time()
    search.train_all_models()
    total_time = time.time() - start_time
    
    # Create results summary
    results_df = search.create_results_summary()
    
    # Create visualizations
    search.create_visualizations()
    
    # Save best model
    search.save_best_model()
    
    print(f"\n SEARCH COMPLETED!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Models trained: {len(search.results)}")
    print(f"Best model: {search.best_model}")
    print("\nFiles created:")
    print("- results/model_comparison_results.csv")
    print("- results/comprehensive_model_comparison.png")
    print("- results/best_model_predictions.png") 
    print("- models/best_model.pkl")
    print("- models/all_model_results.pkl")

if __name__ == "__main__":
    main()
