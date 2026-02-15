"""
PROBLEM 4: SPORTS vs POLITICS CLASSIFIER
Author: B23CM1011
Date: February 2026

This program implements a text classification system to classify documents as Sports or Politics
using multiple machine learning techniques and compares their performance.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report, 
                             roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==================== DATA COLLECTION ====================

def create_sports_politics_dataset():
    """
    Creates a balanced dataset of Sports and Politics texts.
    In a real scenario, this would be collected from news sources, articles, etc.
    """
    
    # SPORTS TEXTS
    sports_texts = [
        "Messi scored an incredible hat-trick in the Champions League final",
        "The cricket match between India and Pakistan was thrilling",
        "Roger Federer won the Wimbledon tournament",
        "LeBron James led the Lakers to victory with 35 points",
        "Manchester United defeated Arsenal in the Premier League",
        "Usain Bolt broke the world record in 100 meters sprint",
        "The rugby team won the World Cup tournament",
        "Serena Williams defeated her opponent in straight sets",
        "Real Madrid dominated the El Clasico match",
        "The NFL Super Bowl was watched by millions of fans",
        "Tiger Woods won the Masters golf tournament",
        "The basketball final ended in overtime",
        "Novak Djokovic claimed another Grand Slam title",
        "The football team qualified for the World Cup",
        "Lewis Hamilton secured another Formula One victory",
        "The cricket team scored 350 runs in one inning",
        "The volleyball championship was won by the home team",
        "Basketball players demonstrated excellent teamwork",
        "The tennis player showed outstanding performance",
        "The hockey tournament was very competitive",
        "Swimming records were broken at the Olympics",
        "The soccer team executed perfect tactics",
        "The badminton player reached the quarterfinals",
        "Baseball game ended with a home run",
        "The athletic team trained rigorously for the championship",
        "Victory was secured in the final seconds",
        "The sports event attracted a massive crowd",
        "Players showed exceptional skill and dedication",
        "The tournament attracted top athletes from around the world",
        "The championship match was broadcast nationwide",
    ]
    
    # POLITICS TEXTS
    politics_texts = [
        "The government announced new policies for economic growth",
        "Parliament passed a new bill on taxation",
        "The president addressed the nation on foreign policy",
        "Opposition leaders criticized the government's approach",
        "Elections were held in the country",
        "The senate approved the proposed legislation",
        "The minister resigned from office",
        "Political debates focused on healthcare reform",
        "The ruling party won the majority in elections",
        "Foreign relations improved between two nations",
        "The government introduced new environmental regulations",
        "Political parties discussed trade agreements",
        "The cabinet reshuffled government positions",
        "Public policy was reformed in the education sector",
        "Government spending was debated in parliament",
        "International diplomacy played a key role in peace talks",
        "The law was amended to protect citizen rights",
        "Political leaders met for peace negotiations",
        "Constitutional changes were proposed",
        "The government implemented fiscal policies",
        "Political consensus was reached on infrastructure",
        "The ministry launched a new initiative",
        "Democratic values were emphasized in the speech",
        "Governance reforms were introduced",
        "The parliament debated crucial political matters",
        "International relations were strengthened",
        "Political stability was achieved after negotiations",
        "Government transparency measures were implemented",
        "Strategic alliances were formed between nations",
        "The administration announced development programs",
    ]
    
    # Combine and create labels
    texts = sports_texts + politics_texts
    labels = [0] * len(sports_texts) + [1] * len(politics_texts)  # 0=Sports, 1=Politics
    
    # Shuffle the data
    combined = list(zip(texts, labels))
    np.random.seed(42)
    np.random.shuffle(combined)
    texts, labels = zip(*combined)
    texts = list(texts)
    labels = list(labels)
    
    return texts, labels

# ==================== FEATURE EXTRACTION ====================

class FeatureExtractor:
    """Handles all feature extraction methods"""
    
    @staticmethod
    def tfidf_features(texts_train, texts_test, max_features=100, ngram_range=(1, 2)):
        """TF-IDF feature extraction with n-grams"""
        vectorizer = TfidfVectorizer(max_features=max_features, 
                                     ngram_range=ngram_range,
                                     lowercase=True,
                                     stop_words='english')
        X_train = vectorizer.fit_transform(texts_train)
        X_test = vectorizer.transform(texts_test)
        return X_train, X_test, vectorizer
    
    @staticmethod
    def bow_features(texts_train, texts_test, max_features=100):
        """Bag of Words feature extraction"""
        vectorizer = CountVectorizer(max_features=max_features,
                                     lowercase=True,
                                     stop_words='english')
        X_train = vectorizer.fit_transform(texts_train)
        X_test = vectorizer.transform(texts_test)
        return X_train, X_test, vectorizer
    
    @staticmethod
    def ngram_features(texts_train, texts_test, max_features=100, ngram_range=(2, 3)):
        """N-gram feature extraction"""
        vectorizer = TfidfVectorizer(max_features=max_features,
                                     ngram_range=ngram_range,
                                     lowercase=True,
                                     stop_words='english')
        X_train = vectorizer.fit_transform(texts_train)
        X_test = vectorizer.transform(texts_test)
        return X_train, X_test, vectorizer

# ==================== ML MODELS ====================

class SportsPolicticsClassifier:
    """Main classifier with multiple algorithms"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.feature_extractors = {}
        
    def train_models(self, X_train, X_test, y_train, y_test, feature_name):
        """Train multiple ML models"""
        
        # 1. Naive Bayes
        nb_model = MultinomialNB()
        nb_model.fit(X_train, y_train)
        nb_pred = nb_model.predict(X_test)
        
        # 2. Support Vector Machine (SVM)
        svm_model = LinearSVC(random_state=42, max_iter=2000)
        svm_model.fit(X_train, y_train)
        svm_pred = svm_model.predict(X_test)
        
        # 3. Logistic Regression
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        
        # 4. Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        
        # Store predictions and models
        predictions = {
            'Naive Bayes': (nb_pred, nb_model),
            'SVM': (svm_pred, svm_model),
            'Logistic Regression': (lr_pred, lr_model),
            'Random Forest': (rf_pred, rf_model)
        }
        
        # Evaluate each model
        self.results[feature_name] = {}
        for model_name, (predictions_arr, model_obj) in predictions.items():
            metrics = self.evaluate_model(predictions_arr, y_test, model_name)
            self.results[feature_name][model_name] = metrics
            self.models[f"{feature_name}_{model_name}"] = model_obj
        
        return self.results[feature_name]
    
    @staticmethod
    def evaluate_model(y_pred, y_test, model_name):
        """Calculate performance metrics"""
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1-Score': f1_score(y_test, y_pred, zero_division=0),
            'Confusion Matrix': confusion_matrix(y_test, y_pred),
            'Classification Report': classification_report(y_test, y_pred, zero_division=0)
        }
        return metrics
    
    def print_results_summary(self):
        """Print summary of all results"""
        print("\n" + "="*80)
        print("COMPREHENSIVE RESULTS SUMMARY".center(80))
        print("="*80 + "\n")
        
        results_data = []
        for feature_method, models_dict in self.results.items():
            print(f"\n{'Feature Method: ' + feature_method}")
            print("-" * 80)
            for model_name, metrics in models_dict.items():
                print(f"\n  {model_name}:")
                print(f"    Accuracy:  {metrics['Accuracy']:.4f}")
                print(f"    Precision: {metrics['Precision']:.4f}")
                print(f"    Recall:    {metrics['Recall']:.4f}")
                print(f"    F1-Score:  {metrics['F1-Score']:.4f}")
                
                results_data.append({
                    'Feature Method': feature_method,
                    'Algorithm': model_name,
                    'Accuracy': metrics['Accuracy'],
                    'Precision': metrics['Precision'],
                    'Recall': metrics['Recall'],
                    'F1-Score': metrics['F1-Score']
                })
        
        return pd.DataFrame(results_data)

# ==================== VISUALIZATION ====================

def create_visualizations(classifier, results_df):
    """Create comprehensive visualizations"""
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 12)
    
    # 1. Accuracy Comparison across Feature Methods
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Subplot 1: Accuracy
    ax1 = axes[0, 0]
    feature_methods = results_df['Feature Method'].unique()
    x_pos = np.arange(len(feature_methods))
    width = 0.2
    
    for i, algo in enumerate(['Naive Bayes', 'SVM', 'Logistic Regression', 'Random Forest']):
        accuracies = []
        for method in feature_methods:
            acc = results_df[(results_df['Feature Method'] == method) & 
                            (results_df['Algorithm'] == algo)]['Accuracy'].values
            accuracies.append(acc[0] if len(acc) > 0 else 0)
        ax1.bar(x_pos + i*width, accuracies, width, label=algo)
    
    ax1.set_xlabel('Feature Method')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy Comparison Across Algorithms and Feature Methods')
    ax1.set_xticks(x_pos + width * 1.5)
    ax1.set_xticklabels(feature_methods)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Subplot 2: F1-Score
    ax2 = axes[0, 1]
    for i, algo in enumerate(['Naive Bayes', 'SVM', 'Logistic Regression', 'Random Forest']):
        f1_scores = []
        for method in feature_methods:
            f1 = results_df[(results_df['Feature Method'] == method) & 
                           (results_df['Algorithm'] == algo)]['F1-Score'].values
            f1_scores.append(f1[0] if len(f1) > 0 else 0)
        ax2.bar(x_pos + i*width, f1_scores, width, label=algo)
    
    ax2.set_xlabel('Feature Method')
    ax2.set_ylabel('F1-Score')
    ax2.set_title('F1-Score Comparison Across Algorithms and Feature Methods')
    ax2.set_xticks(x_pos + width * 1.5)
    ax2.set_xticklabels(feature_methods)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Subplot 3: Precision
    ax3 = axes[1, 0]
    for i, algo in enumerate(['Naive Bayes', 'SVM', 'Logistic Regression', 'Random Forest']):
        precisions = []
        for method in feature_methods:
            prec = results_df[(results_df['Feature Method'] == method) & 
                             (results_df['Algorithm'] == algo)]['Precision'].values
            precisions.append(prec[0] if len(prec) > 0 else 0)
        ax3.bar(x_pos + i*width, precisions, width, label=algo)
    
    ax3.set_xlabel('Feature Method')
    ax3.set_ylabel('Precision')
    ax3.set_title('Precision Comparison Across Algorithms and Feature Methods')
    ax3.set_xticks(x_pos + width * 1.5)
    ax3.set_xticklabels(feature_methods)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Subplot 4: Recall
    ax4 = axes[1, 1]
    for i, algo in enumerate(['Naive Bayes', 'SVM', 'Logistic Regression', 'Random Forest']):
        recalls = []
        for method in feature_methods:
            rec = results_df[(results_df['Feature Method'] == method) & 
                            (results_df['Algorithm'] == algo)]['Recall'].values
            recalls.append(rec[0] if len(rec) > 0 else 0)
        ax4.bar(x_pos + i*width, recalls, width, label=algo)
    
    ax4.set_xlabel('Feature Method')
    ax4.set_ylabel('Recall')
    ax4.set_title('Recall Comparison Across Algorithms and Feature Methods')
    ax4.set_xticks(x_pos + width * 1.5)
    ax4.set_xticklabels(feature_methods)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: metrics_comparison.png")
    plt.close()

def create_heatmap(results_df):
    """Create heatmap of accuracies"""
    plt.figure(figsize=(10, 6))
    
    pivot_table = results_df.pivot_table(values='Accuracy', 
                                         index='Algorithm', 
                                         columns='Feature Method')
    
    sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='YlGnBu', 
                cbar_kws={'label': 'Accuracy'}, linewidths=0.5)
    plt.title('Accuracy Heatmap: Algorithms vs Feature Methods')
    plt.tight_layout()
    plt.savefig('accuracy_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: accuracy_heatmap.png")
    plt.close()

# ==================== MAIN EXECUTION ====================

def main():
    print("\n" + "="*80)
    print("SPORTS vs POLITICS TEXT CLASSIFICATION SYSTEM".center(80))
    print("="*80)
    print(f"\nExecution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Create Dataset
    print("\n[STEP 1] Creating Dataset...")
    texts, labels = create_sports_politics_dataset()
    print(f"  ✓ Dataset size: {len(texts)} documents")
    print(f"    - Sports documents: {sum(1 for l in labels if l == 0)}")
    print(f"    - Politics documents: {sum(1 for l in labels if l == 1)}")
    
    # Step 2: Train-Test Split
    print("\n[STEP 2] Splitting Dataset...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    print(f"  ✓ Training set: {len(X_train_text)} documents")
    print(f"  ✓ Testing set: {len(X_test_text)} documents")
    
    # Step 3: Feature Extraction & Model Training
    print("\n[STEP 3] Feature Extraction & Model Training...")
    classifier = SportsPolicticsClassifier()
    feature_extractor = FeatureExtractor()
    
    # TF-IDF Features
    print("\n  → Extracting TF-IDF Features (1-2 grams)...")
    X_train_tfidf, X_test_tfidf, _ = feature_extractor.tfidf_features(
        X_train_text, X_test_text, max_features=100, ngram_range=(1, 2)
    )
    print(f"    ✓ Feature shape: {X_train_tfidf.shape}")
    classifier.train_models(X_train_tfidf, X_test_tfidf, y_train, y_test, "TF-IDF (1-2 grams)")
    
    # Bag of Words Features
    print("\n  → Extracting Bag of Words Features...")
    X_train_bow, X_test_bow, _ = feature_extractor.bow_features(
        X_train_text, X_test_text, max_features=100
    )
    print(f"    ✓ Feature shape: {X_train_bow.shape}")
    classifier.train_models(X_train_bow, X_test_bow, y_train, y_test, "Bag of Words")
    
    # N-gram Features
    print("\n  → Extracting N-gram Features (2-3 grams)...")
    X_train_ngram, X_test_ngram, _ = feature_extractor.ngram_features(
        X_train_text, X_test_text, max_features=100, ngram_range=(2, 3)
    )
    print(f"    ✓ Feature shape: {X_train_ngram.shape}")
    classifier.train_models(X_train_ngram, X_test_ngram, y_train, y_test, "N-grams (2-3)")
    
    # Step 4: Results Summary
    print("\n[STEP 4] Generating Results Summary...")
    results_df = classifier.print_results_summary()
    
    # Step 5: Visualizations
    print("\n[STEP 5] Creating Visualizations...")
    create_visualizations(classifier, results_df)
    create_heatmap(results_df)
    
    # Step 6: Best Model
    print("\n[STEP 6] Best Performing Model:")
    best_idx = results_df['F1-Score'].idxmax()
    best_row = results_df.loc[best_idx]
    print(f"  ★ Feature Method: {best_row['Feature Method']}")
    print(f"  ★ Algorithm: {best_row['Algorithm']}")
    print(f"  ★ F1-Score: {best_row['F1-Score']:.4f}")
    print(f"  ★ Accuracy: {best_row['Accuracy']:.4f}")
    
    # Save results
    results_df.to_csv('B23CM1011_prob4_results.csv', index=False)
    print("\n✓ Results saved: B23CM1011_prob4_results.csv")
    
    print("\n" + "="*80)
    print("EXECUTION COMPLETED SUCCESSFULLY".center(80))
    print("="*80 + "\n")
    
    return classifier, results_df

if __name__ == "__main__":
    classifier, results_df = main()
