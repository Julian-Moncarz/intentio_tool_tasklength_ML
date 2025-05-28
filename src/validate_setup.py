#!/usr/bin/env python3
"""
Setup Validation Script
Checks if all dependencies and files are ready before running the ML pipeline.
"""

import sys
import os
import importlib
import pandas as pd

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'pandas',
        'numpy', 
        'sklearn',
        'sentence_transformers',
        'matplotlib',
        'seaborn',
        'pickle'
    ]
    
    optional_packages = [
        'xgboost',
        'torch'
    ]
    
    print("🔍 Checking Required Dependencies...")
    missing_required = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_required.append(package)
    
    print("\n🔍 Checking Optional Dependencies...")
    missing_optional = []
    
    for package in optional_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"⚠️  {package} - MISSING (optional)")
            missing_optional.append(package)
    
    if missing_required:
        print(f"\n❌ Missing required packages: {missing_required}")
        print("Install with: pip install " + " ".join(missing_required))
        return False
    
    if missing_optional:
        print(f"\n⚠️  Missing optional packages: {missing_optional}")
        print("For better performance, install with: pip install " + " ".join(missing_optional))
    
    return True

def check_data_files():
    """Check if required data files exist and are valid"""
    print("📁 Checking data files...")
    
    # Check logs.csv
    if not os.path.exists('../data/logs.csv'):
        print("❌ logs.csv - MISSING")
        print("   Expected location: data/logs.csv")
        return False
    else:
        try:
            df = pd.read_csv('../data/logs.csv')
            print(f"✅ logs.csv - {len(df)} rows")
            
            # Check required columns
            required_cols = ['Intent', 'Duration(min)']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"❌ Missing columns in logs.csv: {missing_cols}")
                return False
            
            # Check for valid data
            valid_rows = df.dropna(subset=required_cols)
            if len(valid_rows) < 10:
                print(f"⚠️  Only {len(valid_rows)} valid rows in logs.csv. Need at least 10 for training.")
                return False
            else:
                print(f"✅ Data validation - {len(valid_rows)} valid rows")
                return True
                
        except Exception as e:
            print(f"❌ Error reading logs.csv: {e}")
            return False

def check_generated_files():
    """Check if embeddings have been generated"""
    print("\n🔍 Checking Generated Files...")
    
    embeddings_exist = os.path.exists('../data/embeddings_data.pkl')
    
    if embeddings_exist:
        print("✅ embeddings_data.pkl - EXISTS")
        return True
    else:
        print("⚠️  embeddings_data.pkl - NOT FOUND")
        print("   Run 'python generate_embeddings.py' to create this file")
        return False

def check_script_permissions():
    """Check if Python scripts are executable"""
    print("\n🔍 Checking Script Permissions...")
    
    scripts = [
        'generate_embeddings.py',
        'train_random_forest.py', 
        'comprehensive_model_search.py'
    ]
    
    all_exist = True
    for script in scripts:
        if os.path.exists(script):
            print(f"✅ {script}")
        else:
            print(f"❌ {script} - MISSING")
            all_exist = False
    
    return all_exist

def main():
    print("🚀 ML PROJECT SETUP VALIDATION")
    print("="*50)
    
    checks = [
        ("Dependencies", check_dependencies),
        ("Data Files", check_data_files), 
        ("Scripts", check_script_permissions),
        ("Generated Files", check_generated_files)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            all_passed = all_passed and result
        except Exception as e:
            print(f"❌ Error in {check_name} check: {e}")
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 ALL CHECKS PASSED!")
        print("\n📋 EXECUTION ORDER:")
        print("1. python generate_embeddings.py")
        print("2. python train_random_forest.py")
        print("3. python comprehensive_model_search.py")
    else:
        print("❌ SOME CHECKS FAILED!")
        print("Please fix the issues above before running the ML pipeline.")
        
        print("\n🔧 QUICK FIXES:")
        print("- Install missing packages: pip install -r requirements.txt") 
        print("- Ensure logs.csv has 'Intent' and 'Duration(min)' columns")
        print("- Run generate_embeddings.py first to create embeddings_data.pkl")
    
    print("="*50)
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
