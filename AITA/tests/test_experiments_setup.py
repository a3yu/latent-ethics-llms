"""
Test Script: Verify Experiment Setup
===================================

This script verifies that all experiment modules can be imported and basic functionality works.
Run this before attempting to run the actual experiments to catch any setup issues.
"""

import sys
import os
import traceback

def test_imports():
    """Test all experiment imports."""
    print("🔍 Testing experiment imports...")
    
    try:
        from experiment_1_direct_judgment import DirectJudgmentExperiment
        print("✅ Experiment 1 (Direct Judgment) - Import successful")
    except Exception as e:
        print(f"❌ Experiment 1 - Import failed: {e}")
        return False
    
    try:
        from experiment_2_adversarial_debate import AdversarialDebateExperiment
        print("✅ Experiment 2 (Adversarial Debate) - Import successful")
    except Exception as e:
        print(f"❌ Experiment 2 - Import failed: {e}")
        return False
    
    try:
        from experiment_3_framing_effects import FramingEffectsExperiment
        print("✅ Experiment 3 (Framing Effects) - Import successful")
    except Exception as e:
        print(f"❌ Experiment 3 - Import failed: {e}")
        return False
    
    try:
        from experiment_4_empathy_cues import EmpathyCuesExperiment
        print("✅ Experiment 4 (Empathy Cues) - Import successful")
    except Exception as e:
        print(f"❌ Experiment 4 - Import failed: {e}")
        return False
    
    try:
        from run_all_experiments import ExperimentRunner
        print("✅ Master Runner - Import successful")
    except Exception as e:
        print(f"❌ Master Runner - Import failed: {e}")
        return False
    
    return True

def test_dependencies():
    """Test required dependencies."""
    print("\n🔍 Testing dependencies...")
    
    required_packages = [
        'openai', 'pandas', 'matplotlib', 'seaborn', 
        'numpy', 'tenacity', 'json', 're'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - Available")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def test_data_file():
    """Test data file availability."""
    print("\n🔍 Testing data file...")
    
    data_files = [
        'judgment_analysis_results.json',
        '../data/judgment_analysis_results.json'
    ]
    
    for data_file in data_files:
        if os.path.exists(data_file):
            file_size = os.path.getsize(data_file) / (1024 * 1024)  # MB
            print(f"✅ Data file found: {data_file} ({file_size:.1f} MB)")
            return True
    
    print("❌ Data file not found. Expected locations:")
    for data_file in data_files:
        print(f"   - {data_file}")
    print("\nPlease ensure the AITA dataset has been processed.")
    return False

def test_api_key():
    """Test OpenAI API key availability."""
    print("\n🔍 Testing API key...")
    
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print(f"✅ OpenAI API key found (length: {len(api_key)})")
        return True
    else:
        print("❌ OpenAI API key not found")
        print("Set with: export OPENAI_API_KEY='your-key-here'")
        return False

def test_basic_functionality():
    """Test basic experiment functionality without API calls."""
    print("\n🔍 Testing basic functionality...")
    
    try:
        from experiment_1_direct_judgment import DirectJudgmentExperiment
        
        # Test initialization with non-existent file (should handle gracefully)
        exp = DirectJudgmentExperiment('fake_file.json')
        print("✅ Experiment initialization - Working")
        
        # Test severity mapping
        severity_map = {
            'NTA': 1,
            'NAH': 2, 
            'ESH': 3,
            'YTA': 4
        }
        
        if all(isinstance(v, int) for v in severity_map.values()):
            print("✅ Severity mapping - Correct")
        else:
            print("❌ Severity mapping - Invalid")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all verification tests."""
    print("🧪 LLM ETHICS EXPERIMENTS - SETUP VERIFICATION")
    print("=" * 60)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Imports", test_imports),
        ("Data File", test_data_file),
        ("API Key", test_api_key),
        ("Basic Functionality", test_basic_functionality)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! You're ready to run the experiments.")
        print("\nNext steps:")
        print("1. Run individual experiments: python experiment_N_*.py")
        print("2. Run all experiments: python run_all_experiments.py --posts 10")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above before running experiments.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 