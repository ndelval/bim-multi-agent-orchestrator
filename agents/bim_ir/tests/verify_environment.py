"""
Environment verification script for BIM-IR agent testing.

Checks all prerequisites before running tests:
- Python version
- Dependencies installed
- Datasets present
- API keys configured
- Module imports working
"""

import os
import sys
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, will use system environment variables
    pass

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def print_header(text):
    """Print formatted section header."""
    print("\n" + "="*70)
    print(text)
    print("="*70)


def check_python_version():
    """Verify Python version is 3.8+."""
    print_header("Checking Python Version")

    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version_str} (requirement: 3.8+)")
        return True
    else:
        print(f"❌ Python {version_str} (requirement: 3.8+)")
        print("   Solution: Upgrade Python to 3.8 or higher")
        return False


def check_dependencies():
    """Verify required packages are installed."""
    print_header("Checking Dependencies")

    required = {
        "openai": "OpenAI API client",
        "pydantic": "Data validation",
        "pandas": "Dataset loading"
    }

    all_ok = True
    for package, description in required.items():
        try:
            mod = __import__(package)
            version = getattr(mod, "__version__", "unknown")
            print(f"✅ {package} {version} - {description}")
        except ImportError:
            print(f"❌ {package} - {description}")
            print(f"   Solution: pip install {package}")
            all_ok = False

    return all_ok


def check_module_imports():
    """Verify all BIM-IR modules can be imported."""
    print_header("Checking Module Imports")

    modules = [
        ("agents.bim_ir.blocks", "IntentClassifier"),
        ("agents.bim_ir.blocks", "ParamExtractor"),
        ("agents.bim_ir.blocks", "ValueResolver"),
        ("agents.bim_ir.blocks", "Retriever"),
        ("agents.bim_ir.blocks", "Summarizer"),
        ("agents.bim_ir.utils", "ViewerHighlighter"),
        ("agents.bim_ir.models", "IntentResult"),
        ("agents.bim_ir.models", "ParameterResult"),
        ("agents.bim_ir.models", "ResolvedValues"),
        ("agents.bim_ir.models", "RetrieverResult"),
        ("agents.bim_ir.models", "SummarizerResult"),
        ("agents.bim_ir.models", "HighlightResult")
    ]

    all_ok = True
    for module_name, class_name in modules:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✅ {module_name}.{class_name}")
        except ImportError as e:
            print(f"❌ {module_name}.{class_name}")
            print(f"   Error: {e}")
            print(f"   Solution: Check PYTHONPATH or module structure")
            all_ok = False
        except AttributeError as e:
            print(f"❌ {module_name}.{class_name}")
            print(f"   Error: {e}")
            print(f"   Solution: Check __init__.py exports")
            all_ok = False

    return all_ok


def check_datasets():
    """Verify dataset CSV files exist and have correct format."""
    print_header("Checking Datasets")

    dataset_dir = Path(__file__).parent.parent / "datasets"

    datasets = {
        "intent_examples.csv": ["query", "intent", "is_bim"],
        "parameter_examples.csv": ["query", "intent", "filter_para", "proj_para"],
        "value_resolution_examples.csv": ["query", "filter_para", "normalized_values"]
    }

    all_ok = True

    # Check directory exists
    if not dataset_dir.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        print(f"   Solution: Create directory and add CSV files")
        return False

    print(f"📁 Dataset directory: {dataset_dir}")

    # Check each dataset file
    for filename, required_columns in datasets.items():
        filepath = dataset_dir / filename

        if not filepath.exists():
            print(f"❌ {filename} - File not found")
            print(f"   Solution: Add CSV file to {dataset_dir}")
            all_ok = False
            continue

        # Verify columns
        try:
            import pandas as pd
            df = pd.read_csv(filepath)

            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                print(f"⚠️  {filename} - Missing columns: {missing_columns}")
                print(f"   Current columns: {list(df.columns)}")
                print(f"   Required columns: {required_columns}")
                all_ok = False
            else:
                print(f"✅ {filename} - {len(df)} examples, columns OK")

        except Exception as e:
            print(f"❌ {filename} - Error reading file")
            print(f"   Error: {e}")
            all_ok = False

    return all_ok


def check_api_keys():
    """Verify API keys are configured (presence check only)."""
    print_header("Checking API Keys")

    openai_key = os.environ.get("OPENAI_API_KEY")

    if openai_key:
        masked_key = openai_key[:7] + "..." + openai_key[-4:] if len(openai_key) > 11 else "***"
        print(f"✅ OPENAI_API_KEY is set ({masked_key})")
        print(f"   Note: Key validity not checked (will be tested during actual API calls)")
        return True
    else:
        print(f"❌ OPENAI_API_KEY is not set")
        print(f"   Solution: export OPENAI_API_KEY='sk-...'")
        print(f"   Get key from: https://platform.openai.com/api-keys")
        return False


def check_directory_structure():
    """Verify project directory structure is correct."""
    print_header("Checking Directory Structure")

    base_dir = Path(__file__).parent.parent

    required_dirs = [
        "blocks",
        "models",
        "utils",
        "datasets",
        "tests",
        "llm"
    ]

    all_ok = True
    for dirname in required_dirs:
        dirpath = base_dir / dirname
        if dirpath.exists():
            print(f"✅ {dirname}/ exists")
        else:
            print(f"❌ {dirname}/ not found")
            print(f"   Expected at: {dirpath}")
            all_ok = False

    return all_ok


def check_test_files():
    """Verify test files exist."""
    print_header("Checking Test Files")

    test_dir = Path(__file__).parent

    test_files = [
        "test_summarizer.py",
        "test_viewer_highlighter.py"
    ]

    all_ok = True
    for filename in test_files:
        filepath = test_dir / filename
        if filepath.exists():
            print(f"✅ {filename} exists")
        else:
            print(f"❌ {filename} not found")
            all_ok = False

    return all_ok


def run_quick_import_test():
    """Try a quick import of all blocks to catch any immediate issues."""
    print_header("Quick Import Test")

    try:
        from agents.bim_ir.blocks import (
            IntentClassifier,
            ParamExtractor,
            ValueResolver,
            Summarizer
        )
        from agents.bim_ir.utils import ViewerHighlighter
        from agents.bim_ir.models import (
            IntentResult,
            SummarizerResult,
            HighlightResult
        )

        print("✅ All core imports successful")
        return True

    except Exception as e:
        print(f"❌ Import failed: {e}")
        print(f"   This indicates a structural problem with the modules")
        return False


def main():
    """Run all verification checks."""
    print("\n" + "="*70)
    print("BIM-IR Agent Environment Verification")
    print("="*70)
    print("\nThis script checks all prerequisites for testing the BIM-IR agent.")
    print("Each check shows ✅ (pass) or ❌ (fail) with suggested solutions.")

    results = {}

    # Run all checks
    results["Python Version"] = check_python_version()
    results["Dependencies"] = check_dependencies()
    results["Module Imports"] = check_module_imports()
    results["Datasets"] = check_datasets()
    results["API Keys"] = check_api_keys()
    results["Directory Structure"] = check_directory_structure()
    results["Test Files"] = check_test_files()
    results["Quick Import"] = run_quick_import_test()

    # Summary
    print_header("Verification Summary")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for check, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {check}")

    print(f"\nTotal: {passed}/{total} checks passed")

    if passed == total:
        print("\n🎉 All checks passed! System is ready for testing.")
        print("\nNext steps:")
        print("1. Run unit tests: python agents/bim_ir/tests/test_summarizer.py")
        print("2. Run integration tests (requires OpenAI API)")
        return 0
    else:
        print(f"\n⚠️  {total - passed} check(s) failed. Fix issues before testing.")
        print("\nReview error messages above for solutions.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
