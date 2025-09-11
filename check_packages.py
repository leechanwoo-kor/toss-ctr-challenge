"""Check RAPIDS/Merlin packages availability"""

packages = [
    'xgboost', 'dask', 'numpy', 'scikit-learn'
]

for pkg in packages:
    try:
        module = __import__(pkg)
        version = getattr(module, '__version__', 'unknown')
        print(f"OK {pkg:15} {version}")
    except ImportError:
        print(f"NO {pkg:15} NOT INSTALLED")