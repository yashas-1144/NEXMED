"""
__init__.py — Makes `backend` a Python package.

Importing this package does NOT load the model automatically.
The model is loaded explicitly via `backend.model_loader.load_model()`
called from `backend.app.preload_model()` at server startup.
"""
