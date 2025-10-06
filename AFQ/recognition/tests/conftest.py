def pytest_configure(config):
    """Configure logging before any tests run"""
    import logging
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True  # Override any existing configuration
    )
