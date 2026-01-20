#!/usr/bin/env python3
"""
TransitPulse Lite Startup Script

This script provides a standardized way to launch the TransitPulse Lite application
with proper error handling and logging.
"""

import sys
import os
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('transitpulse.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        "streamlit",
        "pandas", 
        "numpy",
        "requests",
        "pydeck",
        "sklearn",
        "gtfs_realtime_pb2"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == "gtfs_realtime_pb2":
                # Special case for protobuf module
                import gtfs_realtime_pb2
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.error("Please install dependencies using: pip install -r requirements.txt")
        return False
    
    logger.info("All dependencies are available")
    return True


def run_application(host="0.0.0.0", port=8501, debug=False):
    """Run the TransitPulse Lite application."""
    try:
        import streamlit.web.bootstrap as bootstrap
        import streamlit.runtime.source_util as source_util
        
        # Add the project root to the path
        project_root = Path(__file__).parent.resolve()
        sys.path.insert(0, str(project_root))
        
        # Set the main script path
        main_script_path = str(project_root / "app.py")
        
        # Clear any cached sources
        source_util._cached_pages.clear()
        
        # Prepare command line args for streamlit
        command_line = [
            "streamlit", "run", 
            "--server.address", host,
            "--server.port", str(port),
        ]
        
        if debug:
            command_line.extend(["--global.developmentMode", "true"])
        
        logger.info(f"Starting TransitPulse Lite on http://{host}:{port}")
        logger.info("Application logs will be written to transitpulse.log")
        
        # Start the Streamlit app
        bootstrap.run(
            main_script_path=main_script_path,
            command_line=command_line[1:],  # Skip "streamlit" command
            args=[],  # Additional script arguments
            flag_options={}  # Streamlit config options
        )
        
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}", exc_info=True)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="TransitPulse Lite - Public Transit Monitoring")
    parser.add_argument("--host", default="0.0.0.0", help="Host address to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8501, help="Port to run the application on (default: 8501)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    logger.info("Starting TransitPulse Lite application...")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Run the application
    run_application(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()