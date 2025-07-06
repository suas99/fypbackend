"""
Simple script to start the Mental Health Analysis backend server.
"""
import os
import sys

def check_venv():
    """Check if running in a virtual environment"""
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if not in_venv:
        print("\n⚠️  WARNING: Virtual environment not detected!")
        print("It's recommended to run this server in a virtual environment.")
        print("You can create and activate one with:")
        print("    python -m venv venv")
        if os.name == 'nt':  # Windows
            print("    venv\\Scripts\\activate")
        else:  # macOS/Linux
            print("    source venv/bin/activate")
        
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            sys.exit(0)

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import flask
        import flask_cors
        import numpy
        import tensorflow
        print("✅ All required packages are installed.")
    except ImportError as e:
        print(f"\n❌ Missing dependency: {e}")
        print("Please install required packages:")
        print("    pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    # Print welcome message
    print("\n" + "="*60)
    print("  Mental Health Analysis Backend Server")
    print("="*60)
    
    # Check virtual environment
    check_venv()
    
    # Check dependencies
    check_dependencies()
    
    # Start the server
    print("\nStarting backend server...")
    try:
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1) 