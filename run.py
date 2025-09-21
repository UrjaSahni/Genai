#!/usr/bin/env python3
"""
Quick run script for the Research Paper Summarizer & Comparator
This script provides an easy way to start the application with proper setup checks.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if all required packages are installed"""
    required_packages = [
        'streamlit',
        'openai',
        'python-dotenv',
        'PyPDF2',
        'fpdf2'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_').lower())
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed!")
    return True

def check_env_file():
    """Check if .env file exists and has required variables"""
    env_path = Path('.env')
    
    if not env_path.exists():
        print("❌ .env file not found!")
        print("\n📝 Create a .env file with your OpenRouter API key:")
        print("OPENROUTER_API_KEY=your_api_key_here")
        return False
    
    # Read and check env file
    with open(env_path, 'r') as f:
        env_content = f.read()
    
    if 'OPENROUTER_API_KEY' not in env_content:
        print("❌ OPENROUTER_API_KEY not found in .env file!")
        return False
    
    # Check if the key looks valid
    for line in env_content.split('\n'):
        if line.startswith('OPENROUTER_API_KEY='):
            api_key = line.split('=', 1)[1].strip()
            if not api_key or api_key == 'your_api_key_here':
                print("❌ Please set a valid OpenRouter API key in .env file!")
                return False
            if not api_key.startswith('sk-or-'):
                print("⚠️  Warning: API key doesn't start with 'sk-or-'")
                print("   Make sure you're using a valid OpenRouter API key")
    
    print("✅ Environment configuration looks good!")
    return True

def main():
    """Main function to run the application"""
    print("🔬 AI Research Paper Summarizer & Comparator")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required!")
        print(f"   Current version: {sys.version}")
        sys.exit(1)
    
    print(f"✅ Python version: {sys.version.split()[0]}")
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check environment file
    if not check_env_file():
        sys.exit(1)
    
    # Check if streamlit_app.py exists
    if not Path('streamlit_app.py').exists():
        print("❌ streamlit_app.py not found!")
        print("   Make sure you're running this script from the project directory")
        sys.exit(1)
    
    print("\n🚀 Starting the application...")
    print("📱 The app will open in your default web browser")
    print("🔗 Default URL: http://localhost:8501")
    print("\n💡 Tips:")
    print("   - Press Ctrl+C to stop the application")
    print("   - Refresh the page if you encounter any issues")
    print("   - Check the terminal for any error messages")
    print("\n" + "=" * 50)
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py',
            '--server.headless', 'false',
            '--server.enableXsrfProtection', 'false',
            '--server.enableCORS', 'false'
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error running application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
