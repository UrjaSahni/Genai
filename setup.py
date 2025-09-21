#!/usr/bin/env python3
"""
Setup script for the Research Paper Summarizer & Comparator
This script helps with initial project setup and configuration.
"""

import os
import sys
from pathlib import Path

def create_env_file():
    """Create a .env file template if it doesn't exist"""
    env_path = Path('.env')
    
    if env_path.exists():
        print("✅ .env file already exists")
        return True
    
    env_template = """# OpenRouter API Configuration
# Get your API key from: https://openrouter.ai/keys
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Optional: Site information for OpenRouter rankings
SITE_URL=https://github.com/your-username/research-paper-summarizer
SITE_NAME=Research Paper Summarizer

# Application Settings
DEBUG=false
"""
    
    try:
        with open(env_path, 'w') as f:
            f.write(env_template)
        print("✅ Created .env file template")
        print("📝 Please edit .env and add your OpenRouter API key")
        return True
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        'exports',
        'temp',
        'logs'
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"✅ Created directory: {directory}")
            except Exception as e:
                print(f"❌ Error creating directory {directory}: {e}")
                return False
        else:
            print(f"✅ Directory exists: {directory}")
    
    return True

def install_requirements():
    """Install required packages"""
    import subprocess
    
    requirements_file = Path('requirements.txt')
    if not requirements_file.exists():
        print("❌ requirements.txt not found!")
        return False
    
    try:
        print("📦 Installing required packages...")
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ All packages installed successfully!")
            return True
        else:
            print(f"❌ Error installing packages: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error during installation: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    required_version = (3, 7)
    current_version = sys.version_info[:2]
    
    if current_version < required_version:
        print(f"❌ Python {required_version[0]}.{required_version[1]}+ is required!")
        print(f"   Current version: {current_version[0]}.{current_version[1]}")
        return False
    
    print(f"✅ Python version {current_version[0]}.{current_version[1]} is compatible")
    return True

def display_next_steps():
    """Display setup completion message and next steps"""
    print("\n" + "=" * 60)
    print("🎉 Setup completed successfully!")
    print("=" * 60)
    
    print("\n📋 Next steps:")
    print("1. Edit the .env file and add your OpenRouter API key:")
    print("   - Get your key from: https://openrouter.ai/keys")
    print("   - Replace 'your_openrouter_api_key_here' with your actual key")
    
    print("\n2. Test your setup:")
    print("   python run.py")
    print("   or")
    print("   streamlit run streamlit_app.py")
    
    print("\n3. Start using the application:")
    print("   - Upload research papers (PDF, TXT, MD)")
    print("   - Choose summary type (comprehensive, key_insights, methodology)")
    print("   - Generate summaries and comparisons")
    print("   - Export your results")
    
    print("\n📚 For more information:")
    print("   - Read the README.md file")
    print("   - Check the troubleshooting section for common issues")
    
    print("\n🚀 Happy researching!")

def main():
    """Main setup function"""
    print("🔬 Research Paper Summarizer & Comparator - Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create necessary files and directories
    success = True
    
    print("\n📁 Creating project structure...")
    success &= create_directories()
    
    print("\n📄 Setting up configuration...")
    success &= create_env_file()
    
    # Ask user if they want to install requirements
    print("\n📦 Package installation:")
    install_packages = input("Install required packages now? (y/n): ").lower().strip()
    
    if install_packages in ['y', 'yes', '1', 'true']:
        success &= install_requirements()
    else:
        print("⏭️  Skipping package installation")
        print("   Run 'pip install -r requirements.txt' later")
    
    if success:
        display_next_steps()
    else:
        print("\n❌ Setup completed with some errors")
        print("   Please check the messages above and fix any issues")
        sys.exit(1)

if __name__ == "__main__":
    main()
