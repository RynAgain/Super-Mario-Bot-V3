"""
Setup script for Super Mario Bros AI Training System

This script handles the installation of the Super Mario Bros AI training system
as a Python package, including all dependencies and optional components.
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages

# Read the README file for long description
def read_readme():
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Super Mario Bros AI Training System"

# Read requirements from requirements.txt
def read_requirements():
    requirements_path = Path(__file__).parent / "requirements.txt"
    if requirements_path.exists():
        with open(requirements_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []

# Get version from __init__.py or default
def get_version():
    init_path = Path(__file__).parent / "python" / "__init__.py"
    if init_path.exists():
        with open(init_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"').strip("'")
    return "1.0.0"

# Check Python version
if sys.version_info < (3, 8):
    print("Error: Super Mario Bros AI Training System requires Python 3.8 or higher.")
    print(f"You are using Python {sys.version}")
    sys.exit(1)

# Define package data
package_data = {
    "": [
        "*.yaml", "*.yml", "*.json", "*.txt", "*.md",
        "*.lua", "*.bat", "*.sh"
    ]
}

# Define data files
data_files = [
    ("config", [
        "config/training_config.yaml",
        "config/network_config.yaml", 
        "config/game_config.yaml",
        "config/logging_config.yaml"
    ]),
    ("lua", [
        "lua/mario_ai.lua",
        "lua/json.lua",
        "lua/README.md"
    ]),
    ("docs", [
        "docs/architecture.md",
        "docs/communication-protocol.md",
        "docs/configuration-files.md",
        "docs/csv-logging-format.md",
        "docs/data-flow.md",
        "docs/frame-synchronization.md",
        "docs/memory-addresses.md",
        "docs/neural-network-architecture.md",
        "docs/project-structure.md",
        "docs/project-structure-implementation.md",
        "docs/reward-system.md"
    ]),
    ("", [
        "requirements.txt",
        "README.md"
    ])
]

# Define entry points for command-line scripts
entry_points = {
    "console_scripts": [
        "mario-ai-train=python.main:main",
        "mario-ai-test=test_complete_system_integration:main",
        "mario-ai-validate=validate_system:main",
    ]
}

# Define optional dependencies
extras_require = {
    "dev": [
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0",
        "black>=22.0.0",
        "flake8>=4.0.0",
        "mypy>=0.991",
    ],
    "plotting": [
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
    ],
    "gpu": [
        "torch>=1.12.0+cu116",
        "torchvision>=0.13.0+cu116",
    ],
    "all": [
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0",
        "black>=22.0.0",
        "flake8>=4.0.0",
        "mypy>=0.991",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
    ]
}

# Setup configuration
setup(
    name="super-mario-ai",
    version=get_version(),
    author="Super Mario AI Development Team",
    author_email="dev@super-mario-ai.com",
    description="AI Training System for Super Mario Bros using Deep Q-Learning",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/Super-Mario-Bot-V3",
    project_urls={
        "Bug Reports": "https://github.com/your-username/Super-Mario-Bot-V3/issues",
        "Source": "https://github.com/your-username/Super-Mario-Bot-V3",
        "Documentation": "https://github.com/your-username/Super-Mario-Bot-V3/docs",
    },
    
    # Package configuration
    packages=find_packages(include=["python", "python.*"]),
    package_data=package_data,
    data_files=data_files,
    include_package_data=True,
    
    # Dependencies
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require=extras_require,
    
    # Entry points
    entry_points=entry_points,
    
    # Classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Games/Entertainment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    
    # Keywords
    keywords="ai machine-learning deep-learning reinforcement-learning gaming super-mario dqn pytorch",
    
    # Additional metadata
    zip_safe=False,
    platforms=["any"],
    
    # Custom commands
    cmdclass={},
)

# Post-installation message
def print_post_install_message():
    print("\n" + "="*60)
    print("🎮 Super Mario Bros AI Training System Installation Complete!")
    print("="*60)
    print("\nNext Steps:")
    print("1. Verify installation: mario-ai-validate")
    print("2. Run system tests: mario-ai-test")
    print("3. Start training: mario-ai-train")
    print("\nFor detailed instructions, see:")
    print("- README.md for overview")
    print("- INSTALLATION.md for setup guide")
    print("- USAGE.md for usage examples")
    print("\nHappy training! 🚀")
    print("="*60)

if __name__ == "__main__":
    # Run setup
    setup()
    
    # Print post-installation message if this was an install
    if "install" in sys.argv:
        print_post_install_message()