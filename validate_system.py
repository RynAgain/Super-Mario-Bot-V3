"""
System Validation Script for Super Mario Bros AI Training System

This script performs comprehensive validation of the entire system to ensure
all components are properly installed, configured, and working correctly.
It checks dependencies, configurations, system resources, and component functionality.

Usage:
    python validate_system.py [OPTIONS]

Examples:
    python validate_system.py
    python validate_system.py --verbose
    python validate_system.py --fix-issues
    python validate_system.py --export-report validation_report.json
"""

import argparse
import json
import logging
import os
import platform
import psutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class SystemValidator:
    """Comprehensive system validation for the Mario AI training system."""
    
    def __init__(self, verbose: bool = False, fix_issues: bool = False):
        self.verbose = verbose
        self.fix_issues = fix_issues
        self.validation_results = {}
        self.issues_found = []
        self.fixes_applied = []
        
        # Setup logging
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(levelname)s: %(message)s',
            handlers=[logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)
    
    def log_result(self, test_name: str, passed: bool, message: str = "", details: Dict = None):
        """Log validation result."""
        self.validation_results[test_name] = {
            'passed': passed,
            'message': message,
            'details': details or {}
        }
        
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}: {message}")
        
        if not passed:
            self.issues_found.append({
                'test': test_name,
                'message': message,
                'details': details or {}
            })
    
    def validate_python_environment(self) -> bool:
        """Validate Python environment and version."""
        print("\n" + "="*50)
        print("PYTHON ENVIRONMENT VALIDATION")
        print("="*50)
        
        try:
            # Check Python version
            python_version = sys.version_info
            version_str = f"{python_version.major}.{python_version.minor}.{python_version.micro}"
            
            if python_version >= (3, 8):
                self.log_result("Python Version", True, f"Python {version_str}")
            else:
                self.log_result("Python Version", False, 
                              f"Python {version_str} (requires 3.8+)")
                return False
            
            # Check Python executable
            python_exe = sys.executable
            self.log_result("Python Executable", True, python_exe)
            
            # Check virtual environment
            in_venv = hasattr(sys, 'real_prefix') or (
                hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
            )
            
            if in_venv:
                venv_path = sys.prefix
                self.log_result("Virtual Environment", True, f"Active: {venv_path}")
            else:
                self.log_result("Virtual Environment", True, "Not using virtual environment")
            
            # Check pip availability
            try:
                import pip
                pip_version = pip.__version__
                self.log_result("Pip Availability", True, f"pip {pip_version}")
            except ImportError:
                self.log_result("Pip Availability", False, "pip not available")
                return False
            
            return True
            
        except Exception as e:
            self.log_result("Python Environment", False, f"Validation failed: {e}")
            return False
    
    def validate_dependencies(self) -> bool:
        """Validate all required Python dependencies."""
        print("\n" + "="*50)
        print("DEPENDENCIES VALIDATION")
        print("="*50)
        
        # Required dependencies with minimum versions
        required_deps = {
            'torch': '1.12.0',
            'torchvision': '0.13.0',
            'numpy': '1.21.0',
            'opencv-python': '4.5.0',
            'websockets': '10.0',
            'pyyaml': '5.4.0',
            'psutil': '5.8.0',
            'pandas': '1.3.0'
        }
        
        # Optional dependencies
        optional_deps = {
            'matplotlib': '3.5.0',
            'seaborn': '0.11.0',
            'tensorboard': '2.8.0'
        }
        
        all_passed = True
        
        # Check required dependencies
        for package, min_version in required_deps.items():
            try:
                if package == 'opencv-python':
                    import cv2
                    version = cv2.__version__
                    package_name = 'opencv-python'
                else:
                    module = __import__(package)
                    version = getattr(module, '__version__', 'unknown')
                    package_name = package
                
                self.log_result(f"Required: {package_name}", True, f"v{version}")
                
            except ImportError:
                self.log_result(f"Required: {package_name}", False, "Not installed")
                all_passed = False
                
                if self.fix_issues:
                    self._attempt_package_install(package)
        
        # Check optional dependencies
        for package, min_version in optional_deps.items():
            try:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                self.log_result(f"Optional: {package}", True, f"v{version}")
            except ImportError:
                self.log_result(f"Optional: {package}", True, "Not installed (optional)")
        
        # Check CUDA availability
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                cuda_version = torch.version.cuda
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                self.log_result("CUDA Support", True, 
                              f"CUDA {cuda_version}, {gpu_count} GPU(s), {gpu_name}")
            else:
                self.log_result("CUDA Support", True, "CPU-only mode (CUDA not available)")
        except Exception as e:
            self.log_result("CUDA Support", False, f"Error checking CUDA: {e}")
        
        return all_passed
    
    def validate_system_resources(self) -> bool:
        """Validate system resources and performance."""
        print("\n" + "="*50)
        print("SYSTEM RESOURCES VALIDATION")
        print("="*50)
        
        try:
            # System information
            system_info = {
                'platform': platform.system(),
                'platform_version': platform.version(),
                'architecture': platform.architecture()[0],
                'processor': platform.processor(),
                'python_implementation': platform.python_implementation()
            }
            
            self.log_result("System Platform", True, 
                          f"{system_info['platform']} {system_info['architecture']}")
            
            # Memory check
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            memory_available_gb = memory.available / (1024**3)
            
            if memory_gb >= 8.0:
                self.log_result("System Memory", True, 
                              f"{memory_gb:.1f}GB total, {memory_available_gb:.1f}GB available")
            else:
                self.log_result("System Memory", False, 
                              f"{memory_gb:.1f}GB total (8GB+ recommended)")
            
            # CPU check
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            cpu_usage = psutil.cpu_percent(interval=1)
            
            cpu_info = f"{cpu_count} cores"
            if cpu_freq:
                cpu_info += f", {cpu_freq.current:.0f}MHz"
            cpu_info += f", {cpu_usage:.1f}% usage"
            
            self.log_result("CPU Resources", True, cpu_info)
            
            # Disk space check
            disk_usage = psutil.disk_usage('.')
            disk_free_gb = disk_usage.free / (1024**3)
            
            if disk_free_gb >= 2.0:
                self.log_result("Disk Space", True, f"{disk_free_gb:.1f}GB free")
            else:
                self.log_result("Disk Space", False, 
                              f"{disk_free_gb:.1f}GB free (2GB+ recommended)")
            
            # Network connectivity check
            try:
                import socket
                socket.create_connection(("8.8.8.8", 53), timeout=3)
                self.log_result("Network Connectivity", True, "Internet connection available")
            except OSError:
                self.log_result("Network Connectivity", False, "No internet connection")
            
            return True
            
        except Exception as e:
            self.log_result("System Resources", False, f"Validation failed: {e}")
            return False
    
    def validate_project_structure(self) -> bool:
        """Validate project directory structure and files."""
        print("\n" + "="*50)
        print("PROJECT STRUCTURE VALIDATION")
        print("="*50)
        
        # Required directories
        required_dirs = [
            'python', 'config', 'docs', 'lua', 'examples'
        ]
        
        # Required files
        required_files = [
            'README.md', 'requirements.txt', 'setup.py',
            'python/main.py', 'python/__init__.py',
            'config/training_config.yaml', 'config/network_config.yaml',
            'lua/mario_ai.lua', 'lua/json.lua'
        ]
        
        # Optional but recommended files
        optional_files = [
            'INSTALLATION.md', 'USAGE.md', 'TROUBLESHOOTING.md',
            'install.bat', 'run_training.bat', 'validate_system.py'
        ]
        
        all_passed = True
        
        # Check directories
        for dir_name in required_dirs:
            dir_path = Path(dir_name)
            if dir_path.exists() and dir_path.is_dir():
                file_count = len(list(dir_path.rglob('*')))
                self.log_result(f"Directory: {dir_name}", True, f"{file_count} files")
            else:
                self.log_result(f"Directory: {dir_name}", False, "Missing")
                all_passed = False
        
        # Check required files
        for file_name in required_files:
            file_path = Path(file_name)
            if file_path.exists() and file_path.is_file():
                file_size = file_path.stat().st_size
                self.log_result(f"Required: {file_name}", True, f"{file_size} bytes")
            else:
                self.log_result(f"Required: {file_name}", False, "Missing")
                all_passed = False
        
        # Check optional files
        for file_name in optional_files:
            file_path = Path(file_name)
            if file_path.exists() and file_path.is_file():
                file_size = file_path.stat().st_size
                self.log_result(f"Optional: {file_name}", True, f"{file_size} bytes")
            else:
                self.log_result(f"Optional: {file_name}", True, "Missing (optional)")
        
        return all_passed
    
    def validate_configurations(self) -> bool:
        """Validate configuration files."""
        print("\n" + "="*50)
        print("CONFIGURATION VALIDATION")
        print("="*50)
        
        config_files = [
            'config/training_config.yaml',
            'config/network_config.yaml',
            'config/game_config.yaml',
            'config/logging_config.yaml'
        ]
        
        all_passed = True
        
        for config_file in config_files:
            try:
                # Check if file exists
                config_path = Path(config_file)
                if not config_path.exists():
                    self.log_result(f"Config: {config_file}", False, "File not found")
                    all_passed = False
                    continue
                
                # Try to load and validate YAML
                import yaml
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                if config_data is None:
                    self.log_result(f"Config: {config_file}", False, "Empty or invalid YAML")
                    all_passed = False
                    continue
                
                # Basic structure validation
                if config_file.endswith('training_config.yaml'):
                    required_keys = ['training', 'performance', 'network']
                    missing_keys = [key for key in required_keys if key not in config_data]
                    if missing_keys:
                        self.log_result(f"Config: {config_file}", False, 
                                      f"Missing keys: {missing_keys}")
                        all_passed = False
                        continue
                
                self.log_result(f"Config: {config_file}", True, "Valid YAML structure")
                
            except Exception as e:
                self.log_result(f"Config: {config_file}", False, f"Validation error: {e}")
                all_passed = False
        
        # Test configuration loading with the system
        try:
            from python.utils.config_loader import ConfigLoader
            config_loader = ConfigLoader()
            config = config_loader.load_config('config/training_config.yaml')
            self.log_result("Config Loading", True, "System can load configurations")
        except Exception as e:
            self.log_result("Config Loading", False, f"System config loading failed: {e}")
            all_passed = False
        
        return all_passed
    
    def validate_neural_network_components(self) -> bool:
        """Validate neural network components."""
        print("\n" + "="*50)
        print("NEURAL NETWORK VALIDATION")
        print("="*50)
        
        try:
            # Test model creation
            from python.models.dueling_dqn import DuelingDQN
            import torch
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = DuelingDQN(
                input_channels=4,
                num_actions=12,
                hidden_size=512
            ).to(device)
            
            self.log_result("Model Creation", True, f"DuelingDQN created on {device}")
            
            # Test forward pass
            dummy_input = torch.randn(1, 4, 84, 84).to(device)
            with torch.no_grad():
                output = model(dummy_input)
            
            expected_shape = (1, 12)
            if output.shape == expected_shape:
                self.log_result("Model Forward Pass", True, f"Output shape: {output.shape}")
            else:
                self.log_result("Model Forward Pass", False, 
                              f"Wrong output shape: {output.shape}, expected: {expected_shape}")
                return False
            
            # Test agent creation
            from python.agents.dqn_agent import DQNAgent
            agent = DQNAgent(
                state_dim=(4, 84, 84),
                action_dim=12,
                learning_rate=0.001,
                device=device
            )
            
            self.log_result("Agent Creation", True, "DQNAgent initialized")
            
            # Test action selection
            state = torch.randn(4, 84, 84).to(device)
            action = agent.select_action(state)
            
            if 0 <= action < 12:
                self.log_result("Action Selection", True, f"Selected action: {action}")
            else:
                self.log_result("Action Selection", False, f"Invalid action: {action}")
                return False
            
            return True
            
        except Exception as e:
            self.log_result("Neural Network Components", False, f"Validation failed: {e}")
            if self.verbose:
                self.logger.error(traceback.format_exc())
            return False
    
    def _attempt_package_install(self, package: str):
        """Attempt to install missing package."""
        if not self.fix_issues:
            return
        
        try:
            print(f"  Attempting to install {package}...")
            result = subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                                  capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                self.fixes_applied.append(f"Installed {package}")
                print(f"  ✓ Successfully installed {package}")
            else:
                print(f"  ❌ Failed to install {package}: {result.stderr}")
                
        except Exception as e:
            print(f"  ❌ Error installing {package}: {e}")
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        total_tests = len(self.validation_results)
        passed_tests = sum(1 for result in self.validation_results.values() if result['passed'])
        failed_tests = total_tests - passed_tests
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'system_info': {
                'platform': platform.system(),
                'platform_version': platform.version(),
                'architecture': platform.architecture()[0],
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'python_executable': sys.executable
            },
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0
            },
            'test_results': self.validation_results,
            'issues_found': self.issues_found,
            'fixes_applied': self.fixes_applied
        }
        
        return report
    
    def print_validation_summary(self):
        """Print validation summary."""
        total_tests = len(self.validation_results)
        passed_tests = sum(1 for result in self.validation_results.values() if result['passed'])
        failed_tests = total_tests - passed_tests
        
        print("\n" + "="*60)
        print("SYSTEM VALIDATION SUMMARY")
        print("="*60)
        
        print(f"\nValidation Results:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {passed_tests}")
        print(f"  Failed: {failed_tests}")
        print(f"  Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests == 0:
            print(f"\n🎉 ALL TESTS PASSED!")
            print("The Super Mario Bros AI Training System is ready for use!")
        else:
            print(f"\n⚠️  {failed_tests} test(s) failed.")
            print("Please review the issues above and follow the recommendations.")
        
        if self.fixes_applied:
            print(f"\nFixes Applied:")
            for fix in self.fixes_applied:
                print(f"  ✓ {fix}")
        
        print("\nNext Steps:")
        if failed_tests == 0:
            print("  1. Start FCEUX with Super Mario Bros ROM")
            print("  2. Load lua/mario_ai.lua script in FCEUX")
            print("  3. Run training: python python/main.py train")
            print("  4. Monitor progress in logs/ directory")
        else:
            print("  1. Fix the issues identified above")
            print("  2. Re-run validation: python validate_system.py")
            print("  3. Check TROUBLESHOOTING.md for common solutions")
        
        print("="*60)
    
    def run_full_validation(self) -> bool:
        """Run complete system validation."""
        print("🎮 Super Mario Bros AI Training System - System Validation")
        print("="*60)
        print(f"Validation started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        validation_steps = [
            ("Python Environment", self.validate_python_environment),
            ("Dependencies", self.validate_dependencies),
            ("System Resources", self.validate_system_resources),
            ("Project Structure", self.validate_project_structure),
            ("Configurations", self.validate_configurations),
            ("Neural Network Components", self.validate_neural_network_components)
        ]
        
        all_passed = True
        
        for step_name, validation_func in validation_steps:
            try:
                step_passed = validation_func()
                if not step_passed:
                    all_passed = False
            except Exception as e:
                self.log_result(step_name, False, f"Validation error: {e}")
                if self.verbose:
                    self.logger.error(traceback.format_exc())
                all_passed = False
        
        self.print_validation_summary()
        return all_passed


def main():
    """Main function for system validation."""
    parser = argparse.ArgumentParser(description="Validate Super Mario Bros AI Training System")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--fix-issues", action="store_true", help="Attempt to fix issues automatically")
    parser.add_argument("--export-report", type=str, help="Export validation report to JSON file")
    parser.add_argument("--quick", action="store_true", help="Run quick validation (skip benchmarks)")
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = SystemValidator(verbose=args.verbose, fix_issues=args.fix_issues)
    
    # Run validation
    all_passed = validator.run_full_validation()
    
    # Export report if requested
    if args.export_report:
        try:
            report = validator.generate_validation_report()
            with open(args.export_report, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n📄 Validation report exported to: {args.export_report}")
        except Exception as e:
            print(f"❌ Failed to export report: {e}")
    
    # Exit with appropriate code
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)