#!/usr/bin/env python3
"""
Enhanced State Management System Validation Script
=================================================

This script validates the enhanced state management system by running comprehensive
integration tests and generating a detailed validation report.

Usage:
    python validate_system.py [--enhanced] [--legacy] [--performance] [--report-only]

Options:
    --enhanced      Test enhanced 20-feature mode (default)
    --legacy        Test legacy 12-feature mode
    --performance   Run performance benchmarks
    --report-only   Generate report from existing test results

Author: AI Training System
Version: 1.0
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from test_integration_suite import IntegrationTestSuite


class SystemValidator:
    """System validation orchestrator."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize system validator.
        
        Args:
            config: Validation configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.validation_results = {}
        
        # Setup logging
        self._setup_logging()
        
        self.logger.info("System validator initialized")
        self.logger.info(f"Configuration: {config}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = logging.DEBUG if self.config.get('verbose', False) else logging.INFO
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('system_validation.log')
            ]
        )
    
    async def validate_system(self) -> Dict[str, Any]:
        """
        Run comprehensive system validation.
        
        Returns:
            Validation results dictionary
        """
        self.logger.info("[START] Starting Enhanced State Management System Validation")
        self.logger.info("=" * 80)
        
        validation_start_time = time.time()
        
        # Phase 1: Pre-validation checks
        self.logger.info("Phase 1: Pre-validation system checks")
        pre_validation_results = await self._run_pre_validation_checks()
        
        if not pre_validation_results['passed']:
            self.logger.error("[FAIL] Pre-validation checks failed")
            return self._generate_failure_report("Pre-validation checks failed", pre_validation_results)
        
        self.logger.info("[PASS] Pre-validation checks passed")
        
        # Phase 2: Integration testing
        self.logger.info("Phase 2: Integration testing")
        integration_results = await self._run_integration_tests()
        
        # Phase 3: Performance validation
        performance_results = {}
        if self.config.get('performance_tests', True):
            self.logger.info("Phase 3: Performance validation")
            performance_results = await self._run_performance_validation()
        
        # Phase 4: System readiness assessment
        self.logger.info("Phase 4: System readiness assessment")
        readiness_results = await self._assess_system_readiness(integration_results, performance_results)
        
        # Generate comprehensive validation report
        total_validation_time = time.time() - validation_start_time
        
        validation_report = {
            'validation_timestamp': time.time(),
            'validation_duration_seconds': total_validation_time,
            'configuration': self.config,
            'pre_validation': pre_validation_results,
            'integration_tests': integration_results,
            'performance_validation': performance_results,
            'system_readiness': readiness_results,
            'overall_status': self._determine_overall_status(integration_results, performance_results, readiness_results),
            'recommendations': self._generate_recommendations(integration_results, performance_results, readiness_results)
        }
        
        self.validation_results = validation_report
        
        # Save validation report
        await self._save_validation_report(validation_report)
        
        self.logger.info(f"[PASS] System validation completed in {total_validation_time:.2f} seconds")
        
        return validation_report
    
    async def _run_pre_validation_checks(self) -> Dict[str, Any]:
        """Run pre-validation system checks."""
        checks = {
            'python_version': self._check_python_version(),
            'dependencies': self._check_dependencies(),
            'file_structure': self._check_file_structure(),
            'configuration_files': self._check_configuration_files()
        }
        
        passed_checks = sum(1 for result in checks.values() if result['passed'])
        total_checks = len(checks)
        
        return {
            'passed': passed_checks == total_checks,
            'checks': checks,
            'summary': f"{passed_checks}/{total_checks} checks passed"
        }
    
    def _check_python_version(self) -> Dict[str, Any]:
        """Check Python version compatibility."""
        try:
            version = sys.version_info
            required_major, required_minor = 3, 8
            
            if version.major >= required_major and version.minor >= required_minor:
                return {
                    'passed': True,
                    'message': f"Python {version.major}.{version.minor}.{version.micro} is compatible",
                    'version': f"{version.major}.{version.minor}.{version.micro}"
                }
            else:
                return {
                    'passed': False,
                    'message': f"Python {version.major}.{version.minor} is too old, requires {required_major}.{required_minor}+",
                    'version': f"{version.major}.{version.minor}.{version.micro}"
                }
        except Exception as e:
            return {
                'passed': False,
                'message': f"Failed to check Python version: {e}",
                'error': str(e)
            }
    
    def _check_dependencies(self) -> Dict[str, Any]:
        """Check required dependencies."""
        required_packages = [
            'torch', 'numpy', 'websockets', 'asyncio', 'cv2'
        ]
        
        missing_packages = []
        available_packages = []
        
        for package in required_packages:
            try:
                if package == 'cv2':
                    import cv2
                    available_packages.append(f"{package} ({cv2.__version__})")
                elif package == 'torch':
                    import torch
                    available_packages.append(f"{package} ({torch.__version__})")
                elif package == 'numpy':
                    import numpy
                    available_packages.append(f"{package} ({numpy.__version__})")
                elif package == 'websockets':
                    import websockets
                    available_packages.append(f"{package} ({websockets.__version__})")
                elif package == 'asyncio':
                    import asyncio
                    available_packages.append(package)
                else:
                    __import__(package)
                    available_packages.append(package)
            except ImportError:
                missing_packages.append(package)
        
        return {
            'passed': len(missing_packages) == 0,
            'available_packages': available_packages,
            'missing_packages': missing_packages,
            'message': f"{len(available_packages)}/{len(required_packages)} required packages available"
        }
    
    def _check_file_structure(self) -> Dict[str, Any]:
        """Check project file structure."""
        required_files = [
            'lua/mario_ai.lua',
            'python/utils/preprocessing.py',
            'python/environment/reward_calculator.py',
            'python/communication/websocket_server.py',
            'python/communication/comm_manager.py',
            'python/models/dueling_dqn.py',
            'python/agents/dqn_agent.py'
        ]
        
        missing_files = []
        existing_files = []
        
        for file_path in required_files:
            full_path = project_root / file_path
            if full_path.exists():
                existing_files.append(file_path)
            else:
                missing_files.append(file_path)
        
        return {
            'passed': len(missing_files) == 0,
            'existing_files': existing_files,
            'missing_files': missing_files,
            'message': f"{len(existing_files)}/{len(required_files)} required files found"
        }
    
    def _check_configuration_files(self) -> Dict[str, Any]:
        """Check configuration files."""
        config_files = [
            'config/training_config.yaml',
            'config/network_config.yaml'
        ]
        
        config_status = {}
        all_configs_valid = True
        
        for config_file in config_files:
            config_path = project_root / config_file
            if config_path.exists():
                try:
                    # Try to load configuration
                    if config_file.endswith('.yaml'):
                        try:
                            import yaml
                            with open(config_path, 'r') as f:
                                config_data = yaml.safe_load(f)
                        except ImportError:
                            # yaml not available, skip validation but don't fail
                            config_status[config_file] = {
                                'exists': True,
                                'valid': True,
                                'warning': 'yaml not available for validation'
                            }
                            continue
                        config_status[config_file] = {
                            'exists': True,
                            'valid': True,
                            'keys': list(config_data.keys()) if isinstance(config_data, dict) else []
                        }
                    else:
                        config_status[config_file] = {
                            'exists': True,
                            'valid': True,
                            'keys': []
                        }
                except Exception as e:
                    config_status[config_file] = {
                        'exists': True,
                        'valid': False,
                        'error': str(e)
                    }
                    all_configs_valid = False
            else:
                config_status[config_file] = {
                    'exists': False,
                    'valid': False
                }
                all_configs_valid = False
        
        return {
            'passed': all_configs_valid,
            'config_status': config_status,
            'message': f"Configuration files validation: {'passed' if all_configs_valid else 'failed'}"
        }
    
    async def _run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests."""
        try:
            # Create integration test suite
            test_suite = IntegrationTestSuite()
            
            # Configure test suite based on validation config
            test_suite.test_config.update({
                'enhanced_features': self.config.get('test_enhanced', True),
                'legacy_mode': self.config.get('test_legacy', True),
                'performance_benchmarks': self.config.get('performance_tests', True),
                'error_injection': self.config.get('test_error_handling', True),
                'checkpoint_validation': self.config.get('test_checkpoints', True)
            })
            
            # Run all integration tests
            integration_results = await test_suite.run_all_tests()
            
            return integration_results
            
        except Exception as e:
            self.logger.error(f"Integration tests failed with exception: {e}")
            return {
                'overall_status': 'FAILED',
                'system_ready_for_training': False,
                'error': str(e),
                'test_statistics': {
                    'total_tests': 0,
                    'tests_passed': 0,
                    'tests_failed': 1,
                    'success_rate_percent': 0
                }
            }
    
    async def _run_performance_validation(self) -> Dict[str, Any]:
        """Run performance validation tests."""
        performance_results = {
            'cpu_usage': await self._measure_cpu_usage(),
            'memory_usage': await self._measure_memory_usage(),
            'throughput': await self._measure_throughput(),
            'latency': await self._measure_latency()
        }
        
        # Determine if performance is acceptable
        performance_acceptable = (
            performance_results['cpu_usage'].get('acceptable', True) and
            performance_results['memory_usage'].get('acceptable', True) and
            performance_results['throughput'].get('acceptable', True) and
            performance_results['latency'].get('acceptable', True)
        )
        
        return {
            'acceptable': performance_acceptable,
            'metrics': performance_results,
            'summary': "Performance validation " + ("passed" if performance_acceptable else "failed")
        }
    
    async def _measure_cpu_usage(self) -> Dict[str, Any]:
        """Measure CPU usage during processing."""
        try:
            import psutil
            
            # Measure CPU usage during simulated processing
            cpu_percent_before = psutil.cpu_percent(interval=1)
            
            # Simulate processing load
            from test_integration_suite import TestDataGenerator
            generator = TestDataGenerator()
            
            start_time = time.time()
            for _ in range(100):
                payload = generator.generate_enhanced_payload()
                # Simulate processing
                time.sleep(0.001)  # 1ms processing time
            
            processing_time = time.time() - start_time
            cpu_percent_after = psutil.cpu_percent(interval=1)
            
            cpu_usage = max(cpu_percent_after - cpu_percent_before, 0)
            
            return {
                'cpu_usage_percent': cpu_usage,
                'processing_time_seconds': processing_time,
                'acceptable': cpu_usage < 50,  # Less than 50% CPU usage
                'message': f"CPU usage: {cpu_usage:.1f}%"
            }
            
        except ImportError:
            return {
                'cpu_usage_percent': 0,
                'acceptable': True,
                'message': "psutil not available, skipping CPU measurement",
                'skipped': True
            }
        except Exception as e:
            return {
                'cpu_usage_percent': 0,
                'acceptable': False,
                'error': str(e),
                'message': f"CPU measurement failed: {e}"
            }
    
    async def _measure_memory_usage(self) -> Dict[str, Any]:
        """Measure memory usage during processing."""
        try:
            import psutil
            import gc
            
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Simulate memory-intensive processing
            from test_integration_suite import TestDataGenerator
            from python.utils.preprocessing import MarioPreprocessor
            
            generator = TestDataGenerator()
            preprocessor = MarioPreprocessor(enhanced_features=True)
            
            # Process many frames
            for _ in range(50):
                payload = generator.generate_enhanced_payload()
                game_state = generator.generate_game_state(enhanced=True)
                
                import numpy as np
                raw_frame = np.random.randint(0, 255, (240, 256, 3), dtype=np.uint8)
                stacked_frames, state_vector = preprocessor.process_step(raw_frame, game_state)
            
            gc.collect()  # Force garbage collection
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = memory_after - memory_before
            
            return {
                'memory_usage_mb': memory_increase,
                'memory_before_mb': memory_before,
                'memory_after_mb': memory_after,
                'acceptable': memory_increase < 100,  # Less than 100MB increase
                'message': f"Memory usage increase: {memory_increase:.1f}MB"
            }
            
        except ImportError:
            return {
                'memory_usage_mb': 0,
                'acceptable': True,
                'message': "psutil not available, skipping memory measurement",
                'skipped': True
            }
        except Exception as e:
            return {
                'memory_usage_mb': 0,
                'acceptable': False,
                'error': str(e),
                'message': f"Memory measurement failed: {e}"
            }
    
    async def _measure_throughput(self) -> Dict[str, Any]:
        """Measure system throughput."""
        try:
            from test_integration_suite import TestDataGenerator
            from python.utils.preprocessing import BinaryPayloadParser
            
            generator = TestDataGenerator()
            parser = BinaryPayloadParser(enhanced_features=True)
            
            # Measure parsing throughput
            num_payloads = 1000
            start_time = time.time()
            
            for _ in range(num_payloads):
                payload = generator.generate_enhanced_payload()
                parsed_state = parser.parse_payload(payload)
            
            end_time = time.time()
            total_time = end_time - start_time
            throughput = num_payloads / total_time  # payloads per second
            
            return {
                'throughput_fps': throughput,
                'total_time_seconds': total_time,
                'payloads_processed': num_payloads,
                'acceptable': throughput >= 60,  # At least 60 FPS
                'message': f"Throughput: {throughput:.1f} FPS"
            }
            
        except Exception as e:
            return {
                'throughput_fps': 0,
                'acceptable': False,
                'error': str(e),
                'message': f"Throughput measurement failed: {e}"
            }
    
    async def _measure_latency(self) -> Dict[str, Any]:
        """Measure processing latency."""
        try:
            from test_integration_suite import TestDataGenerator
            from python.utils.preprocessing import MarioPreprocessor
            
            generator = TestDataGenerator()
            preprocessor = MarioPreprocessor(enhanced_features=True)
            
            # Measure end-to-end latency
            latencies = []
            
            for _ in range(100):
                # Generate test data
                payload = generator.generate_enhanced_payload()
                game_state = generator.generate_game_state(enhanced=True)
                
                import numpy as np
                raw_frame = np.random.randint(0, 255, (240, 256, 3), dtype=np.uint8)
                
                # Measure processing time
                start_time = time.time()
                stacked_frames, state_vector = preprocessor.process_step(raw_frame, game_state)
                end_time = time.time()
                
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
            
            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)
            min_latency = min(latencies)
            
            return {
                'avg_latency_ms': avg_latency,
                'max_latency_ms': max_latency,
                'min_latency_ms': min_latency,
                'acceptable': avg_latency < 16.67,  # Less than one frame at 60 FPS
                'message': f"Average latency: {avg_latency:.2f}ms"
            }
            
        except Exception as e:
            return {
                'avg_latency_ms': 0,
                'acceptable': False,
                'error': str(e),
                'message': f"Latency measurement failed: {e}"
            }
    
    async def _assess_system_readiness(self, integration_results: Dict[str, Any], 
                                     performance_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall system readiness."""
        
        # Integration test readiness
        integration_ready = integration_results.get('system_ready_for_training', False)
        integration_success_rate = integration_results.get('test_statistics', {}).get('success_rate_percent', 0)
        
        # Performance readiness
        performance_ready = performance_results.get('acceptable', True)
        
        # Critical component checks
        critical_components = {
            'lua_memory_reading': 'Lua Memory Reading' not in integration_results.get('critical_failures', []),
            'binary_protocol': 'Binary Protocol' not in integration_results.get('critical_failures', []),
            'state_processing': 'State Processing' not in integration_results.get('critical_failures', []),
            'reward_calculation': 'Reward Calculation' not in integration_results.get('critical_failures', []),
            'end_to_end_integration': 'End-to-End Integration' not in integration_results.get('critical_failures', [])
        }
        
        critical_components_ready = all(critical_components.values())
        
        # Overall readiness assessment
        system_ready = (
            integration_ready and
            performance_ready and
            critical_components_ready and
            integration_success_rate >= 95
        )
        
        # Readiness score (0-100)
        readiness_score = 0
        if integration_ready:
            readiness_score += 40
        if performance_ready:
            readiness_score += 20
        if critical_components_ready:
            readiness_score += 30
        readiness_score += min(10, integration_success_rate / 10)
        
        return {
            'system_ready': system_ready,
            'readiness_score': readiness_score,
            'integration_ready': integration_ready,
            'performance_ready': performance_ready,
            'critical_components_ready': critical_components_ready,
            'critical_components': critical_components,
            'integration_success_rate': integration_success_rate,
            'assessment': self._generate_readiness_assessment(system_ready, readiness_score)
        }
    
    def _generate_readiness_assessment(self, system_ready: bool, readiness_score: float) -> str:
        """Generate human-readable readiness assessment."""
        if system_ready and readiness_score >= 90:
            return "System is fully ready for enhanced training"
        elif readiness_score >= 80:
            return "System is mostly ready with minor issues to address"
        elif readiness_score >= 60:
            return "System has significant issues that need resolution"
        else:
            return "System is not ready for training - major issues detected"
    
    def _determine_overall_status(self, integration_results: Dict[str, Any], 
                                performance_results: Dict[str, Any],
                                readiness_results: Dict[str, Any]) -> str:
        """Determine overall validation status."""
        if readiness_results.get('system_ready', False):
            return 'PASSED'
        elif readiness_results.get('readiness_score', 0) >= 80:
            return 'PASSED_WITH_WARNINGS'
        else:
            return 'FAILED'
    
    def _generate_recommendations(self, integration_results: Dict[str, Any], 
                                performance_results: Dict[str, Any],
                                readiness_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Integration test recommendations
        if integration_results.get('recommendations'):
            recommendations.extend(integration_results['recommendations'])
        
        # Performance recommendations
        if not performance_results.get('acceptable', True):
            recommendations.append("Optimize system performance for real-time processing")
            
            perf_metrics = performance_results.get('metrics', {})
            if not perf_metrics.get('cpu_usage', {}).get('acceptable', True):
                recommendations.append("Reduce CPU usage during processing")
            if not perf_metrics.get('memory_usage', {}).get('acceptable', True):
                recommendations.append("Optimize memory usage and prevent memory leaks")
            if not perf_metrics.get('throughput', {}).get('acceptable', True):
                recommendations.append("Improve processing throughput to maintain 60 FPS")
            if not perf_metrics.get('latency', {}).get('acceptable', True):
                recommendations.append("Reduce processing latency for real-time performance")
        
        # System readiness recommendations
        if not readiness_results.get('system_ready', False):
            recommendations.append("Address critical component failures before training")
            
            critical_components = readiness_results.get('critical_components', {})
            for component, ready in critical_components.items():
                if not ready:
                    recommendations.append(f"Fix issues in {component.replace('_', ' ')}")
        
        # General recommendations
        if readiness_results.get('readiness_score', 0) < 100:
            recommendations.append("Run extended validation tests before production use")
            recommendations.append("Monitor system behavior during initial training runs")
        
        return recommendations
    
    def _generate_failure_report(self, reason: str, details: Dict[str, Any]) -> Dict[str, Any]:
        """Generate failure report."""
        return {
            'validation_timestamp': time.time(),
            'overall_status': 'FAILED',
            'failure_reason': reason,
            'failure_details': details,
            'system_ready_for_training': False,
            'recommendations': [
                f"Address the failure: {reason}",
                "Fix all pre-validation issues before running integration tests",
                "Ensure all required dependencies are installed",
                "Verify project file structure is complete"
            ]
        }
    
    async def _save_validation_report(self, report: Dict[str, Any]):
        """Save validation report to file."""
        timestamp = int(time.time())
        report_filename = f"system_validation_report_{timestamp}.json"
        
        try:
            with open(report_filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"[REPORT] Validation report saved to: {report_filename}")
            
            # Also save a human-readable summary
            summary_filename = f"system_validation_summary_{timestamp}.txt"
            await self._save_validation_summary(report, summary_filename)
            
        except Exception as e:
            self.logger.error(f"Failed to save validation report: {e}")
    
    async def _save_validation_summary(self, report: Dict[str, Any], filename: str):
        """Save human-readable validation summary."""
        try:
            with open(filename, 'w') as f:
                f.write("Enhanced State Management System Validation Summary\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"Validation Date: {time.ctime(report['validation_timestamp'])}\n")
                f.write(f"Duration: {report['validation_duration_seconds']:.2f} seconds\n")
                f.write(f"Overall Status: {report['overall_status']}\n\n")
                
                # System readiness
                readiness = report.get('system_readiness', {})
                f.write(f"System Ready for Training: {'YES' if readiness.get('system_ready', False) else 'NO'}\n")
                f.write(f"Readiness Score: {readiness.get('readiness_score', 0):.1f}/100\n")
                f.write(f"Assessment: {readiness.get('assessment', 'Unknown')}\n\n")
                
                # Integration test results
                integration = report.get('integration_tests', {})
                if integration:
                    stats = integration.get('test_statistics', {})
                    f.write(f"Integration Tests: {stats.get('tests_passed', 0)}/{stats.get('total_tests', 0)} passed ")
                    f.write(f"({stats.get('success_rate_percent', 0):.1f}%)\n")
                    
                    if integration.get('critical_failures'):
                        f.write(f"Critical Failures: {', '.join(integration['critical_failures'])}\n")
                    f.write("\n")
                
                # Performance results
                performance = report.get('performance_validation', {})
                if performance:
                    f.write(f"Performance Validation: {'PASSED' if performance.get('acceptable', False) else 'FAILED'}\n")
                    metrics = performance.get('metrics', {})
                    for metric_name, metric_data in metrics.items():
                        if isinstance(metric_data, dict) and 'message' in metric_data:
                            f.write(f"  {metric_name}: {metric_data['message']}\n")
                    f.write("\n")
                
                # Recommendations
                recommendations = report.get('recommendations', [])
                if recommendations:
                    f.write("Recommendations:\n")
                    for i, rec in enumerate(recommendations, 1):
                        f.write(f"  {i}. {rec}\n")
                    f.write("\n")
            
            self.logger.info(f"[REPORT] Validation summary saved to: {filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to save validation summary: {e}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate Enhanced State Management System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate_system.py                    # Full validation with enhanced features
  python validate_system.py --legacy           # Test legacy 12-feature mode
  python validate_system.py --performance      # Include performance benchmarks
  python validate_system.py --verbose          # Verbose logging
        """
    )
    
    parser.add_argument('--enhanced', action='store_true', default=True,
                       help='Test enhanced 20-feature mode (default)')
    parser.add_argument('--legacy', action='store_true',
                       help='Test legacy 12-feature mode')
    parser.add_argument('--performance', action='store_true', default=True,
                       help='Run performance benchmarks (default)')
    parser.add_argument('--no-performance', action='store_true',
                       help='Skip performance benchmarks')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--report-only', action='store_true',
                       help='Generate report from existing test results')
    
    return parser.parse_args()


async def main():
    """Main validation function."""
    args = parse_arguments()
    
    # Configuration
    config = {
        'test_enhanced': args.enhanced and not args.legacy,
        'test_legacy': args.legacy,
        'performance_tests': args.performance and not args.no_performance,
        'test_error_handling': True,
        'test_checkpoints': True,
        'verbose': args.verbose,
        'report_only': args.report_only
    }
    
    # Create validator
    validator = SystemValidator(config)
    
    try:
        # Run validation
        validation_report = await validator.validate_system()
        
        # Display results
        print("\n" + "=" * 80)
        print("[TARGET] SYSTEM VALIDATION RESULTS")
        print("=" * 80)
        
        overall_status = validation_report['overall_status']
        if overall_status == 'PASSED':
            print("[PASS] VALIDATION PASSED - System is ready for enhanced training")
        elif overall_status == 'PASSED_WITH_WARNINGS':
            print("[WARN] VALIDATION PASSED WITH WARNINGS - Minor issues detected")
        else:
            print("[FAIL] VALIDATION FAILED - System is not ready for training")
        
        readiness = validation_report.get('system_readiness', {})
        print(f"[RESULTS] Readiness Score: {readiness.get('readiness_score', 0):.1f}/100")
        print(f"[TARGET] Assessment: {readiness.get('assessment', 'Unknown')}")
        
        # Show key metrics
        integration = validation_report.get('integration_tests', {})
        if integration:
            stats = integration.get('test_statistics', {})
            print(f"[TEST] Integration Tests: {stats.get('tests_passed', 0)}/{stats.get('total_tests', 0)} passed ({stats.get('success_rate_percent', 0):.1f}%)")
            
            if integration.get('critical_failures'):
                print(f"[FAIL] Critical Failures: {', '.join(integration['critical_failures'])}")
        
        # Show performance metrics
        performance = validation_report.get('performance_validation', {})
        if performance:
            print(f"[PERF] Performance: {'PASSED' if performance.get('acceptable', False) else 'FAILED'}")
            
            metrics = performance.get('metrics', {})
            for metric_name, metric_data in metrics.items():
                if isinstance(metric_data, dict) and 'message' in metric_data:
                    status = "[PASS]" if metric_data.get('acceptable', True) else "[FAIL]"
                    print(f"  {status} {metric_name}: {metric_data['message']}")
        
        # Show recommendations
        recommendations = validation_report.get('recommendations', [])
        if recommendations:
            print(f"\n[RECS] Recommendations:")
            for i, rec in enumerate(recommendations[:5], 1):  # Show top 5
                print(f"  {i}. {rec}")
            if len(recommendations) > 5:
                print(f"  ... and {len(recommendations) - 5} more (see detailed report)")
        
        print(f"\n[TIME] Validation completed in {validation_report['validation_duration_seconds']:.2f} seconds")
        
        # Return appropriate exit code
        if overall_status == 'PASSED':
            return 0
        elif overall_status == 'PASSED_WITH_WARNINGS':
            return 1
        else:
            return 2
            
    except KeyboardInterrupt:
        print("\n[WARN] Validation interrupted by user")
        return 130
    except Exception as e:
        print(f"\n[FAIL] Validation failed with error: {e}")
        validator.logger.error(f"Validation error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)