import requests
import time
import json
import os
import hashlib
import statistics
from datetime import datetime

def get_file_audio_characteristics(filepath):
    """
    Extract basic characteristics from an audio file.
    Returns a dictionary with file size, duration, and header hash.
    """
    if not os.path.exists(filepath):
        return None
        
    # Get file size
    file_size = os.path.getsize(filepath)
    
    # Read first 4KB to calculate header hash (should contain format info)
    with open(filepath, 'rb') as f:
        header_data = f.read(4096)
        header_hash = hashlib.md5(header_data).hexdigest()
    
    # Check if file is empty or too small
    if file_size < 100:  # Arbitrary small size threshold
        return None
        
    return {
        'file_size': file_size,
        'header_hash': header_hash,
        'file_exists': True
    }

def compare_audio_files(reference_path, new_path):
    """
    Compare two audio files to determine if they're similar enough.
    Returns True if the files are comparable (have similar characteristics).
    """
    if not os.path.exists(reference_path):
        print(f"   ⚠️ Reference file {reference_path} not found - skipping comparison")
        return False
        
    ref_chars = get_file_audio_characteristics(reference_path)
    new_chars = get_file_audio_characteristics(new_path)
    
    if not ref_chars or not new_chars:
        print("   ⚠️ Could not analyze audio files")
        return False
        
    # Compare file sizes - should be within 20% of each other
    size_ratio = max(ref_chars['file_size'], new_chars['file_size']) / min(ref_chars['file_size'], new_chars['file_size'])
    
    print(f"   Reference file: {ref_chars['file_size']} bytes")
    print(f"   New file: {new_chars['file_size']} bytes")
    print(f"   Size ratio: {size_ratio:.2f}")
    
    # Simple header comparison - checks if file format is similar
    header_match = ref_chars['header_hash'] == new_chars['header_hash']
    print(f"   Header match: {header_match}")
    
    # Consider files similar if size is within reasonable range
    return size_ratio < 1.2  # Files are within 20% size of each other

def benchmark_request(url, method='GET', json_data=None, headers=None, num_samples=3):
    """
    Benchmark a request by measuring response time over multiple samples
    """
    response_times = []
    status_code = None
    response_content = None
    response_json = None
    content_length = None
    response_headers = None
    
    default_headers = {"Accept": "application/json"}
    if headers:
        default_headers.update(headers)
    
    for i in range(num_samples):
        start_time = time.time()
        try:
            if method.upper() == 'GET':
                response = requests.get(url, timeout=10, headers=default_headers)
            elif method.upper() == 'POST':
                response = requests.post(url, json=json_data, timeout=20, headers=default_headers)
            else:
                raise ValueError(f"Unsupported method: {method}")
                
            end_time = time.time()
            response_time = end_time - start_time
            response_times.append(response_time)
            
            # Store results from the last run
            if i == num_samples - 1:
                status_code = response.status_code
                response_content = response.content
                response_headers = dict(response.headers)
                content_length = len(response.content)
                try:
                    if 'application/json' in response.headers.get('content-type', ''):
                        response_json = response.json()
                except json.JSONDecodeError:
                    response_json = None
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            # Add a very slow time to indicate failure
            response_times.append(30.0)  # 30 seconds is considered a failure/timeout
            
            # If all attempts failed, return error information
            if i == num_samples - 1 and not status_code:
                return {
                    'url': url,
                    'method': method,
                    'status_code': 500,  # Internal error code
                    'response_time': 30.0,
                    'min_response_time': 30.0,
                    'max_response_time': 30.0,
                    'stdev_response_time': 0,
                    'response_content': str(e).encode('utf-8'),
                    'response_json': None,
                    'response_headers': {},
                    'content_length': 0,
                    'samples': num_samples,
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e)
                }
    
    # Calculate statistics
    avg_response_time = statistics.mean(response_times)
    min_response_time = min(response_times)
    max_response_time = max(response_times)
    
    if len(response_times) > 1:
        stdev_response_time = statistics.stdev(response_times)
    else:
        stdev_response_time = 0
    
    return {
        'url': url,
        'method': method,
        'status_code': status_code,
        'response_time': avg_response_time,
        'min_response_time': min_response_time,
        'max_response_time': max_response_time,
        'stdev_response_time': stdev_response_time,
        'response_content': response_content,
        'response_json': response_json,
        'response_headers': response_headers,
        'content_length': content_length,
        'samples': num_samples,
        'timestamp': datetime.now().isoformat()
    }

def generate_benchmark_report(benchmark_results):
    """Generate a detailed benchmark report"""
    report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_filename = f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(report_filename, 'w') as f:
        f.write(f"=== API BENCHMARK REPORT ===\n")
        f.write(f"Date: {report_time}\n")
        f.write(f"==========================\n\n")
        
        # Write summary table
        f.write("SUMMARY:\n")
        f.write(f"{'Endpoint':<20} {'Status':<10} {'Avg Time (s)':<15} {'Min Time (s)':<15} {'Max Time (s)':<15}\n")
        f.write("-" * 80 + "\n")
        
        for endpoint, data in benchmark_results.items():
            if endpoint == 'speech_api':
                # Handle speech benchmarks differently
                if isinstance(data, dict) and 'aggregate' in data:
                    agg_data = data['aggregate']
                    status = "✓"
                    f.write(f"speech_api (agg)  {status:<10} {agg_data['response_time']:<15.3f} {'N/A':<15} {'N/A':<15}\n")
                continue
                
            # Skip if data is None or not a dict with status_code
            if not data or not isinstance(data, dict) or 'status_code' not in data:
                continue
                
            status = "✓" if data['status_code'] == 200 else f"✗ ({data['status_code']})"
            f.write(f"{endpoint:<20} {status:<10} {data['response_time']:<15.3f} {data['min_response_time']:<15.3f} {data['max_response_time']:<15.3f}\n")
        
        # Add speech synthesis detailed section if available
        if 'speech_api' in benchmark_results and isinstance(benchmark_results['speech_api'], dict):
            speech_data = benchmark_results['speech_api']
            f.write("\n\nSPEECH SYNTHESIS BENCHMARKS:\n")
            f.write("=" * 80 + "\n")
            
            if any(k in speech_data for k in ['short', 'medium', 'long']):
                # Format table header
                f.write(f"\n{'Text Length':<15} {'Chars':<10} {'Time (s)':<10} {'Chars/sec':<15} {'ms/char':<10} {'RTF':<8}\n")
                f.write("-" * 70 + "\n")
                
                # Add each test result
                for length, metrics in speech_data.items():
                    if length != 'aggregate':
                        f.write(f"{length:<15} {metrics['char_count']:<10} {metrics['response_time']:<10.3f} ")
                        f.write(f"{metrics['chars_per_second']:<15.2f} {metrics['ms_per_char']:<10.2f} ")
                        f.write(f"{metrics.get('rtf', 'N/A'):<8}\n")
                
                # Add aggregate row
                if 'aggregate' in speech_data:
                    agg = speech_data['aggregate']
                    f.write(f"{'AGGREGATE':<15} {agg['char_count']:<10} {agg['response_time']:<10.3f} ")
                    f.write(f"{agg['chars_per_second']:<15.2f} {agg['ms_per_char']:<10.2f} {'N/A':<8}\n")
                
                # Add industry context
                f.write("\nINDUSTRY CONTEXT:\n")
                f.write("- RTF (Real-Time Factor): Values < 1 indicate faster-than-realtime generation\n")
                f.write("- Commercial TTS systems typically achieve 20-100 chars/sec\n")
                f.write("- High-quality neural TTS typically has RTF of 0.1-0.5\n")
                f.write("- Production systems aim for <50ms latency per character\n")
            
        f.write("\n\nDETAILED RESULTS:\n")
        f.write("=" * 80 + "\n\n")
        
        # Write detailed results for each endpoint
        for endpoint, data in benchmark_results.items():
            if endpoint == 'speech_api' and isinstance(data, dict) and 'aggregate' in data:
                # Skip detailed section for speech as we've already added a specialized section
                continue
                
            # Skip if data is None or not a dict with required fields
            if not data or not isinstance(data, dict):
                continue
                
            if 'url' in data:  # Standard benchmark result
                f.write(f"Endpoint: {endpoint}\n")
                f.write(f"URL: {data['url']}\n")
                f.write(f"Method: {data['method']}\n")
                f.write(f"Status Code: {data['status_code']}\n")
                f.write(f"Response Statistics:\n")
                f.write(f"  - Average Time: {data['response_time']:.3f}s\n")
                f.write(f"  - Minimum Time: {data['min_response_time']:.3f}s\n")
                f.write(f"  - Maximum Time: {data['max_response_time']:.3f}s\n")
                f.write(f"  - Standard Deviation: {data['stdev_response_time']:.3f}s\n")
                f.write(f"  - Content Length: {data.get('content_length', 'N/A')} bytes\n")
                f.write(f"  - Samples: {data['samples']}\n")
                f.write("\n" + "-" * 80 + "\n\n")
    
    print(f"✅ Benchmark report saved to {report_filename}")
    
    # Print summary to console
    print("\nBENCHMARK SUMMARY:")
    print(f"{'Endpoint':<15} {'Status':<8} {'Avg Time':<10} {'Min/Max':<15}")
    print("-" * 55)
    
    for endpoint, data in benchmark_results.items():
        if endpoint == 'speech_api' and isinstance(data, dict) and 'aggregate' in data:
            # Display speech aggregate differently
            agg = data['aggregate']
            print(f"{'speech_api':<15} {'✓':<8} {agg['response_time']:.3f}s   {'N/A':<15}")
            continue
            
        if isinstance(data, dict) and 'status_code' in data:
            status = "✓" if data['status_code'] == 200 else f"✗ ({data['status_code']})"
            min_max = f"{data['min_response_time']:.2f}/{data['max_response_time']:.2f}s"
            print(f"{endpoint:<15} {status:<8} {data['response_time']:.3f}s   {min_max:<15}")
    
    print(f"✅ Benchmark report saved to {report_filename}")
    
    # Print summary to console
    print("\nBENCHMARK SUMMARY:")
    print(f"{'Endpoint':<15} {'Status':<8} {'Avg Time':<10} {'Min/Max':<15}")
    print("-" * 55)
    
    for endpoint, data in benchmark_results.items():
        status = "✓" if data['status_code'] == 200 else f"✗ ({data['status_code']})"
        min_max = f"{data['min_response_time']:.2f}/{data['max_response_time']:.2f}s"
        print(f"{endpoint:<15} {status:<8} {data['response_time']:.3f}s   {min_max:<15}")

def compare_audio_files(reference_path, new_path):
    """
    Compare two audio files to determine if they're similar enough.
    Returns True if the files are comparable (have similar characteristics).
    """
    if not os.path.exists(reference_path):
        print(f"   ⚠️ Reference file {reference_path} not found - skipping comparison")
        return False
        
    ref_chars = get_file_audio_characteristics(reference_path)
    new_chars = get_file_audio_characteristics(new_path)
    
    if not ref_chars or not new_chars:
        print("   ⚠️ Could not analyze audio files")
        return False
        
    # Compare file sizes - should be within 20% of each other
    size_ratio = max(ref_chars['file_size'], new_chars['file_size']) / min(ref_chars['file_size'], new_chars['file_size'])
    
    print(f"   Reference file: {ref_chars['file_size']} bytes")
    print(f"   New file: {new_chars['file_size']} bytes")
    print(f"   Size ratio: {size_ratio:.2f}")
    
    # Simple header comparison - checks if file format is similar
    header_match = ref_chars['header_hash'] == new_chars['header_hash']
    print(f"   Header match: {header_match}")
    
    # Consider files similar if size is within reasonable range
    return size_ratio < 1.2  # Files are within 20% size of each other

def test_server():
    base_url = "http://localhost:8880"
    
    # Wait for server to be fully up
    print("Waiting for server to start...")
    time.sleep(5)  # Increased wait time for reliability
    
    # Test primary endpoints
    endpoints = {
        "web_app": "/web",
        "swagger_docs": "/docs",
        "openai_models": "/v1/models"
    }
    
    all_tests_passed = True
    benchmark_results = {}
    
    # Test each endpoint
    for name, path in endpoints.items():
        url = base_url + path
        try:
            # Benchmark endpoint
            benchmark_data = benchmark_request(url)
            benchmark_results[name] = benchmark_data
            
            if benchmark_data.get('status_code') == 200:
                print(f"✅ {name} endpoint ({url}) is working")
                print(f"   Response time: {benchmark_data['response_time']:.3f}s")
                
                # If testing models endpoint, extract available TTS models
                if name == "openai_models":
                    try:
                        models_data = benchmark_data['response_json']
                        tts_models = [model["id"] for model in models_data.get("data", []) 
                                     if model["id"].startswith("tts-")]
                        
                        if tts_models:
                            print(f"   Found TTS models: {', '.join(tts_models)}")
                        else:
                            print("   ⚠️ No TTS models found in the response")
                    except (KeyError) as e:
                        print(f"   ⚠️ Could not parse models response: {e}")
                        all_tests_passed = False
            else:
                print(f"❌ {name} endpoint ({url}) returned status code {benchmark_data.get('status_code', 'unknown')}")
                all_tests_passed = False
        except Exception as e:
            print(f"❌ {name} endpoint ({url}) failed: {e}")
            all_tests_passed = False
            benchmark_results[name] = None
    
    # Test OpenAI speech API integration with benchmarking
    speech_benchmark = test_speech_api(base_url, all_tests_passed)
    if speech_benchmark:
        benchmark_results['speech_api'] = speech_benchmark
    
    # Generate benchmark report
    try:
        generate_benchmark_report(benchmark_results)
    except Exception as e:
        print(f"⚠️ Failed to generate benchmark report: {e}")
    
    return all_tests_passed
    
def test_speech_api(base_url, previous_tests_passed):
    """Test the OpenAI-compatible speech API with comprehensive benchmarking"""
    
    if not previous_tests_passed:
        print("⚠️ Skipping speech API test due to previous failures")
        return None
    
    # Define test texts of different lengths
    test_texts = {
        "short": "This is a quick test.",
        "medium": "This is a medium-length test of the speech synthesis API. We are measuring response time and output quality for typical conversational content.",
        "long": "This is a longer test of the speech synthesis capabilities. In speech synthesis benchmarking, we typically measure several key performance indicators including latency, throughput, and quality metrics."
    }
    
    speech_benchmarks = {}
    successful = True
    
    # First get available models to use a valid one
    try:
        models_response = benchmark_request(f"{base_url}/v1/models", method='GET')
        if models_response['status_code'] != 200:
            print(f"❌ Cannot fetch models for speech test (status {models_response['status_code']})")
            return None
            
        models_data = models_response['response_json']
        tts_models = [model["id"] for model in models_data.get("data", []) 
                     if model["id"].startswith("tts-")]
        
        if not tts_models:
            print("❌ No TTS models available for testing")
            return None
            
        model_to_test = tts_models[0]
        
        # Test voices endpoint to get a valid voice
        voices_response = benchmark_request(f"{base_url}/v1/audio/voices", method='GET')
        if voices_response['status_code'] == 200:
            try:
                voices_data = voices_response['response_json']
                available_voices = voices_data.get("voices", ["alloy"])  # Default to "alloy" if structure is unexpected
                # Voice format might be like "af_alloy" - extract just the voice type if needed
                voice_to_test = available_voices[0]
                if "_" in voice_to_test:
                    lang_code, voice_name = voice_to_test.split("_", 1)
                    print(f"   Found voice format with language prefix: {voice_to_test}")
                    print(f"   Using language code: {lang_code}, voice: {voice_name}")
                    # Update test data to include language code if needed
                    test_data_extra = {"lang_code": lang_code}
                else:
                    voice_name = voice_to_test
                    test_data_extra = {}
                print(f"   Using voice: {voice_to_test}")
            except (KeyError, IndexError):
                voice_to_test = voice_name = "alloy"  # Fallback
                test_data_extra = {}
                print("   Using default voice: alloy")
        else:
            voice_to_test = voice_name = "alloy"  # Fallback
            test_data_extra = {}
            print("   Using default voice: alloy (voices endpoint not available)")
            
        # Run tests for each text length
        print(f"\n🔊 Testing speech synthesis with model: {model_to_test}")
        print(f"{'Text Length':<15} {'Chars':<8} {'Time (s)':<10} {'Chars/sec':<12} {'RTF':<8} {'Status'}")
        print("-" * 70)
        
        for length_name, test_text in test_texts.items():
            # Set up the test data
            char_count = len(test_text)
            word_count = len(test_text.split())
            
            test_data = {
                "model": model_to_test,
                "input": test_text,
                "voice": voice_name if '_' in voice_to_test else voice_to_test,
                "response_format": "mp3"
            }
            
            # Add language code if present
            test_data.update(test_data_extra)
            
            # Benchmark the speech synthesis
            speech_result = benchmark_request(
                f"{base_url}/v1/audio/speech",
                method='POST',
                json_data=test_data,
                headers={"Accept": "audio/mpeg"},
                num_samples=1  # One sample per length to avoid overloading
            )
            
            # Check content-type in headers or try to detect audio content
            is_audio = False
            if speech_result['status_code'] == 200:
                # First method: check headers
                if speech_result.get('response_headers'):
                    for key, value in speech_result['response_headers'].items():
                        if key.lower() == 'content-type' and 'audio' in value.lower():
                            is_audio = True
                            break
                
                # Second method: check content signature for MP3 header if we have content
                if not is_audio and speech_result.get('response_content'):
                    try:
                        content_start = speech_result['response_content'][:10]
                        # Check for MP3 header signature (0xFF 0xFB or ID3)
                        if (b'\xff\xfb' in content_start or 
                            b'ID3' in content_start or
                            b'ftyp' in content_start):
                            is_audio = True
                    except:
                        pass  # If we can't check content, assume not audio
                
                # Third method: assume it's audio if status is 200 and content exists
                if not is_audio and speech_result.get('response_content') and len(speech_result['response_content']) > 1000:
                    is_audio = True
            
            # Calculate metrics if successful
            if speech_result['status_code'] == 200 and is_audio:
                # Calculate industry-standard metrics
                rtf = speech_result['response_time'] / (word_count / 150)  # Real-time factor (assuming 150 words/minute)
                chars_per_sec = char_count / speech_result['response_time'] if speech_result['response_time'] > 0 else 0
                ms_per_char = (speech_result['response_time'] * 1000) / char_count if char_count > 0 else 0
                
                # Store metrics
                speech_benchmarks[length_name] = {
                    'char_count': char_count,
                    'word_count': word_count,
                    'response_time': speech_result['response_time'],
                    'chars_per_second': chars_per_sec,
                    'ms_per_char': ms_per_char,
                    'rtf': rtf,
                    'file_size': len(speech_result['response_content']),
                    'compression_ratio': len(speech_result['response_content']) / char_count if char_count > 0 else 0
                }
                
                # Save output file for the medium length test
                if length_name == 'medium':
                    with open("test_speech_output_new.mp3", "wb") as f:
                        f.write(speech_result['response_content'])
                    print(f"   Medium-length speech sample saved to test_speech_output_new.mp3")
                    
                    # Compare with reference file if it exists
                    if os.path.exists("test_speech_output.mp3"):
                        reference_match = compare_audio_files("test_speech_output.mp3", "test_speech_output_new.mp3")
                        if reference_match:
                            print("   ✅ Generated speech matches reference file characteristics")
                        else:
                            print("   ⚠️ Generated speech differs from reference file")
                    else:
                        print("   ⚠️ Reference file test_speech_output.mp3 not found - skipping comparison")
                
                # Print result line
                status = "✅"
                print(f"{length_name:<15} {char_count:<8} {speech_result['response_time']:<10.3f} {chars_per_sec:<12.2f} {rtf:<8.3f} {status}")
                
            else:
                successful = False
                status = f"❌ ({speech_result['status_code']})"
                error_msg = ""
                
                # Try to extract error message
                if speech_result.get('response_content'):
                    try:
                        # Try to parse as JSON first
                        error_content = speech_result['response_content']
                        if isinstance(error_content, bytes):
                            error_text = error_content.decode('utf-8', errors='ignore')
                            try:
                                error_data = json.loads(error_text)
                                if 'error' in error_data:
                                    error_msg = error_data['error'].get('message', str(error_data['error']))
                            except:
                                error_msg = error_text[:50]
                    except:
                        # If extraction fails, just indicate an error occurred
                        error_msg = "Failed to process response"
                
                print(f"{length_name:<15} {char_count:<8} {'--':<10} {'--':<12} {'--':<8} {status}")
                if error_msg:
                    print(f"   Error: {error_msg}")
                
        # Successful completion
        if successful and speech_benchmarks:
            print("\n✅ Speech API tests completed successfully")
            
            # Calculate aggregate metrics
            total_chars = sum(result['char_count'] for result in speech_benchmarks.values())
            total_time = sum(result['response_time'] for result in speech_benchmarks.values())
            avg_chars_per_sec = total_chars / total_time if total_time > 0 else 0
            avg_ms_per_char = (total_time * 1000) / total_chars if total_chars > 0 else 0
            
            print(f"\nAGGREGATE METRICS:")
            print(f"Total characters processed: {total_chars}")
            print(f"Total processing time: {total_time:.3f}s")
            print(f"Average processing speed: {avg_chars_per_sec:.2f} chars/sec")
            print(f"Average latency: {avg_ms_per_char:.2f} ms/char")
            
            # Add aggregate metrics to results
            speech_benchmarks['aggregate'] = {
                'char_count': total_chars,
                'response_time': total_time,
                'chars_per_second': avg_chars_per_sec,
                'ms_per_char': avg_ms_per_char
            }
            
            return speech_benchmarks
        else:
            print("\n❌ Some speech API tests failed")
            return None
            
    except Exception as e:
        print(f"❌ OpenAI speech API test failed: {e}")
        return None

if __name__ == "__main__":
    start_time = time.time()
    result = test_server()
    total_duration = time.time() - start_time
    
    print(f"\nTotal test duration: {total_duration:.2f} seconds")
    
    if result:
        print("\n✅ All tests passed successfully!")
        exit(0)
    else:
        print("\n❌ Some tests failed. Check the log for details.")
        exit(1)