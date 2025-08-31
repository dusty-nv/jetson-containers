# How to Use Log Files for Debugging

This guide explains how to effectively use the log files generated during the build and test process to debug issues.

## 📁 Log File Structure

### Directory Organization
Logs are organized by timestamp and stage:
```
logs/
├── YYYYMMDD_HHMMSS/ # Timestamp-based session
│ ├── build/ # Build phase logs
│ │ ├── 01o1_package_name.sh
│ │ ├── 01o1_package_name.txt
│ │ ├── 02o2_package_name.sh
│ │ ├── 02o2_package_name.txt
│ │ └── ...
│ └── test/ # Test phase logs
│   ├── 03-1_package_name_test.sh.sh
│   ├── 03-1_package_name_test.sh.txt
│   ├── 05-1_package_name_test.py.sh
│   └── 05-1_package_name_test.py.txt
│
├── (other session)
```

### File Naming Convention

#### Build Logs
- **Format**: `{order}o{total}_{package_name}.txt`
- **Example**: `01o5_sudonim_r38.2.tegra-aarch64-cu130-24.04-sudonim.txt`
  - `01` = Build order (1st package)
  - `o5` = Out of 5 total packages
  - `sudonim_r38.2.tegra-aarch64-cu130-24.04-sudonim` = Package identifier

#### Test Logs
- **Format**: `{order}-{test_number}_{package_name}_test.sh.txt`
- **Example**: `01-1_sudonim_r38.2.tegra-aarch64-cu130-24.04-sudonim_test.sh.txt`
  - `01` = Test order (1st test)
  - `1` = Test number for this package
  - `sudonim_r38.2.tegra-aarch64-cu130-24.04-sudonim_test.sh` = Test script identifier

## 🔍 Understanding Phases and Stages

### Build Phase
- **Purpose**: Container image creation and package installation
- **Stages**: Each package build counts as one stage (tracked by `current_stage` in the code)
- **Logs**: Docker build output, package installation, dependency resolution
- **Key indicators**:
  - `#X [step/X] FROM/COPY/RUN... DONE`
  - Build errors, missing dependencies, compilation failures

### Test Phase
- **Purpose**: Validation of built packages and functionality testing
- **Stages**: Each package test counts as one stage (tracked by `current_stage` in the code)
- **Logs**: Test script execution, runtime errors, package behavior
- **Key indicators**:
  - Test script output
  - Runtime errors and stack traces
  - Package initialization results

### How Stages Work
In the jetson-containers build system, **stages** are individual steps within each phase:
- **Build stages**: Each package build increments the stage counter
- **Test stages**: Each package test increments the stage counter
- **Stage numbering**: Matches the `current_stage` variable in the `BuildTimer` class
- **Sequential progression**: Stages increment as packages are built and tested in order

## 🛠️ Best Practices for Debugging

### 1. Start with the Latest Session
```bash
# Find the most recent log directory
ls -t logs/ | head -1

# Navigate to the latest session
cd logs/$(ls -t logs/ | head -1)
```

### 2. Follow the Execution Flow
1. **Check build logs first** - Ensure packages built successfully
2. **Examine test logs** - Look for runtime failures
3. **Correlate errors** - Build issues often cause test failures

### 3. Use Log File Patterns
```bash
# Find all logs for a specific package
find . -name "*sudonim*" -type f

# Find failed builds (look for error patterns)
grep -r "ERROR\|FAILED\|failed" build/

# Find test failures
grep -r "Traceback\|Error\|Exception" test/
```

### 4. Compare with Working Sessions
```bash
# Compare current failing logs with previous working logs
diff logs/working_session/ logs/failing_session/
```

### 5. Focus on Key Error Messages
- **Build errors**: Look for `ERROR`, `failed`, `exit status`
- **Test errors**: Look for `Traceback`, `AttributeError`, `ImportError`
- **System errors**: Look for `No such file`, `Permission denied`

## 🚨 Common Pitfalls

### 1. Not Using --no-cache
- **Problem**: Docker layer caching masks real issues
- **Solution**: Always use `--build-flags="--no-cache"` when debugging build issues

## 🐛 Debugging Workflow

### Step 1: Identify the Issue
```bash
# Check the latest session
cd logs/$(ls -t logs/ | head -1)

# Look for obvious errors
grep -r "ERROR\|FAILED\|failed" .
```

### Step 2: Examine Build Logs
```bash
# Check if packages built successfully
ls build/
cat build/*.txt | grep -A5 -B5 "ERROR\|failed"
```

### Step 3: Examine Test Logs
```bash
# Check test execution
ls test/
cat test/*.txt | grep -A10 -B5 "Traceback\|Error"
```

### Step 4: Correlate Issues
- **Build success + Test failure**: Runtime or dependency issue
- **Build failure**: Dependency, compilation, or configuration issue
- **Both fail**: Systemic issue affecting the entire build process

### Step 5: Compare with Working State
```bash
# Find a working session to compare against
# Look for differences in build configuration, dependencies, or system state
```

## 📊 Example Debugging Session

### Scenario: Sudonim CUDA Test Failing

1. **Check build logs**: `01o5_sudonim_*.txt`
   - ✅ Build completed successfully
   - ✅ No compilation errors

2. **Check test logs**: `01-1_sudonim_*_test.sh.txt`
   - ❌ `cudaDeviceQuery() failed: 'NoneType' object has no attribute 'decode'`
   - ❌ CUDA initialization failure

3. **Analysis**:
   - Build works but test fails
   - CUDA runtime issue, not build issue
   - Problem likely in CUDA library resolution or linking

4. **Investigation**:
   - Check CUDA package configuration
   - Verify library paths and dependencies
   - Compare with working commits

## 💡 Pro Tips

- **Use `less` or `vim`** for large log files to search efficiently
- **Create aliases** for common log navigation commands
- **Keep working logs** for comparison when debugging regressions
- **Document patterns** you find useful for future debugging sessions
