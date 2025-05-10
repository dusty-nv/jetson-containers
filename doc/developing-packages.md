## Prioritized Recommendations for Implementing Jetson Containers

**Top Priority**

1.  **Use Pre-built Wheels:** Configure pip to utilize the Jetson AI Lab PyPI mirror (`pypi.jetson-ai-lab.dev`) for significantly faster and more reliable builds.
    * **Example:** In your Dockerfile, ensure your pip install commands point to the mirror:
        ```dockerfile
        RUN pip install --index-url https://pypi.jetson-ai-lab.dev/simple/ some_package
        ```

2.  **Define Clear Dependencies:** Accurately specify package dependencies (`depends`) and environment constraints (`requires`) in your `config.py` to guarantee the correct build order and compatibility.
    * **Example (in `config.py`):**
        ```python
        package['depends'] = ['python', 'numpy']
        package['requires'] = ['cuda>=11.0', 'cudnn>=8.0']
        ```

3.  **Create Thorough Testing:** Implement a `test.sh` script to automatically verify the package's functionality after installation.
    * **Example (`test.sh`):**
        ```bash
        #!/bin/bash
        python -c "import your_package; print('Package imported successfully')"
        ```

**High Priority**

4.  **Resource Management:** When compiling from source on Jetson's limited resources, restrict parallel jobs and use architecture-specific compiler flags.
    * **Example (in build script or `config.py`):**
        ```bash
        cmake -j$(nproc --all) ... # Avoid using all cores
        ```
        or in `config.py`:
        ```python
        package['cmake_args'] = ['-DCMAKE_CUDA_ARCHITECTURES=87']
        ```

5.  **Pin Dependencies:** Specify exact versions for packages in `requirements.txt` to prevent unexpected issues due to updates.
    * **Example (`requirements.txt`):**
        ```
        numpy>=1.23.0
        torch>=1.13.1
        ```

6.  **Documentation:** Provide a comprehensive `docs.md` file with usage instructions, configuration tips, and common troubleshooting steps specific to your package.
    * **Example (`docs.md` - excerpt):**
        ```markdown
        ## Usage

        To run this container, use the following command:

        ```bash
        docker run --runtime nvidia -it your_image:latest ...
        ```

        ## Configuration

        You can configure the application via environment variables...
        ```

**Medium Priority**

7.  **Error Handling:** Use robust error handling in build scripts (e.g., `set -e` in bash) to ensure failures are detected and reported clearly.
    * **Example (in build script):**
        ```bash
        #!/bin/bash
        set -e
        cmake ...
        make -j$(nproc)
        make install
        ```

8.  **Cleanup:** Remove unnecessary build dependencies, source code directories, and package caches to minimize the final container image size.
    * **Example (in Dockerfile):**
        ```dockerfile
        RUN apt-get update && apt-get install -y --no-install-recommends some-dev-package && \
            ... && \
            apt-get remove -y some-dev-package && \
            apt-get autoremove -y && \
            rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
        ```

9.  **Version Variants:** Define package variants in `config.py` if multiple versions are available, including logic to select appropriate defaults based on the environment.
    * **Example (in `config.py` - simplified):**
        ```python
        package['variants'] = {
            '1.0': {'src': 'url/to/version1.0.tar.gz'},
            '1.1': {'src': 'url/to/version1.1.tar.gz', 'requires': ['some_newer_dep']}
        }
        package['default'] = '1.1' if get_cuda_version() >= (11,0) else '1.0'
        ```

**Lower Priority**

10. **Runtime Configuration:** Create a `run.sh` script that defines the default behavior of the container when it's started.
    * **Example (`run.sh`):**
        ```bash
        #!/bin/bash
        python /app/main.py "$@"
        ```

11. **Code Formatting:** Ensure Python code adheres to basic style guidelines (e.g., PEP 8) before submission.

12. **Dependency Analysis:** Consider using tools like `pip-tools` or `pipdeptree` to analyze and resolve potential dependency conflicts.
