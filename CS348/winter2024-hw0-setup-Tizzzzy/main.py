import os
import requests
import sys


def get_latest_python_version():
    '''
    Fetches the latest stable Python version, though isn't MacOS- or Windows-specific.
    Also, the URL used by the get() function may be stale, so use this function optionally.
    '''
    try:
        response = requests.get("https://endoflife.date/api/python.json")

        # Check directly for status code 200
        if response.status_code == 200:
            parsed_result = response.json()
            version_string = parsed_result[0]["latest"]
            return version_string
        else:
            return 'not found'

    except requests.exceptions.RequestException as e:
        # This will catch any request-related errors (including network issues)
        print(f'\nWeb query for latest Python version failed: {e}')
        return 'not found'

def get_installed_python_version():
    return sys.version.split()[0]

def is_virtual_environment():
    '''
    Checks if Python is installed within a virtual environment.
    '''
    # return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    return (hasattr(sys, 'real_prefix') or 
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or
            'CONDA_PREFIX' in os.environ)

def get_virtual_env_info():
    if is_virtual_environment():
        return os.path.basename(sys.prefix), sys.prefix
    return None, None

def main():
    # Get the latest and installed Python version, and also get the virtual environment name and path.
    latest_version = get_latest_python_version()
    installed_version = get_installed_python_version()
    env_name, env_path = get_virtual_env_info()

    setup_config_output = f'The latest version of Python is {latest_version}, and you have ' + ('the same' if installed_version == latest_version else f'version {installed_version}') + ' installed.'

    if env_name:
        setup_config_output += f'\nPython is running in a virtual environment called {env_name}, located at {env_path}.'
    else:
        setup_config_output += '\nPython is not running in a virtual environment.'

    print('\n' + setup_config_output)
    with open('setup_config.txt', 'w') as file:
        file.write(setup_config_output)


if __name__ == "__main__":
    main()
